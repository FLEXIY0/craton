"""Удаление фона из видео (руки, коробок, спичка) через BiRefNet на GPU.

Установка (Windows, NVIDIA GPU, Python 3.10-3.12):
    pip install "rembg[gpu]" opencv-python numpy pillow imageio-ffmpeg
Запуск:
    python remove_bg_video.py input.mp4
Результат в папке out/:
    result_alpha.webm  - VP9 с прозрачным фоном (браузер, CapCut, DaVinci)
    result_alpha.mov   - ProRes 4444 с прозрачным фоном (Premiere, After Effects)
    result_green.mp4   - на зелёном фоне (хромакей, работает везде)
    png/               - кадры PNG с прозрачностью
"""
import glob
import os
import subprocess
import sys

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image
from rembg import new_session, remove

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MODEL = "birefnet-general"  # ~930 MB, скачается сам при первом запуске

src = sys.argv[1]
out = "out"
frames_dir = os.path.join(out, "frames")
png_dir = os.path.join(out, "png")
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

# 1. видео -> кадры без потерь
subprocess.run([FFMPEG, "-y", "-v", "error", "-i", src,
                os.path.join(frames_dir, "%04d.png")], check=True)

# fps исходника
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.release()

# 2. маски + чистка + сборка RGBA
session = new_session(MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
prev = None
for i, p in enumerate(files, 1):
    name = os.path.basename(p)
    frame = cv2.imread(p)
    m = np.asarray(remove(Image.open(p).convert("RGB"), session=session,
                          only_mask=True), dtype=np.float32) / 255.0
    # подтянуть контраст маски, оставить тонкую растушёвку по краю
    m = np.clip((m - 0.35) / 0.4, 0, 1)
    hard = (m > 0.5).astype(np.uint8)
    # убрать мелкие остатки фона (линии двери, ручка, тени)
    k, lab, st, _ = cv2.connectedComponentsWithStats(hard, 8)
    keep = np.zeros_like(hard)
    if k > 1:
        big = st[1:, cv2.CC_STAT_AREA].max()
        for j in range(1, k):
            if st[j, cv2.CC_STAT_AREA] >= max(300, 0.08 * big):
                keep[lab == j] = 1
    keep = cv2.dilate(keep, np.ones((9, 9), np.uint8))
    m = m * keep
    # лёгкое сглаживание во времени против мерцания краёв
    if prev is not None:
        m = 0.75 * m + 0.25 * prev
    prev = m
    alpha = (np.clip(m, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(png_dir, name), np.dstack([frame, alpha]))
    print(f"{i}/{len(files)}", end="\r")
print()

# 3. сборка видео (со звуком исходника, если он есть)
pat = os.path.join(png_dir, "%04d.png")
base = [FFMPEG, "-y", "-v", "error", "-framerate", str(fps), "-i", pat,
        "-i", src, "-map", "0:v", "-map", "1:a?", "-shortest"]
subprocess.run(base + ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0",
                       "-crf", "18", "-auto-alt-ref", "0", "-c:a", "libopus",
                       os.path.join(out, "result_alpha.webm")], check=True)
subprocess.run(base + ["-c:v", "prores_ks", "-profile:v", "4444",
                       "-pix_fmt", "yuva444p10le", "-c:a", "pcm_s16le",
                       os.path.join(out, "result_alpha.mov")], check=True)
subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
                f"color=c=0x00b140:s={frame.shape[1]}x{frame.shape[0]}:r={fps}",
                "-framerate", str(fps), "-i", pat, "-i", src,
                "-filter_complex", "[0][1]overlay=shortest=1,format=yuv420p[v]",
                "-map", "[v]", "-map", "2:a?", "-shortest",
                "-c:v", "libx264", "-crf", "12", "-c:a", "aac",
                os.path.join(out, "result_green.mp4")], check=True)
print("Готово, файлы в папке", out)
