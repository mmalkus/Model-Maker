"""Find the fast-forward stretches of raw.webm (where the purple
"Fast-forward" badge is on screen), compress each to a few seconds, keep the
interactive parts at real speed, and write demo.mp4."""
import subprocess
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "raw.webm"
TARGET = 8.0  # seconds each fast-forward stretch lasts in the final video
FPS = 4  # badge-detection sampling rate
# Badge area: centred at x=800, top 76px, ~40px tall (see record.js #demo-ff).
CROP = (560, 82, 480, 24)  # x, y, w, h

dur = float(subprocess.check_output(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", SRC]).decode())

x, y, w, h = CROP
raw = subprocess.check_output(
    ["ffmpeg", "-v", "error", "-i", SRC, "-vf", f"fps={FPS},crop={w}:{h}:{x}:{y}",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
frame = w * h * 3
flags = []
for i in range(len(raw) // frame):
    f = raw[i * frame:(i + 1) * frame]
    purple = sum(1 for j in range(0, len(f), 3 * 7)
                 if abs(f[j] - 0x7C) < 30 and abs(f[j + 1] - 0x3A) < 30 and abs(f[j + 2] - 0xED) < 30)
    flags.append(purple > 0.15 * (len(f) / (3 * 7)))

segs, start = [], None
for i, on in enumerate(flags + [False]):
    t = i / FPS
    if on and start is None:
        start = t
    elif not on and start is not None:
        if t - start > TARGET * 1.5:
            segs.append((start, t))
        start = None
print("fast-forward segments:", [(round(a, 1), round(b, 1)) for a, b in segs])

parts, t = [], 0.0
for a, b in segs:
    if a > t:
        parts.append((t, a, 1.0))
    parts.append((a, b, (b - a) / TARGET))
    t = b
parts.append((t, dur, 1.0))

names = []
for i, (a, b, speed) in enumerate(parts):
    print(f"{a:7.1f}-{b:7.1f}s  x{speed:.1f}")
    name = f"part{i}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", SRC,
                    "-vf", f"setpts=(PTS-STARTPTS)/{speed:.4f},fps=30", "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "medium", "-threads", "2", name], check=True)
    names.append(name)
open("parts.txt", "w").write("".join(f"file '{n}'\n" for n in names))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "parts.txt", "-c", "copy",
                "-movflags", "+faststart", "demo.mp4"], check=True)
