import os
from glob import glob
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

src_dir = "data/chessboard"
dst_dir = "data/chessboard_jpg"
os.makedirs(dst_dir, exist_ok=True)

files = sorted(glob(os.path.join(src_dir, "*.*")))
print("found:", len(files))

ok = 0
bad = 0
for fn in files:
    try:
        im = Image.open(fn).convert("RGB")
        base = os.path.splitext(os.path.basename(fn))[0]
        out = os.path.join(dst_dir, base + ".jpg")
        im.save(out, "JPEG", quality=95)
        ok += 1
    except Exception as e:
        bad += 1
        print("failed:", fn, "err:", e)

print("converted:", ok, "failed:", bad)