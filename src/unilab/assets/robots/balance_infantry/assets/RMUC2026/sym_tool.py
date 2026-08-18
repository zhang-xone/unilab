from PIL import Image
import numpy as np

# =========配置========
INPUT_NAME  = "hfield_sym"
OUTPUT_NAME = "hfield_fixed.png"
HEIGHT_OFFSET = 10    # 右侧统一抬高 10
# ======================

im = Image.open(INPUT_NAME)
data = np.array(im, dtype=np.int16)
H, W = data.shape
half_w = W // 2
out = data.copy()

# 左半边 → 中心对称填充对角
for y in range(H):
    for x in range(half_w):
        mx = W - 1 - x
        my = H - 1 - y
        val = data[y, x] + HEIGHT_OFFSET
        val = np.clip(val, 0, 255)
        out[my, mx] = val

out_img = Image.fromarray(out.astype(np.uint8))
out_img.save(OUTPUT_NAME)
print("✅ 修正完成，输出", OUTPUT_NAME)

