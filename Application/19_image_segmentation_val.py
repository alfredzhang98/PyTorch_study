import os
import torch
import numpy as np
from PIL import Image

# --------------------
# 基础配置
# --------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, 'ckpts/work2/unet_epoch_99_265.pth')  # 整模型保存
IMG_PATH   = os.path.join(_BASE_DIR, 'data/work2/JPEGImages/21.jpg')
SAVE_PATH  = os.path.join(_BASE_DIR, 'output.png')
IMG_SIZE   = (256, 256)   # 需与训练时一致（最好被 16 整除）

# --------------------
# 加载模型（先 CPU，避免反序列化阶段占显存）
# --------------------
cpu_device = torch.device('cpu')
unet = torch.load(MODEL_PATH, map_location=cpu_device)  # 强制先落到 CPU
unet.eval()

# 尝试迁移到 GPU；显存不足则回退 CPU
if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
        unet = unet.to('cuda:0', non_blocking=True).eval()
        device = torch.device('cuda:0')
        print('[Info] Model moved to CUDA:0')
    except RuntimeError as e:
        print('[Warn] CUDA OOM，回退到 CPU 推理：', e)
        unet = unet.to(cpu_device).eval()
        device = cpu_device
else:
    device = cpu_device
    print('[Info] CUDA 不可用，使用 CPU 推理')

# --------------------
# 读取与预处理输入图像
# --------------------
# 确保为 RGB 三通道
ori_image = Image.open(IMG_PATH).convert('RGB')
resized = ori_image.resize(IMG_SIZE)

# HWC -> CHW，归一化到 [0,1]，再加 batch 维
im = np.asarray(resized, dtype=np.float32) / 255.0     # [H,W,3]
im = im.transpose(2, 0, 1)                             # [3,H,W]
im = np.expand_dims(im, axis=0)                        # [1,3,H,W]

# 转 Tensor 并放到与模型一致的 device
im_t = torch.from_numpy(im)                            # float32
im_t = im_t.to(device, non_blocking=True)

# --------------------
# 推理
# --------------------
with torch.no_grad():
    out = unet(im_t)             # 期望形状 [B,1,H,W]，模型里已 sigmoid
    out = out.squeeze(0)         # [1,H,W]
    out = out.squeeze(0)         # [H,W]
    out_np = out.detach().cpu().numpy()

# --------------------
# 后处理：阈值化 -> 保存调色板 PNG 掩码
# --------------------
# 二值阈值（与训练/评估口径一致）
mask_bin = (out_np > 0.5).astype(np.uint8)   # [H,W]，值为 0/1

# 恢复到原图尺寸
mask_img = Image.fromarray(mask_bin, mode='P').resize(ori_image.size, resample=Image.NEAREST)

# 构造 256*3 的调色板：0 -> 黑(0,0,0), 1 -> 绿(0,128,0)，其余填 0
palette = [0, 0, 0,  0, 128, 0] + [0, 0, 0] * (256 - 2)
mask_img.putpalette(palette)

# 保存
mask_img.save(SAVE_PATH)
print(f'[Done] Saved mask to: {SAVE_PATH}')

# --------------------
# 叠加可视化：将掩码半透明覆盖在原图上并另存
# --------------------
# 构造 alpha 掩码：掩码区域 alpha=~0.3，其余为 0
alpha_val = int(255 * 0.3)
mask_rg = Image.fromarray((mask_bin * 255).astype(np.uint8), mode='L').resize(ori_image.size, resample=Image.NEAREST)
alpha_mask = mask_rg.point(lambda p: alpha_val if p > 0 else 0).convert('L')

# 构造带透明度的有色覆盖层（绿色）
overlay = Image.new('RGBA', ori_image.size, (0, 128, 0, 0))
overlay.putalpha(alpha_mask)

# 原图转 RGBA 后进行 alpha 合成
base_rgba = ori_image.convert('RGBA')
image_mask = Image.alpha_composite(base_rgba, overlay)

overlay_path = os.path.splitext(SAVE_PATH)[0] + '_overlay.png'
image_mask.save(overlay_path)
print(f'[Done] Saved overlay to: {overlay_path}')
