import torch
# 加载模型
ckpt = torch.load('logs/fast_sac/G1WalkFlat/2026-06-06_14-56-33_mujoco/model_5000.pt', map_location='cpu')
# 查看键
print('=== 模型键 ===')
print(list(ckpt.keys()))
# 查看 actor 网络结构
if 'actor_state_dict' in ckpt:
    print('\n=== Actor 网络层 ===')
    for key, value in ckpt['actor_state_dict'].items():
        print(f'  {key}: {value.shape}')
"
