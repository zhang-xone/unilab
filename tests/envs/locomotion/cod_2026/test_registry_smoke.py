"""验证 Cod2026 环境能被训练框架的注册表找到并创建。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unilab.training import ensure_registries
from unilab.base import registry

ENV_NAME = "Cod2026JoystickFlat"

print("=" * 70)
print("  Cod2026 注册表冒烟测试")
print("=" * 70)

print("\n[1/3] 调用 ensure_registries()...")
ensure_registries()
print("  ✓ 注册表加载完成")

print(f"\n[2/3] 检查环境 '{ENV_NAME}' 是否可发现...")
if not registry.contains(ENV_NAME):
    print(f"  ✗ 注册表中没有 '{ENV_NAME}'")
    sys.exit(1)
print(f"  ✓ 已注册 (backend={registry.find_available_sim_backend(ENV_NAME)})")

print(f"\n[3/3] 通过 registry.make 创建 num_envs=4 的环境...")
env = registry.make(ENV_NAME, num_envs=4, sim_backend="mujoco")
print(f"  ✓ 创建成功")
print(f"    类型: {type(env).__name__}")
print(f"    num_envs: {env.num_envs}")
print(f"    obs_groups_spec: {env.obs_groups_spec}")
print(f"    observation_space shape: {env.observation_space.shape}")
print(f"    action_space shape: {env.action_space.shape}")
env.close()

print("\n" + "=" * 70)
print("  ✓ 全部通过 — 环境可接入训练")
print("=" * 70)
print()
print("  下一步可运行训练:")
print(f"  python scripts/train_rsl_rl.py task=cod_2026_joystick_flat/mujoco")
print()