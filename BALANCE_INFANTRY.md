# balance_infantry（ppo_sequence）快速开始

轮腿平衡步兵：4 个五连杆腿关节 + 2 个驱动轮，**PPO + 序列编码器**训练。
策略只吃本体感知，用最近 5 帧 obs 估出 3 维 latent（≈ 机身系线速度）再拼当前帧送 actor
—— 本质是 "learned velocity estimator + PPO"，部署不需要外部测速。

> 仓库整体入口先看 [README_zh.md](README_zh.md)（亮点、快速演示、示例运行、文档目录、引用）；
> 本文只讲 balance_infantry 这条线。

下面按 **装环境 → 训练 → 测试 → 部署 → 辨识** 走一遍。

---

## 0. 装环境

官方引导链接：

| 用途 | 链接 |
|---|---|
| 官方仓库 | https://github.com/unilabsim/UniLab |
| 官方文档 | https://unilabsim.github.io/UniLab-doc/ （中文 `/zh_CN/0-index.html`） |
| 安装页 | [本地 `2-installation.md`](docs/sphinx/source/zh_CN/1-getting_started/2-installation.md) ／ [在线](https://unilabsim.github.io/UniLab-doc/zh_CN/1-getting_started/2-installation.html) |
| 开发者入口 | [CONTRIBUTING.md](CONTRIBUTING.md) → *Development Environment Setup* |
| make 目标 | [Makefile](Makefile)：`setup` / `setup-motrix` / `sync-rocm` / `sync-xpu` |
| Agent 契约 | [CLAUDE.md](CLAUDE.md)（与 [AGENTS.md](AGENTS.md) 内容一致） |
| 项目主页 / 论文 | https://unilabsim.github.io ／ https://unilabsim.github.io/paper/ |

---

## 1. 训练（操作流程）

命令清单只维护在你的 test 文件里：

| 文件 | 里面是什么 |
|---|---|
| [test.txt](src/unilab/envs/locomotion/balance_infantry/test.txt) | 训练、续训、只训练不播放、训练完自动关机、`play_interactive`、`eval` 的整段命令 |
| [test/test](src/unilab/envs/locomotion/balance_infantry/test/test) | `test_gyro.py` / `test_damping.py` 的启动命令 |

操作流程：

1. **选任务**：`--task balance_infantry_jump_fudan_flat --sim mujoco`（jumpvf，可换的任务名见下）。
2. **跑训练**：`uv run train --algo ppo_sequence --task <task> --sim <sim>`，其余参数照抄 test.txt。
3. **看产物**：`logs/rsl_rl_sequence/<registry 名>/<时间戳>_<sim>/` —— tensorboard 看曲线，
   `model_*.pt` 是 checkpoint。
4. **续训**：同一条命令加 `algo.load_run=<run_dir> algo.checkpoint=-1`。
5. **看效果**：转到第 2 节的 play 交互。

可选任务名：`balance_infantry_jump_fudan_flat`（jumpvf，日志目录 `BalanceInfantryJumpvf`）、
`balance_infantry_joystick_flat`、`balance_infantry_joystick_vmc_flat`、
`balance_infantry_jumpv0_flat`、`balance_infantry_jumpv0_flat_pretrain`。

---

## 2. 测试（操作流程）

| 文件 | 里面是什么 |
|---|---|
| [test.txt](src/unilab/envs/locomotion/balance_infantry/test.txt) | `play_interactive`（键盘驱动）和 `eval --render-mode record` 的整段命令 |
| [test/test](src/unilab/envs/locomotion/balance_infantry/test/test) | `test_gyro.py` / `test_damping.py` 的启动命令 |
| [test_gyro.py](src/unilab/envs/locomotion/balance_infantry/test/test_gyro.py) | 交互：注入 gyro / 姿态 / 关节速度，验轴向与正负号 |
| [test_damping.py](src/unilab/envs/locomotion/balance_infantry/test/test_damping.py) | 交互：PD 拉到极限后零力下摆，扫 damping / armature / frictionloss |
| [test_box_pyramid.py](src/unilab/envs/locomotion/balance_infantry/test/test_box_pyramid.py) | 交互：box 金字塔台阶地形 |
| [tests/envs/locomotion](tests/envs/locomotion) | pytest：reward / VMC / action_smooth / 地形命令限幅 |

操作流程：

1. **先跑单测**：`uv run pytest tests/envs/locomotion -k balance_infantry`（40 条）。
2. **官方 play 交互**：照抄 test.txt 里那条 `play_interactive`，`algo.load_run=-1` 取最新 run、
   `algo.checkpoint=-1` 取最新 ckpt，`interactive.action_mode` 可切 `policy` / `zero` / `random`。
3. **交互脚本验细节**：符号方向 → `test_gyro.py`；阻尼/摩擦 → `test_damping.py`；地形 → `test_box_pyramid.py`。
   （这几个要在 `src/unilab/envs/locomotion/balance_infantry` 目录下跑，命令见 test/test。）
4. **无窗口环境**：用 `eval ... --render-mode record` 直接出视频。

---

## 3. 部署（操作流程）

量化脚本都在 [quantize/](src/unilab/envs/locomotion/balance_infantry/quantize)：

| 文件 | 里面是什么 |
|---|---|
| [sync_trained_params.py](src/unilab/envs/locomotion/balance_infantry/quantize/sync_trained_params.py) | 把该 run 的 kp/kd、归一化系数、噪声落成 `quantize/trained_params.py` |
| [export_onnx.py](src/unilab/envs/locomotion/balance_infantry/quantize/export_onnx.py) | 导出 float32 ONNX + `norm_params.h` / `ctrl_params.h`，`--calib_from_sim` 顺便采校准数据 |
| [quantize_model.py](src/unilab/envs/locomotion/balance_infantry/quantize/quantize_model.py) | INT8 量化（`--static` 需要上一步的 `calib_*.npy`） |
| [elu_lut.h](src/unilab/envs/locomotion/balance_infantry/quantize/elu_lut.h) | MCU 上的 ELU 查表 |
| [AI_policy.c](src/unilab/envs/locomotion/balance_infantry/test/test1/AI_policy.c) | 真机固件侧：推理 + 辨识激励 |
| [test_gyro.py](src/unilab/envs/locomotion/balance_infantry/test/test_gyro.py) | 符号测试：部署前验观测通道的正负号 |

操作流程：

1. **同步参数**：`uv run python $Q/sync_trained_params.py --run logs/rsl_rl_sequence/BalanceInfantryJumpvf/<run>`
2. **导出 ONNX**：`uv run python $Q/export_onnx.py --input <run>/model_XXXX.pt --output $Q/<时间戳>_mujoco/model_XXXX_deploy.onnx --calib_from_sim`
3. **量化 INT8**：`uv run python $Q/quantize_model.py --input <ckpt> --output $Q/<时间戳>_mujoco/model_XXXX_int8.pt --static`
4. **符号测试**：跑 `test_gyro.py` 对照 [reference/obs.md](src/unilab/envs/locomotion/balance_infantry/reference/obs.md) 的符号表
   （实机 `RLobs.c` 对 RF/RR 电机速度做了翻转），确认 obs 通道与训练侧一致再上机。

其中 `Q=src/unilab/envs/locomotion/balance_infantry/quantize`。每次导出落在
`$Q/<时间戳>_<backend>/`，含 `model_*.pt`、`*_deploy.onnx`、`*_int8.pt`、`norm_params.h`、
`ctrl_params.h`、`calib_*.npy`。ONNX 输入是**已归一化**的 `obs_history` + `obs`（latent
归一化在 ONNX 内部完成），MCU 侧只做「标准化 → 推理」。

---

## 4. 辨识（VOFA 曲线）

数据和曲线全在 [test1/](src/unilab/envs/locomotion/balance_infantry/test/test1)：

| 文件 | 里面是什么 |
|---|---|
| `vofasine.csv` | 实机轮速 1→24 Hz 对数扫频（±5 rad/s） |
| `vofasin.csv` / `vofagas.csv` / `vofalegsin+.csv` / `wheel_slope.csv` | 5 Hz 定频、气弹簧、腿长激励、斜坡 |
| [cmp_wheel_frf_sweep.png](src/unilab/envs/locomotion/balance_infantry/test/test1/cmp_wheel_frf_sweep.png) / [cmp_vofalegsin+.png](src/unilab/envs/locomotion/balance_infantry/test/test1/cmp_vofalegsin+.png) / [cmp_vofagas.png](src/unilab/envs/locomotion/balance_infantry/test/test1/cmp_vofagas.png) | 实机 vs 仿真的时域/频域对比图 |
| [real2sim_report.md](src/unilab/envs/locomotion/balance_infantry/test/test1/real2sim_report.md) | 辨识结论与仿真参数修正 |

生成这些曲线的脚本：

| 脚本 | 做什么 |
|---|---|
| [wheel_frf_sweep.py](src/unilab/envs/locomotion/balance_infantry/test/test1/wheel_frf_sweep.py) | 轮速环扫频：`vofasine.csv` vs 仿真同指令重放，滑窗拟合出增益/相位 |
| [wheel_frf.py](src/unilab/envs/locomotion/balance_infantry/test/test1/wheel_frf.py) | 仿真 0.5–20 Hz 扫频 + 实机 5 Hz 单点 |
| [leg_frf_sim.py](src/unilab/envs/locomotion/balance_infantry/test/test1/leg_frf_sim.py) / [swing_frf.py](src/unilab/envs/locomotion/balance_infantry/test/test1/swing_frf.py) | 腿长 L / 摆角 φ0 方向扫频 + Bode |
| [vmc_replay.py](src/unilab/envs/locomotion/balance_infantry/test/test1/vmc_replay.py) | 复刻实机 VMC 链路：(L, φ0) → 五杆逆解 → 关节目标 → PD |
| [firmware_test.py](src/unilab/envs/locomotion/balance_infantry/test/test1/firmware_test.py) | 把 `AI_policy.c` 的激励（TRAJ_MODE / SWEEP / 幅度）在仿真原样跑一遍 |
| [sim_replay_id.py](src/unilab/envs/locomotion/balance_infantry/test/test1/sim_replay_id.py) | 实机录波指令在 MuJoCo 重放，出同格式 CSV + 对比图 |
| [id_analyze.py](src/unilab/envs/locomotion/balance_infantry/test/test1/id_analyze.py) | 数据分析（只依赖 numpy） |

操作流程：

1. **录波**：实机跑激励，VOFA+ 存 CSV 到 `test1/`。
2. **仿真重放**：`sim_replay_id.py <csv>`（可选 `--frf`）把实机指令在 MuJoCo 里跑一遍，
   直接出 `sim_<name>.csv` + `cmp_<name>.png`；或跑 `wheel_frf.py` / `wheel_frf_sweep.py` /
   `leg_frf_sim.py` / `swing_frf.py` 做扫频，默认只落 CSV。
3. **出图**：这几个扫频脚本要额外加 `--plot` 才跑 matplotlib。
4. **归档结论**：把参数修正写进 `real2sim_report.md`。

两个坑：φ0 ±0.05 rad 正好等于静摩擦死区（`F/kp = 2/40`），实机那组摆动激励腿全程没动；
本机 mujoco 与 matplotlib 同进程会 segfault，所以脚本拆成「落 CSV」+「出图」两段，
并带 `UV_CACHE_DIR=/tmp/uv-cache`（出图再加 `MPLCONFIGDIR=/tmp/mpl`）。

---

## 硬约束与已知问题

- `num_encoder_obs` = actor obs × `obs_history_length`（29×5 / 35×5），否则 `encode()` 报维度错。
- `local_linvel` 只进 critic，actor 侧靠 encoder 蒸馏，别把它塞进 actor obs。
- 换后端只换 `--sim`（文件名），`training.sim_backend` 只是身份字段。
- [vmc 的 mujoco.yaml](conf/ppo_sequence/task/balance_infantry_joystick_vmc_flat/mujoco.yaml) 的 `reward:`
  残留 9 个已删除字段（`jump_prob` 等），建 env 报
  `ValueError: Config class 'BalanceRewardConfig' has no attribute 'jump_prob'`。
- [jumpv0 的 motrix.yaml](conf/ppo_sequence/task/balance_infantry_jumpv0_flat/motrix.yaml) 的 `defaults:`
  指向不存在的 `balance_infantry_jump_flat/mujoco`，compose 直接失败。
