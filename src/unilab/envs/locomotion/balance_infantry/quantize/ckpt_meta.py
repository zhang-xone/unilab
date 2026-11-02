"""Checkpoint 元数据解析 —— 量化 / 导出 / 参数同步共用。

这里只做纯 state_dict / numpy 逻辑，不 import torch 运行时、不碰 mujoco，
这样离线工具（``sync_trained_params.py``）可以直接复用，不会拉起仿真依赖。

职责边界：
* ``infer_dims_from_ckpt``: 从 float32 / qint8 state_dict 推断网络维度；
* ``extract_norm_stats``: 从 EmpiricalNormalization buffer 提取部署侧归一化系数。

不要在调用方再复制一份同样的解析逻辑；结构或归一化格式变化时只改这里。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# 旧 checkpoint 缺字段时的兜底；正常路径所有值都从 state_dict 推断。
_FALLBACK_NUM_OBS = 29
_FALLBACK_NUM_ACTIONS = 6
_FALLBACK_OBS_HISTORY_LENGTH = 5
_FALLBACK_LATENT_DIM = 3
_FALLBACK_NUM_CRITIC_OBS = 92
_FALLBACK_ENCODER_HIDDEN_DIMS = [128, 64]
_FALLBACK_ACTOR_HIDDEN_DIMS = [128, 64, 32]
_FALLBACK_CRITIC_HIDDEN_DIMS = [256, 128, 64]
_FALLBACK_ACTIVATION = "elu"


def mlp_dims(state_dict, prefix="", weight_key="weight"):
    """按 ``<prefix>.<i>.<weight_key>`` 顺序读出每个 Linear 的输出维度。"""
    dims = []
    i = 0
    while True:
        key = f"{prefix}.{i}.{weight_key}" if prefix else f"{i}.{weight_key}"
        if key not in state_dict:
            break
        dims.append(int(state_dict[key].shape[0]))
        i += 2
    return dims


def split_hidden_dims(all_dims, output_dim):
    """从尾部剥离输出层，返回真正的 hidden_dims。

    例: [128,64,3,3] → output_dim=3 → hidden=[128,64]
        [128,64,32,6,6] → output_dim=6 → hidden=[128,64,32]
    """
    n = 0
    for d in reversed(all_dims):
        if d == output_dim:
            n += 1
        else:
            break
    return all_dims[:-n] if n else all_dims[:-1]


def infer_dims_from_ckpt(ckpt):
    """从 float32 / qint8 checkpoint 反推模型维度，返回 dict。

    不依赖任何硬编码的观测/动作维度；float 与量化 checkpoint 都支持。
    """
    if ckpt.get("quantized", False):
        encoder_all = mlp_dims(ckpt["encoder_state_dict"], weight_key="weight_q")
        actor_all = mlp_dims(ckpt["actor_state_dict"], weight_key="weight_q")
        num_obs = int(np.asarray(ckpt["obs_mean"]).reshape(-1).size)
        num_encoder_obs = int(np.asarray(ckpt["enc_mean"]).reshape(-1).size)
        latent_mean = ckpt.get("latent_mean")
        latent_dim = (int(np.asarray(latent_mean).reshape(-1).size)
                      if latent_mean is not None else _FALLBACK_LATENT_DIM)
        num_actions = actor_all[-1] if actor_all else _FALLBACK_NUM_ACTIONS
        num_critic_obs = None
        encoder_hidden_dims = (split_hidden_dims(encoder_all, encoder_all[-1])
                               if encoder_all else list(_FALLBACK_ENCODER_HIDDEN_DIMS))
        actor_hidden_dims = (split_hidden_dims(actor_all, num_actions)
                             if actor_all else list(_FALLBACK_ACTOR_HIDDEN_DIMS))
        critic_hidden_dims = list(_FALLBACK_CRITIC_HIDDEN_DIMS)
    else:
        sd = ckpt.get("model_state_dict", ckpt)
        encoder_all = mlp_dims(sd, "encoder")
        actor_all = mlp_dims(sd, "actor")
        num_encoder_obs = int(sd["encoder.0.weight"].shape[1])
        latent_dim = encoder_all[-1]
        num_actions = actor_all[-1]
        if "actor_obs_normalizer._mean" in sd:
            num_obs = int(sd["actor_obs_normalizer._mean"].shape[-1]) - latent_dim
        else:
            num_obs = num_encoder_obs // _FALLBACK_OBS_HISTORY_LENGTH
        num_critic_obs = (int(sd["critic.0.weight"].shape[1])
                          if "critic.0.weight" in sd else _FALLBACK_NUM_CRITIC_OBS)
        encoder_hidden_dims = split_hidden_dims(encoder_all, latent_dim)
        actor_hidden_dims = split_hidden_dims(actor_all, num_actions)
        critic_all = mlp_dims(sd, "critic")
        critic_hidden_dims = (split_hidden_dims(critic_all, critic_all[-1])
                              if critic_all else list(_FALLBACK_CRITIC_HIDDEN_DIMS))

    if num_obs <= 0 or num_encoder_obs % num_obs != 0:
        raise ValueError(
            f"无法推断观测结构: num_obs={num_obs}, num_encoder_obs={num_encoder_obs}"
        )
    obs_history_length = num_encoder_obs // num_obs

    return {
        "num_obs": num_obs,
        "num_actions": num_actions,
        "obs_history_length": obs_history_length,
        "num_encoder_obs": num_encoder_obs,
        "latent_dim": latent_dim,
        "num_critic_obs": num_critic_obs,
        "encoder_hidden_dims": encoder_hidden_dims,
        "actor_hidden_dims": actor_hidden_dims,
        "critic_hidden_dims": critic_hidden_dims,
        "activation": _FALLBACK_ACTIVATION,
    }


def _as_numpy(x):
    """torch tensor / ndarray / list → float32 ndarray（不 import torch）。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def extract_norm_stats(ckpt, dims=None):
    """从 float checkpoint 的 normalizer buffer 提取部署侧归一化系数。

    Returns:
        dict，包含:
        * ``obs_mean`` / ``obs_std``: 单帧 obs 归一化 (num_obs,)
        * ``latent_mean`` / ``latent_std``: latent 归一化 (latent_dim,)
        * ``actor_mean`` / ``actor_std``: [obs, latent] 全量 (num_obs+latent_dim,)
        * ``enc_mean`` / ``enc_std``: 按 history 展开后的 encoder 输入归一化
        * ``enc_mean_per_frame`` / ``enc_std_per_frame``: 单帧均值/方差统计
    """
    sd = ckpt.get("model_state_dict", ckpt)
    dims = infer_dims_from_ckpt(ckpt) if dims is None else dims
    num_obs = int(dims["num_obs"])
    num_encoder_obs = int(dims["num_encoder_obs"])
    obs_history_length = int(dims["obs_history_length"])

    if "encoder_obs_normalizer._mean" not in sd or "actor_obs_normalizer._mean" not in sd:
        raise ValueError(
            "checkpoint 不含 EmpiricalNormalization 统计量 "
            "(encoder_obs_normalizer._mean / actor_obs_normalizer._mean)。\n"
            "量化/导出/同步必须先有训练归一化系数，否则部署侧无法复现训练输入。"
        )

    enc_mean_raw = _as_numpy(sd["encoder_obs_normalizer._mean"])
    enc_std_raw = _as_numpy(sd["encoder_obs_normalizer._std"])
    if enc_mean_raw.shape[-1] == num_encoder_obs:
        # 训练时 encoder normalizer 覆盖整个 history；部署侧复用同一组单帧统计。
        enc_mean_per_frame = enc_mean_raw.reshape(
            1, obs_history_length, num_obs).mean(axis=1).reshape(-1)
        enc_std_per_frame = enc_std_raw.reshape(
            1, obs_history_length, num_obs).mean(axis=1).reshape(-1)
    else:
        enc_mean_per_frame = enc_mean_raw.reshape(-1)
        enc_std_per_frame = enc_std_raw.reshape(-1)

    if enc_mean_per_frame.size != num_obs or enc_std_per_frame.size != num_obs:
        raise ValueError(
            f"encoder normalizer 维度不匹配: mean={enc_mean_per_frame.size}, "
            f"std={enc_std_per_frame.size}, 期望 {num_obs}"
        )

    actor_mean = _as_numpy(sd["actor_obs_normalizer._mean"]).reshape(-1)
    actor_std = _as_numpy(sd["actor_obs_normalizer._std"]).reshape(-1)
    if actor_mean.size != num_obs + int(dims["latent_dim"]):
        raise ValueError(
            f"actor normalizer 维度不匹配: mean={actor_mean.size}, "
            f"期望 {num_obs}+{int(dims['latent_dim'])}"
        )

    return {
        "obs_mean": actor_mean[:num_obs].astype(np.float32),
        "obs_std": actor_std[:num_obs].astype(np.float32),
        "latent_mean": actor_mean[num_obs:].astype(np.float32),
        "latent_std": actor_std[num_obs:].astype(np.float32),
        "actor_mean": actor_mean.astype(np.float32),
        "actor_std": actor_std.astype(np.float32),
        "enc_mean_per_frame": enc_mean_per_frame.astype(np.float32),
        "enc_std_per_frame": enc_std_per_frame.astype(np.float32),
        "enc_mean": np.tile(enc_mean_per_frame, obs_history_length).astype(np.float32),
        "enc_std": np.tile(enc_std_per_frame, obs_history_length).astype(np.float32),
    }


def warn_if_trained_params_mismatch(checkpoint_path):
    """checkpoint 所属 run 与 trained_params.py 的同步源不一致时给出提示。

    只警告不退出：checkpoint 可能被复制到临时目录，无法总是根据路径判断。
    但它能拦住“训练了新 run，却忘了 sync 就量化/测试”的静默错误。
    """
    try:
        from trained_params import SOURCE
    except Exception:  # noqa: BLE001 - 没有生成文件时由上层 import 报错
        return

    expected = SOURCE.get("run")
    actual = Path(str(checkpoint_path)).resolve().parent.name
    if expected and actual and actual != expected:
        print(
            f"[params] WARN: trained_params.py 同步自 run={expected}，"
            f"当前 checkpoint 属于 run={actual}。\n"
            f"         先运行 quantize/sync_trained_params.py --run {actual}，"
            "否则 PD / action_scale / VMC / 噪声可能不是该 run 的值。"
        )
