#!/usr/bin/env python3
"""导出 Cube.AI / 部署所需文件: float32 ONNX + 归一化头文件 + 校准数据.

用法:
  python export_onnx.py \\
    --input /path/to/model_4000.pt \\
    --output quantize/model_f32.onnx \\
    --calib_from_sim --calib_samples 2000

导出的 ONNX 输入是**已归一化**的 ``obs_history`` 与 ``obs``；latent 归一化
在 ONNX 内部完成（训练时 ``actor_obs_normalizer`` 归一化的是 29+3 维
``cat(obs, latent)``，latent 分量的统计量不是 (0, 1)）。因此 MCU 只需执行
头文件里描述的第 1、2 步。
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch

_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PARENT_DIR)
sys.path.insert(0, os.path.join(_PARENT_DIR, "lqr"))

from quantize_model import (  # noqa: E402
    ACTIVATION,
    DETECTED,
    ActorCriticSequence,
    ExportModel,
    _dequantize_module_inplace,
    _replace_linear_with_int8_calib,
    _verify_onnx_parity,
    collect_calibration_from_sim,
    export_ctrl_header,
    infer_dims_from_ckpt,
    load_float_model,
)
from quantize_model import (
    _export_norm_header as export_norm_header,
)
from ckpt_meta import warn_if_trained_params_mismatch  # noqa: E402

# ═══════════════════════════════════════════════════════════════
# 反量化 int8 → float32
# ═══════════════════════════════════════════════════════════════


def dequantize_to_float32(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not ckpt.get("quantized") or ckpt.get("quantize_dtype") != "qint8_per_channel":
        raise ValueError("checkpoint 不是 qint8_per_channel 量化模型")

    dims = infer_dims_from_ckpt(ckpt)
    dims["activation"] = ACTIVATION
    DETECTED.update({k: v for k, v in dims.items() if v is not None})

    model = ActorCriticSequence(
        num_obs=dims["num_obs"],
        num_critic_obs=dims["num_critic_obs"] or DETECTED["num_critic_obs"],
        num_actions=dims["num_actions"],
        num_encoder_obs=dims["num_encoder_obs"],
        latent_dim=dims["latent_dim"],
        encoder_hidden_dims=dims["encoder_hidden_dims"],
        actor_hidden_dims=dims["actor_hidden_dims"],
        critic_hidden_dims=dims["critic_hidden_dims"],
        activation=dims["activation"],
        obs_normalization=False,
        use_real_vel=False,
    )
    _replace_linear_with_int8_calib(model.encoder)
    _replace_linear_with_int8_calib(model.actor)
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.actor.load_state_dict(ckpt["actor_state_dict"])
    _dequantize_module_inplace(model.encoder)
    _dequantize_module_inplace(model.actor)
    model.eval()
    print("  反量化: int8 → nn.Linear (float32)")
    return model, ckpt


def _norm_dict_from_ckpt(ckpt):
    return {
        "enc_mean": ckpt["enc_mean"],
        "enc_std": ckpt["enc_std"],
        "obs_mean": ckpt["obs_mean"],
        "obs_std": ckpt["obs_std"],
        "latent_mean": ckpt.get("latent_mean"),
        "latent_std": ckpt.get("latent_std"),
    }


# ═══════════════════════════════════════════════════════════════
# 导出
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="导出 Cube.AI 所需文件")
    parser.add_argument("--input", type=str, required=True,
                        help="模型路径 (float32 PPO 或 qint8_per_channel)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 ONNX 路径 (默认: quantize/<run>/<model>_deploy.onnx)")
    parser.add_argument("--calib_from_sim", action="store_true",
                        help="从仿真收集校准数据 (用 float32 策略闭环)")
    parser.add_argument("--calib_samples", type=int, default=2000,
                        help="校准样本数 (控制步)")
    parser.add_argument("--calib_scene", type=str, default="rmuc",
                        help="校准场景 (rmuc/flat)")
    args = parser.parse_args()

    device = torch.device("cpu")

    input_abs = os.path.abspath(args.input)
    warn_if_trained_params_mismatch(input_abs)
    input_name = os.path.splitext(os.path.basename(input_abs))[0]
    project_name = os.path.basename(os.path.dirname(input_abs))
    if args.output is None:
        output_dir = os.path.join(_PARENT_DIR, project_name)
        output_path = os.path.join(output_dir, f"{input_name}_deploy.onnx")
    else:
        output_path = os.path.abspath(args.output)
        output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 复制原始 .pt 到导出目录（test_quantized --compare 依赖同名 float baseline）
    dst_pt = os.path.join(output_dir, os.path.basename(input_abs))
    if os.path.abspath(input_abs) != os.path.abspath(dst_pt):
        shutil.copy2(input_abs, dst_pt)
        print(f"复制原始模型: {dst_pt}")

    print(f"输出 ONNX: {output_path}")

    # ── 1. 加载 ──
    ckpt = torch.load(input_abs, map_location="cpu", weights_only=False)
    if ckpt.get("quantized"):
        print("加载 int8 模型, 反量化到 float32...")
        model, ckpt = dequantize_to_float32(input_abs)
        norm_dict = _norm_dict_from_ckpt(ckpt)
        vel_scale = ckpt.get("vel_scale", model.vel_scale)
    else:
        print("加载 float32 PPO 模型...")
        model, norm_dict = load_float_model(input_abs, device)
        vel_scale = model.vel_scale

    # ── 2. 归一化 / 控制参数头文件 ──
    export_norm_header(norm_dict, vel_scale, output_dir)
    export_ctrl_header(output_dir)

    # ── 3. 导出模型 (encoder + latent 归一化 + actor) ──
    export_model = ExportModel(
        model.encoder, model.actor, vel_scale,
        norm_dict.get("latent_mean"), norm_dict.get("latent_std"),
    )
    export_model.eval()
    export_model.to(device)

    # ── 4. 导出 ONNX + 一致性校验 ──
    num_encoder_obs = DETECTED["num_encoder_obs"]
    num_obs = DETECTED["num_obs"]
    oh = torch.randn(1, num_encoder_obs, device=device)
    o = torch.randn(1, num_obs, device=device)
    torch.onnx.export(
        export_model, (oh, o), output_path,
        input_names=["obs_history_norm", "obs_norm"],
        output_names=["action"],
        opset_version=13,
        dynamic_axes={
            "obs_history_norm": {0: "batch"},
            "obs_norm": {0: "batch"},
            "action": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"  ONNX 已保存: {os.path.abspath(output_path)}")
    _verify_onnx_parity(output_path, model.encoder, model.actor, vel_scale,
                        norm_dict, num_encoder_obs, num_obs, tol=2e-3)

    # ── 5. 校准数据 (归一化后, 与 ONNX 输入同分布) ──
    if args.calib_from_sim:
        oh_arr, o_arr = collect_calibration_from_sim(
            model, norm_dict, scene=args.calib_scene,
            num_steps=args.calib_samples, device=device,
        )
        np.save(os.path.join(output_dir, "calib_obs_history_norm.npy"), oh_arr)
        np.save(os.path.join(output_dir, "calib_obs_norm.npy"), o_arr)
        print(f"  校准数据: calib_obs_history_norm.npy ({oh_arr.shape})")
        print(f"  校准数据: calib_obs_norm.npy ({o_arr.shape})")
    else:
        print("  跳过校准数据 (--calib_from_sim 启用)")

    # ── 6. 总结 ──
    print(f"\n{'=' * 60}")
    print(f"Cube.AI 部署文件 (在 {output_dir}/):")
    print(f"  {os.path.basename(output_path)}  → Cube.AI import")
    print("  norm_params.h               → MCU 归一化 (C 头文件)")
    print("  ctrl_params.h               → MCU 控制参数 (C 头文件)")
    if args.calib_from_sim:
        print("  calib_obs_history_norm.npy  → 校准输入 (obs_history, 归一化后)")
        print("  calib_obs_norm.npy          → 校准输入 (obs, 归一化后)")
    print("\nMCU 推理流程:")
    print(f"  1. 构建原始观测 obs({num_obs}) 与 obs_history({num_encoder_obs})")
    print("  2. float32 归一化: oh = (oh - enc_mean) / (enc_std + NORM_EPS)")
    print("                     o  = (o  - obs_mean) / (obs_std + NORM_EPS)")
    print("  3. 喂入 ONNX → action(6)；latent 归一化已在 ONNX 内部完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
