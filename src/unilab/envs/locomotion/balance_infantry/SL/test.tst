



python lqr/test_vmc_lqr.py --scene flat --collect
python SL/train_sl.py --data ../lqr/lqr_collected.npz --epochs 600 --lr 5e-4

cd /home/xxx/ubuntu_software/unilab/UniLab/src/unilab/envs/locomotion/balance_infantry
python SL/train_sl.py
python SL/train_sl.py --epochs 300 --lr 5e-4

# 无头训练（快速攒数据）
python SL/combined_train.py --headless --scene flat
python SL/combined_train.py --headless --dagger --scene flat
python SL/combined_train.py --headless --scene flat --mix_steps 60000
# 纯测试（自动加载最新模型，按 N 切 NN）
python SL/combined_train.py --test --scene flat
python SL/combined_train.py --test --scene flat --load SL/lqr_pretrain.pth
cd /home/xxx/ubuntu_software/unilab/UniLab/src/unilab/envs/locomotion/balance_infantry
cd /home/xxx/ubuntu_software/unilab/UniLab/src/unilab/envs/locomotion/balance_infantry && python SL/export_for_ppo.py --input SL/log/combined_pretrain_0826_1918_62276samples.pth --output SL/log/ppo_init.pt

# 加载指定模型继续训练
cd /home/xxx/ubuntu_software/unilab/UniLab/src/unilab/envs/locomotion/balance_infantry && python SL/combined_train.py --headless --scene flat

cd /home/xxx/ubuntu_software/unilab/UniLab && uv run scripts/play_interactive.py \
  --algo ppo_sequence \
  --task balance_infantry_joystick_flat \
  --sim mujoco \
  algo.load_run=/home/xxx/ubuntu_software/unilab/UniLab/src/unilab/envs/locomotion/balance_infantry/SL/log/ppo_init.pt \
  algo.checkpoint=-1 \
  interactive.action_mode=policy \
  interactive.keyboard=true \
  +interactive.require_keyboard_command_obs=false \
  algo.num_envs=1 \
  training.device=null \
  interactive.speed=1.0

  cd /home/xxx/ubuntu_software/unilab/UniLab
  uv run train --algo ppo_sequence --task balance_infantry_joystick_flat --sim mujoco \
    algo.resume_path=/home/xxx/ubuntu_software/unilab/UniLab/src/unilab/envs/locomotion/balance_infantry/SL/log/ppo_init.pt
    
checkpoint = torch.load('SL/lqr_pretrain.pth')
policy.encoder.load_state_dict(checkpoint['encoder_state_dict'])
policy.actor.load_state_dict(checkpoint['actor_state_dict'])