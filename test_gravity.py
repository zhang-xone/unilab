#!/usr/bin/env python3
"""不固定 base，让整个机器人自由下落"""
import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

xml = 'src/unilab/assets/robots/balance_infantry/scene_flat.xml'
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)

# 去掉地面碰撞
floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
if floor_id >= 0:
    m.geom_contype[floor_id] = 0
    m.geom_conaffinity[floor_id] = 0

print('nq=%d nv=%d' % (m.nq, m.nv))
print('重力:', m.opt.gravity)
print()
print('>>> 操作：Space暂停/继续，左上角Joint面板拖滑块，Esc退出')
print()


def print_qpos_compact():
    s = ' '.join('%.6f' % d.qpos[i] for i in range(len(d.qpos)))
    print('<key name="home" qpos="' + s + '" />')


paused_flag = {'paused': False}


def on_key(keycode):
    if keycode == ord(' '):
        paused_flag['paused'] = not paused_flag['paused']
        print('---', 'PAUSED' if paused_flag['paused'] else 'RUNNING', '---')
        print_qpos_compact()


with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
    frame = 0
    print('--- RUNNING ---')
    print_qpos_compact()
    print()
    while v.is_running():
        if not paused_flag['paused']:
            mujoco.mj_step(m, d)
        # 固定 base 在 (0,0,0.5)
        d.qpos[0] = 0.0
        d.qpos[1] = 0.0
        d.qpos[2] = 0.5
        d.qpos[3] = 1.0
        d.qpos[4] = 0.0
        d.qpos[5] = 0.0
        d.qpos[6] = 0.0
        d.qvel[0:6] = 0.0
        frame += 1
        if frame % 30 == 0:
            print_qpos_compact()
        v.sync()
        time.sleep(0.016)