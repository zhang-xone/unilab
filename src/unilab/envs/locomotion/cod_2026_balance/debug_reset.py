import numpy as np, mujoco, mujoco.viewer, time
from pathlib import Path

xml = 'src/unilab/assets/robots/cod_2026_balance/scene_flat.xml'
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)

floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
if floor_id >= 0:
    m.geom_contype[floor_id] = 0
    m.geom_conaffinity[floor_id] = 0

base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base')  # 模型里 body 名是 'base'，不是 'base_link'

# 通过 wheel joint 反查它挂载的 body id（避免 body name 跟模型不一致导致拿到 (0,0,0)）
r_wheel_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'Right_Wheel_joint')
l_wheel_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'Left_Wheel_joint')
right_wheel_id = int(m.jnt_bodyid[r_wheel_jid]) if r_wheel_jid >= 0 else -1
left_wheel_id  = int(m.jnt_bodyid[l_wheel_jid]) if l_wheel_jid >= 0 else -1

print('joint_id:  R_wheel_joint=%d  L_wheel_joint=%d' % (r_wheel_jid, l_wheel_jid))
print('body_id:   base=%d  R_wheel_body=%d  L_wheel_body=%d' % (base_id, right_wheel_id, left_wheel_id))
print('nq=%d nv=%d' % (m.nq, m.nv))

if base_id < 0 or right_wheel_id < 0 or left_wheel_id < 0:
    print('❌  有 name 没找到，先确认模型里正确的 joint/body name：')
    print('前 80 个 body name：')
    for i in range(min(m.nbody, 80)):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f'  body[{i:3d}] = {nm!r}')
    print('前 80 个 joint name：')
    for i in range(min(m.njnt, 80)):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f'  joint[{i:3d}] = {nm!r}')
    raise SystemExit(1)
print()
print('>>> Space 暂停 → 拖 Joints 面板滑块 → V 导出')
print()

paused_flag = {'paused': False}


def print_qpos_compact():
    s = ' '.join('%.6f' % d.qpos[i] for i in range(len(d.qpos)))
    print('<key name="home" qpos="' + s + '" />')


def print_wheel_body_pos():
    mujoco.mj_kinematics(m, d)
    base_world = d.xpos[base_id]
    R_b2w = d.xmat[base_id].reshape(3, 3)
    rw_world = d.xpos[right_wheel_id] - base_world
    lw_world = d.xpos[left_wheel_id]  - base_world
    rw_body = R_b2w.T @ rw_world
    lw_body = R_b2w.T @ lw_world
    print(
        "wheel_body_pos: "
        "R=(%+0.4f, %+0.4f, %+0.4f)  L=(%+0.4f, %+0.4f, %+0.4f)"
        % (rw_body[0], rw_body[1], rw_body[2],
           lw_body[0], lw_body[1], lw_body[2])
    )
    swing_rw = float(-np.arctan2(rw_body[0], -rw_body[2])) * 180 / np.pi
    swing_lw = float(-np.arctan2(lw_body[0], -lw_body[2])) * 180 / np.pi
    tag_rw = "前方" if rw_body[0] > 0 else "后方"
    tag_lw = "前方" if lw_body[0] > 0 else "后方"
    print("  swing (body-frame):  R=%+0.2f° (%s)   L=%+0.2f° (%s)"
          % (swing_rw, tag_rw, swing_lw, tag_lw))


def on_key(keycode):
    if keycode == ord(' '):
        paused_flag['paused'] = not paused_flag['paused']
        print('---', 'PAUSED' if paused_flag['paused'] else 'RUNNING', '---')
        print_qpos_compact()
        print_wheel_body_pos()
        return
    if keycode in (ord('V'), ord('v')):
        qpos_list = ['%.6f' % d.qpos[i] for i in range(len(d.qpos))]
        qpos_str = ' '.join(qpos_list)
        new_line = '    <key name="home" qpos="' + qpos_str + '" />'
        print()
        print('=' * 70)
        print(new_line)
        print('=' * 70)
        print()
        print_qpos_compact()
        print_wheel_body_pos()
        print()
        try:
            lines = Path(xml).read_text().splitlines()
            new_lines = []
            replaced = False
            for ln in lines:
                if not replaced and ln.strip().startswith('<key ') and 'home' in ln:
                    new_lines.append(new_line)
                    replaced = True
                else:
                    new_lines.append(ln)
            if not replaced:
                for i, ln in enumerate(new_lines):
                    if '</keyframe>' in ln:
                        new_lines.insert(i, new_line)
                        replaced = True
                        break
            Path(xml).write_text('\n'.join(new_lines) + '\n')
            print('OK 已写入 scene_flat.xml')
        except Exception as exc:
            print('写入失败:', exc)
        print()


with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as v:
    frame = 0
    print_period = 30
    print('--- RUNNING --- (Space 暂停/继续, V 导出, Esc 退出)')
    print_qpos_compact()
    print_wheel_body_pos()
    while True:
        if not v.is_running():
            break
        if not paused_flag['paused']:
            mujoco.mj_step(m, d)
        d.qpos[0] = 0.0
        d.qpos[1] = 0.0
        d.qpos[2] = 0.4
        d.qpos[3] = 1.0
        d.qpos[4] = 0.0
        d.qpos[5] = 0.0
        d.qpos[6] = 0.0
        d.qvel[0:6] = 0.0
        frame += 1
        if frame % print_period == 0:
            print_qpos_compact()
            print_wheel_body_pos()
        v.sync()
        time.sleep(0.016)