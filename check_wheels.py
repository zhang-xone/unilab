import mujoco

model = mujoco.MjModel.from_xml_path('src/unilab/assets/robots/cod_2026_balance/cod_2026_balance.xml')
data = mujoco.MjData(model)

print('='*70)
print('🔍 左右轮子参数对比')
print('='*70)

# 找到两个轮子
wheel_bodies = {}
for i in range(1, model.nbody):
    name = model.body(i).name
    if 'Wheel' in name:
        wheel_bodies[name] = i

print(f"\n【两个轮子 body】")
for name, idx in wheel_bodies.items():
    pos = model.body_pos[idx]
    ipos = model.body_ipos[idx]
    mass = model.body_mass[idx]
    inertia = model.body_inertia[idx]
    print(f"\n  {name}:")
    print(f"    pos         = ({pos[0]:.6f}, {pos[1]:+.6f}, {pos[2]:.6f})")
    print(f"    mass         = {mass:.6f}")
    print(f"    inertia diag = ({inertia[0]:.6f}, {inertia[1]:.6f}, {inertia[2]:.6f})")
    print(f"    所在 Y 方向   = {'+Y (左侧)' if pos[1] > 0 else '-Y (右侧)'}")

print(f"\n\n【两个轮子 joint】")
for j in range(model.njnt):
    name = model.joint(j).name
    if 'Wheel' in name:
        pos = model.jnt_pos[j]
        axis = model.jnt_axis[j]
        damping = model.dof_damping[j]
        armature = model.dof_armature[j]
        print(f"\n  {name}:")
        print(f"    axis    = ({axis[0]:.1f}, {axis[1]:.1f}, {axis[2]:.1f})")
        print(f"    damping = {damping:.6f}")
        print(f"    armature= {armature:.6f}")
        print(f"    所属 body pos Y = {model.body_pos[model.jnt_bodyid[j]][1]:+.6f}")

print(f"\n\n【两个轮子执行器】")
for i in range(model.nu):
    name = model.actuator(i).name
    if 'Wheel' in name:
        gain = model.actuator_gainprm[i]
        trntype = model.actuator_trntype[i]
        joint_id = model.actuator_trnid[i][0]
        joint_name = model.joint(joint_id).name
        ctrlrange = model.actuator_ctrlrange[i]
        print(f"\n  {name}:")
        print(f"    控制关节: {joint_name}")
        print(f"    gainprm  = {gain}")
        print(f"    trntype  = {trntype}")
        print(f"    ctrlrange = [{ctrlrange[0]:.2f}, {ctrlrange[1]:.2f}]")

# 看看两个轮子的接触几何
print(f"\n\n【两个轮子 geom】")
for i in range(model.ngeom):
    name = model.geom(i).name
    if 'Wheel' in name:
        contype = int(model.geom_contype[i])
        conaff = int(model.geom_conaffinity[i])
        size = model.geom_size[i]
        print(f"\n  geom {i}: {name}")
        print(f"    contype  = {contype}")
        print(f"    conaffinity = {conaff}")
        print(f"    size     = ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")

print('\n' + '='*70)
print('💡 检查完成！看看两边是否有不一致的参数')
print('='*70)
