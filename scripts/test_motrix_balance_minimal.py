"""Minimal test: load balance_infantry, step, check stability."""
import numpy as np
import motrixsim as mtx
import motrixsim.msd as msd
from pathlib import Path

from unilab.base.backend.motrix.scene import _soften_equality_constraint_springs

XML = Path("src/unilab/assets/robots/balance_infantry/scene_flat.xml").resolve()

world = msd.from_file(str(XML))
_soften_equality_constraint_springs(world, xml_path=str(XML))
model = msd.build(world)
model.options.timestep = 0.002

data = mtx.SceneData(model, batch=[1])

print(f"Joints: {len(model.joints)}, DOF: {model.num_dof_pos}, Gravity: {model.options.gravity}")

for step_i in range(500):
    try:
        model.step(data)
    except Exception as e:
        print(f"\nCRASH at step {step_i}: {e}")
        break

    if step_i % 100 == 0:
        dof_pos = np.array(data.dof_pos)
        dof_vel = np.array(data.dof_vel)
        print(f"step {step_i:4d}: max|pos|={np.max(np.abs(dof_pos)):.4f} max|vel|={np.max(np.abs(dof_vel)):.4f}")
        if np.isnan(dof_vel).any():
            print("  NaN!")
            break
else:
    print("500 steps OK")