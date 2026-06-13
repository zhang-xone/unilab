#!/usr/bin/env python3
"""View the COD-2026 robot model with full UI."""

import mujoco
import mujoco.viewer

def main():
    model_path = "COD-2026RoboMaster-Balance-Simulation_File/MJCF/COD-2026RoboMaster-Balance.xml"
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"Model loaded successfully!")
    print(f"DOF: {model.nq}, Actuators: {model.nu}")
    
    # 初始化
    mujoco.mj_forward(model, data)
    
    # 打开查看器（启用完整UI）
    with mujoco.viewer.launch_passive(
        model, 
        data,
        show_left_ui=True,   # 启用左侧面板
        show_right_ui=True    # 启用右侧控制面板
    ) as viewer:
        print("MuJoCo viewer opened. Close window to exit.")
        
        # 设置默认视角
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 60
        viewer.cam.elevation = -15
        
        try:
            while viewer.is_running():
                viewer.sync()
        except KeyboardInterrupt:
            print("\nViewer closed.")

if __name__ == "__main__":
    main()
