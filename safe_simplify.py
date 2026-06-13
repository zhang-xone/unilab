import trimesh
import os

input_dir = "COD-2026RoboMaster-Balance-Simulation_File/MJCF/"

def voxel_downsample(mesh, voxel_size):
    """使用体素下采样（安全）"""
    # 创建体素网格
    voxel_grid = trimesh.voxel.VoxelGrid.from_mesh(mesh, pitch=voxel_size)
    
    # 从体素重建网格
    return voxel_grid.mesh

for f in os.listdir(input_dir):
    if f.endswith('.STL'):
        mesh = trimesh.load(os.path.join(input_dir, f))
        orig = len(mesh.faces)
        
        # 根据部件大小设置体素尺寸
        if "base" in f:
            voxel_size = 0.006
        elif "Wheel" in f:
            voxel_size = 0.004
        else:
            voxel_size = 0.002
        
        simplified = voxel_downsample(mesh, voxel_size)
        simplified.export(os.path.join(input_dir, f))
        print(f"{f}: {orig:,} -> {len(simplified.faces):,}")

print("Simplification complete!")
