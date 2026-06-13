import trimesh
import os

input_dir = "COD-2026RoboMaster-Balance-Simulation_File/MJCF/"

def fix_mesh(mesh):
    """修复网格问题"""
    # 填充孔洞
    mesh = mesh.fill_holes()
    
    # 移除孤立的面片
    mesh = mesh.remove_degenerate_faces()
    
    # 修复法线
    mesh.fix_normals()
    
    # 确保是水密的
    if not mesh.is_watertight:
        mesh = mesh.fill_holes()
    
    return mesh

for f in os.listdir(input_dir):
    if f.endswith('.STL'):
        mesh = trimesh.load(os.path.join(input_dir, f))
        orig_faces = len(mesh.faces)
        
        # 修复网格
        fixed = fix_mesh(mesh)
        fixed_faces = len(fixed.faces)
        
        fixed.export(os.path.join(input_dir, f))
        print(f"{f}: {orig_faces:,} -> {fixed_faces:,} (fixed)")

print("Mesh fixing complete!")
