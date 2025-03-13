import trimesh
import sys
import os

def convert_glb_pointcloud_to_ply(input_file, output_file=None):
    """
    将GLB格式的点云文件转换为PLY格式
    
    参数:
    input_file: 输入的GLB文件路径
    output_file: 输出的PLY文件路径，如果为None，则使用输入文件名但扩展名改为.ply
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.ply'
    
    print(f"正在加载GLB文件: {input_file}")
    
    # 使用trimesh加载GLB文件
    try:
        # 读取GLB文件中的点云
        scene = trimesh.load(input_file)
        
        # 检查是否为PointCloud
        if isinstance(scene, trimesh.PointCloud):
            print(f"成功加载点云，包含 {len(scene.vertices)} 个点")
            
            # 保存为PLY文件
            print(f"正在保存PLY文件: {output_file}")
            scene.export(output_file, file_type='ply')
            
            print(f"转换完成: {input_file} -> {output_file}")
            return True
        else:
            print("错误: 输入文件不包含有效的点云数据")
            return False
    except Exception as e:
        print(f"转换失败: {e}")
        return False

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("用法: python glb2ply.py <input_glb_file> [output_ply_file]")
        sys.exit(1)
    
    input_file = args[0]
    output_file = args[1] if len(args) > 1 else None
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
        sys.exit(1)
    
    convert_glb_pointcloud_to_ply(input_file, output_file)