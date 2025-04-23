import trimesh
import sys
import os
import numpy as np

def convert_glb_pointcloud_to_ply(input_file, output_file):
    print(f"正在加载GLB文件: {input_file}")
    try:
        scene = trimesh.load(input_file)
        if isinstance(scene, trimesh.PointCloud):
            print(f"成功加载点云，包含 {len(scene.vertices)} 个点")
            
            # 保存为PLY文件
            print(f"正在保存PLY文件: {output_file}")
            scene.export(output_file, file_type='ply')
            
            print(f"转换完成: {input_file} -> {output_file}")
            return True
        elif isinstance(scene, trimesh.Scene):
            # 合并所有点云几何体
            combined_vertices = []
            combined_colors = []
            for name, geometry in scene.geometry.items():
                if isinstance(geometry, trimesh.PointCloud):
                    print(f"成功加载点云 '{name}'，包含 {len(geometry.vertices)} 个点")
                    combined_vertices.append(geometry.vertices)
                    if geometry.colors is not None:
                        combined_colors.append(geometry.colors)
            
            if combined_vertices:
                # 创建合并的点云
                all_vertices = np.vstack(combined_vertices)
                all_colors = np.vstack(combined_colors) if combined_colors else None
                if len(all_colors) == len(all_vertices):
                    combined_pointcloud = trimesh.PointCloud(vertices=all_vertices, colors=all_colors)
                else:
                    combined_pointcloud = trimesh.PointCloud(vertices=all_vertices)
                
                # 保存合并的点云为单个PLY文件
                print(f"正在保存合并的PLY文件: {output_file}")
                combined_pointcloud.export(output_file, file_type='ply')
                print(f"转换完成: {input_file} -> {output_file}")
                return True
            else:
                print("错误: 场景中不包含有效的点云数据")
                return False
        else:
            print("错误: 输入文件不包含有效的点云数据")
            return False
    except Exception as e:
        print(f"转换失败: {e}")
        return False

if __name__ == "__main__":
    input_file = '/home/rick/Downloads/scene.glb'

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    output_file = os.path.splitext(input_file)[0] + '.ply'
    convert_glb_pointcloud_to_ply(input_file, output_file)