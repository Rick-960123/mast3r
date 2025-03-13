import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp
import PIL
import torch
import os
import sys
import cv2
import open3d as o3d
from tumDatasets import TUMDataset

OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate((
        np.c_[idx1, idx2, idx3],
        np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
        np.c_[idx2, idx3, idx4],
        np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
    ), axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((
        img[:-1, :-1].reshape(-1, 3),
        img[:-1, :-1].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3)
    ), axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, 
                  screen_width=0.03, marker=None):
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H,W) * 1.1 # default value

    # create fake camera
    height = max( screen_width/10, focal * screen_width / H )
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width/4)
        marker.vertices += pose_c2w[:3,3]
        marker.visual.face_colors[:,:3] = edge_color
        scene.add_geometry(marker)

def cat_meshes(meshes):
    vertices, faces, colors = zip(*[(m['vertices'], m['faces'], m['face_colors']) for m in meshes])
    n_vertices = np.cumsum([0]+[len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)

# 新增函数：将深度图转换为3D点云
def depth_to_points3d(depth, rgb, fx, fy, cx, cy):
    """将深度图转换为3D点云"""
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    
    valid = (depth > 0) & np.isfinite(depth)
    
    z = depth
    x = (c - cx) * z / fx
    y = (r - cy) * z / fy
    
    points = np.stack((x, y, z), axis=-1)
    
    # 创建与点云形状相同的RGB图像
    pts3d_rgb = np.zeros((rows, cols, 3))
    pts3d_rgb[valid] = rgb[valid]
    
    return points, valid, pts3d_rgb

def transform_points_open3d(points, transform_matrix):
    """使用Open3D将点云从一个坐标系转换到另一个坐标系"""
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 将点云数据设置为Open3D点云对象的点
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    
    # 应用变换
    pcd.transform(transform_matrix)
    
    # 获取变换后的点云
    transformed_points = np.asarray(pcd.points)
    
    # 重塑回原始形状
    transformed_points = transformed_points.reshape(points.shape)
    
    return transformed_points

# 修改main函数以处理没有associations.txt的情况
def main(tum_dataset_path, output_file='tum_mesh.ply'):
    # 相机内参（根据TUM数据集调整）
    fx, fy = 535.4, 539.2  # 焦距
    cx, cy = 320.1, 247.6  # 主点
    
    tum = TUMDataset(tum_dataset_path)
    poses, colors, depths, T_poses = tum.load_poses()

    meshes = []
    cam_colors = []
    focals = []
    imgs = []
    cams2world = []
    
    # 创建全局点云
    global_pcd = o3d.geometry.PointCloud()
    global_points = []
    global_colors = []

    for i, assoc in enumerate(colors):
        # 加载RGB和深度图像
        rgb = cv2.imread(colors[i])
        depth = cv2.imread(depths[i], cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 5000.0
        T_c2w = T_poses[i]

        # 将深度图转换为3D点云
        pts3d, valid, pts3d_rgb = depth_to_points3d(depth, rgb, fx, fy, cx, cy)
        pts3d = transform_points_open3d(pts3d, T_c2w)

        # 创建网格
        mesh = pts3d_to_trimesh(rgb, pts3d, valid)
        meshes.append(mesh)
        
        # 保存相机信息
        cam_colors.append([0, 255, 0])  # 绿色相机
        focals.append(fx)
        imgs.append(rgb)
        cams2world.append(T_c2w)
        
        # 添加有效点到全局点云
        valid_points = pts3d.reshape(-1, 3)[valid.ravel()]
        valid_colors = rgb.reshape(-1, 3)[valid.ravel()] / 255.0  # 归一化颜色值
        
        global_points.append(valid_points)
        global_colors.append(valid_colors)

    # 合并所有网格
    mesh = trimesh.Trimesh(**cat_meshes(meshes))
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    
    # 添加相机
    for i, pose_c2w in enumerate(cams2world):
        camera_edge_color = cam_colors[i]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=0.03)
    
    # 应用旋转变换
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    
    # 导出3D场景
    print(f'(导出3D场景到 {output_file})')
    scene.export(file_obj=output_file)
    print(f"网格已保存到 {output_file}")
    
    # 保存全局点云
    if global_points:
        # 合并所有点云
        all_points = np.vstack(global_points)
        all_colors = np.vstack(global_colors)
        
        # 创建Open3D点云对象
        global_pcd.points = o3d.utility.Vector3dVector(all_points)
        global_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        # 应用相同的旋转变换
        global_pcd.transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
        
        # 可选：对点云进行下采样以减小文件大小
        global_pcd = global_pcd.voxel_down_sample(voxel_size=0.01)
        
        # 保存点云
        pointcloud_file = os.path.splitext(output_file)[0] + '_pointcloud.ply'
        o3d.io.write_point_cloud(pointcloud_file, global_pcd)
        print(f"全局点云已保存到 {pointcloud_file}")

if __name__ == '__main__':
    tum_dataset_path = "/home/rick/Datasets/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household"
    if len(sys.argv) > 1:
        tum_dataset_path = sys.argv[1]
    output_file = 'tum_mesh.glb'
    main(tum_dataset_path, os.path.join(tum_dataset_path, output_file))
