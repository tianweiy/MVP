import numpy as np 
import pickle 
import torch

def to_tensor(x, device='cuda:0', dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=device)

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

def to_batch_tensor(tensor, device='cuda:0', dtype=torch.float32):
    return torch.stack([to_tensor(x, device=device, dtype=dtype) for x in tensor], dim=0)

def batch_view_points(points, view, normalize, device='cuda:0'):
    # points: batch x 3 x N 
    # view: batch x 3 x 3
    batch_size, _, nbr_points = points.shape 

    viewpad = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    viewpad[:, :view.shape[1], :view.shape[2]] = view 

    points = torch.cat((points, torch.ones([batch_size, 1, nbr_points], device=device)), dim=1)

    # (6 x 4 x 4) x (6 x 4 x N)   -> 6 x 4 x N 
    points = torch.bmm(viewpad, points)
    # points = torch.einsum('abc,def->abd', viewpad, points)

    points = points[:, :3]

    if normalize:
        # 6 x 1 x N
        points = points / points[:, 2:3].repeat(1, 3, 1)

    return points 

def reverse_view_points(points, depths, view, device='cuda:0'):
    # TODO: write test for reverse projection 
    nbr_points = points.shape[1]
    points  = points * depths.repeat(3, 1).reshape(3, nbr_points)

    points = torch.cat((points, torch.ones([1, nbr_points]).to(device)), dim=0)

    viewpad = torch.eye(4).to(device)
    viewpad[:view.shape[0], :view.shape[1]] = torch.inverse(view)

    points = torch.matmul(viewpad, points)
    points = points[:3, :]

    return points 

def read_file(path, num_point_feature=4):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points

@torch.no_grad()
def projectionV2(points, all_cams_from_lidar, all_cams_intrinsic, H=900, W=1600, device='cuda:0'):
    # projected_points
    # camera_x, camera_y, depth in camera coordinate, camera_id 
    num_lidar_point = points.shape[0]
    num_camera = len(all_cams_from_lidar)

    projected_points = torch.zeros((num_camera, points.shape[0], 4), device=device)

    point_padded = torch.cat([
                points.transpose(1, 0)[:3, :], 
                torch.ones(1, num_lidar_point, dtype=points.dtype, device=device)
            ], dim=0)

    # (6 x 4 x 4) x (4 x N) 
    transform_points = torch.einsum('abc,cd->abd', all_cams_from_lidar, point_padded)[:, :3, :]
    
    depths = transform_points[:, 2]

    points_2d = batch_view_points(transform_points[:, :3], all_cams_intrinsic, normalize=True)[:, :2].transpose(2, 1)
    points_2d = torch.floor(points_2d)

    points_x, points_y = points_2d[..., 0].long(), points_2d[..., 1].long()    

    valid_mask = (points_x > 0) & (points_x < W) & (points_y >0) & (points_y < H) & (depths > 0)

    valid_projected_points = projected_points[valid_mask]

    valid_projected_points[:, :2] = points_2d[valid_mask]
    valid_projected_points[:, 2] = depths[valid_mask]
    valid_projected_points[:, 3] = 1 # indicate that there is a valid projection 

    projected_points[valid_mask] = valid_projected_points 

    return projected_points
