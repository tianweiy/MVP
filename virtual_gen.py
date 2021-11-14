from nusc_image_projection import read_file, to_batch_tensor, to_tensor, projectionV2, reverse_view_points, get_obj
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm 
import argparse 
import numpy as np 
import torch 
import cv2 
import os 
H=900
W=1600


class PaintDataSet(Dataset):
    def __init__(
        self,
        info_path,
        predictor
    ):
        infos = get_obj(info_path)
        sweeps = []

        paths = set()
        for info in infos:
            if info['lidar_path'] not in paths:
                paths.add(info['lidar_path'])
                sweeps.append(info)

            for sweep in info['sweeps']:
                if sweep['lidar_path'] not in paths: 
                    sweeps.append(sweep)
                    paths.add(sweep['lidar_path'])

        self.sweeps = sweeps
        self.predictor = predictor

    @torch.no_grad()
    def __getitem__(self, index):
        info = self.sweeps[index]
        tokens = info['lidar_path'].split('/')
        output_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        if os.path.isfile(output_path):
            return [] 

        all_cams_path = info['all_cams_path']

        all_data = [info]
        for path in all_cams_path:
            original_image = cv2.imread(path)
            
            if self.predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs = {"image": image, "height": height, "width": width}
            
            all_data.append(inputs) 

        return all_data 
    
    def __len__(self):
        return len(self.sweeps)

def is_within_mask(points_xyc, masks, H=900, W=1600):
    seg_mask = masks[:, :-1].reshape(-1, W, H)
    camera_id = masks[:, -1]
    points_xyc = points_xyc.long()
    valid = seg_mask[:, points_xyc[:, 0], points_xyc[:, 1]] * (camera_id[:, None] == points_xyc[:, -1][None])
    return valid.transpose(1, 0) 

@torch.no_grad()
def add_virtual_mask(masks, labels, points, raw_points, num_virtual=50, dist_thresh=3000, num_camera=6, intrinsics=None, transforms=None):
    points_xyc = points.reshape(-1, 5)[:, [0, 1, 4]] # x, y, z, valid_indicator, camera id 

    valid = is_within_mask(points_xyc, masks)
    valid = valid * points.reshape(-1, 5)[:, 3:4]

    # remove camera id from masks 
    camera_ids = masks[:, -1]
    masks = masks[:, :-1]

    box_to_label_mapping = torch.argmax(valid.float(), dim=1).reshape(-1, 1).repeat(1, 11)
    point_labels = labels.gather(0, box_to_label_mapping)
    point_labels *= (valid.sum(dim=1, keepdim=True) > 0 )  

    foreground_real_point_mask = (valid.sum(dim=1, keepdim=True) > 0 ).reshape(num_camera, -1).sum(dim=0).bool()

    offsets = [] 
    for mask in masks:
        indices = mask.reshape(W, H).nonzero()
        selected_indices = torch.randperm(len(indices), device=masks.device)[:num_virtual]
        if len(selected_indices) < num_virtual:
                selected_indices = torch.cat([selected_indices, selected_indices[
                    selected_indices.new_zeros(num_virtual-len(selected_indices))]])

        offset = indices[selected_indices]
        offsets.append(offset)
    
    offsets = torch.stack(offsets, dim=0)
    virtual_point_instance_ids = torch.arange(1, 1+masks.shape[0], 
        dtype=torch.float32, device='cuda:0').reshape(masks.shape[0], 1, 1).repeat(1, num_virtual, 1)

    virtual_points = torch.cat([offsets, virtual_point_instance_ids], dim=-1).reshape(-1, 3)
    virtual_point_camera_ids = camera_ids.reshape(-1, 1, 1).repeat(1, num_virtual, 1).reshape(-1, 1)

    valid_mask = valid.sum(dim=1)>0
    real_point_instance_ids = (torch.argmax(valid.float(), dim=1) + 1)[valid_mask]
    real_points = torch.cat([points_xyc[:, :2][valid_mask], real_point_instance_ids[..., None]], dim=-1)

    # avoid matching across instances 
    real_points[:, -1] *= 1e4 
    virtual_points[:, -1] *= 1e4 

    if len(real_points) == 0:
        return None 
    
    dist = torch.norm(virtual_points.unsqueeze(1) - real_points.unsqueeze(0), dim=-1) 
    nearest_dist, nearest_indices = torch.min(dist, dim=1) 
    mask = nearest_dist < dist_thresh 

    indices = valid_mask.nonzero(as_tuple=False).reshape(-1)

    nearest_indices = indices[nearest_indices[mask]]
    virtual_points = virtual_points[mask]
    virtual_point_camera_ids = virtual_point_camera_ids[mask]
    all_virtual_points = [] 
    all_real_points = [] 
    all_point_labels = []

    for i in range(num_camera):
        camera_mask = (virtual_point_camera_ids == i).squeeze()
        per_camera_virtual_points = virtual_points[camera_mask]
        per_camera_indices = nearest_indices[camera_mask]
        per_camera_virtual_points_depth = points.reshape(-1, 5)[per_camera_indices, 2].reshape(1, -1)

        per_camera_virtual_points = per_camera_virtual_points[:, :2] # remove instance id 
        per_camera_virtual_points_padded = torch.cat(
                [per_camera_virtual_points.transpose(1, 0).float(), 
                torch.ones((1, len(per_camera_virtual_points)), device=per_camera_indices.device, dtype=torch.float32)],
                dim=0
            )
        per_camera_virtual_points_3d = reverse_view_points(per_camera_virtual_points_padded, per_camera_virtual_points_depth, intrinsics[i])

        per_camera_virtual_points_3d[:3] = torch.matmul(torch.inverse(transforms[i]),
                torch.cat([
                        per_camera_virtual_points_3d[:3, :], 
                        torch.ones(1, per_camera_virtual_points_3d.shape[1], dtype=torch.float32, device=per_camera_indices.device)
                    ], dim=0)
            )[:3]

        all_virtual_points.append(per_camera_virtual_points_3d.transpose(1, 0))
        all_real_points.append(raw_points.reshape(1, -1, 4).repeat(num_camera, 1, 1).reshape(-1,4)[per_camera_indices][:, :3])
        all_point_labels.append(point_labels[per_camera_indices])

    all_virtual_points = torch.cat(all_virtual_points, dim=0)
    all_real_points = torch.cat(all_real_points, dim=0)
    all_point_labels = torch.cat(all_point_labels, dim=0)

    all_virtual_points = torch.cat([all_virtual_points, all_point_labels], dim=1)

    real_point_labels = point_labels.reshape(num_camera, raw_points.shape[0], -1)
    real_point_labels  = torch.max(real_point_labels, dim=0)[0]

    all_real_points = torch.cat([raw_points[foreground_real_point_mask.bool()], real_point_labels[foreground_real_point_mask.bool()]], dim=1)

    return all_virtual_points, all_real_points, foreground_real_point_mask.bool().nonzero(as_tuple=False).reshape(-1)

def init_detector(args):
    from CenterNet2.train_net import setup
    from detectron2.engine import DefaultPredictor

    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    return predictor 

def postprocess(res):
    result = res['instances']
    labels = result.pred_classes
    scores = result.scores 
    masks = result.pred_masks.reshape(scores.shape[0], 1600*900) 
    boxes = result.pred_boxes.tensor

    # remove empty mask and their scores / labels 
    empty_mask = masks.sum(dim=1) == 0

    labels = labels[~empty_mask]
    scores = scores[~empty_mask]
    masks = masks[~empty_mask]
    boxes = boxes[~empty_mask]
    masks = masks.reshape(-1, 900, 1600).permute(0, 2, 1).reshape(-1, 1600*900)
    return labels, scores, masks


@torch.no_grad()
def process_one_frame(info, predictor, data, num_camera=6):
    all_cams_from_lidar = info['all_cams_from_lidar']
    all_cams_intrinsic = info['all_cams_intrinsic']
    lidar_points = read_file(info['lidar_path'])


    one_hot_labels = [] 
    for i in range(10):
        one_hot_label = torch.zeros(10, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)

    one_hot_labels = torch.stack(one_hot_labels, dim=0) 

    masks = [] 
    labels = [] 
    camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1)

    result = predictor.model(data[1:])

    for camera_id in range(num_camera):
        pred_label, score, pred_mask = postprocess(result[camera_id])
        camera_id = torch.tensor(camera_id, dtype=torch.float32, device='cuda:0').reshape(1,1).repeat(pred_mask.shape[0], 1)
        pred_mask = torch.cat([pred_mask, camera_id], dim=1)
        transformed_labels = one_hot_labels.gather(0, pred_label.reshape(-1, 1).repeat(1, 10))
        transformed_labels = torch.cat([transformed_labels, score.unsqueeze(-1)], dim=1)

        masks.append(pred_mask)
        labels.append(transformed_labels)
    
    masks = torch.cat(masks, dim=0)
    labels = torch.cat(labels, dim=0)

    P = projectionV2(to_tensor(lidar_points), to_batch_tensor(all_cams_from_lidar), to_batch_tensor(all_cams_intrinsic))
    camera_ids = torch.arange(6, dtype=torch.float32, device='cuda:0').reshape(6, 1, 1).repeat(1, P.shape[1], 1)
    P = torch.cat([P, camera_ids], dim=-1)

    if len(masks) == 0:
        res = None
    else:
        res  = add_virtual_mask(masks, labels, P, to_tensor(lidar_points), 
            intrinsics=to_batch_tensor(all_cams_intrinsic), transforms=to_batch_tensor(all_cams_from_lidar))
    
    if res is not None:
        virtual_points, foreground_real_points, foreground_indices = res 
        return virtual_points.cpu().numpy(), foreground_real_points.cpu().numpy(), foreground_indices.cpu().numpy()
    else:
        return None 


def simple_collate(batch_list):
    assert len(batch_list)==1
    batch_list = batch_list[0]
    return batch_list


def main(args):
    predictor = init_detector(args)
    data_loader = DataLoader(
        PaintDataSet(args.info_path, predictor),
        batch_size=1,
        num_workers=8,
        collate_fn=simple_collate,
        pin_memory=False,
        shuffle=False
    )

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue 
        info = data[0]
        tokens = info['lidar_path'].split('/')
        output_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        
        res = process_one_frame(info, predictor, data)

        if res is not None:
            virtual_points, real_points, indices = res 
        else:
            virtual_points = np.zeros([0, 14])
            real_points = np.zeros([0, 15])
            indices = np.zeros(0)

        data_dict = {
            'virtual_points': virtual_points, 
            'real_points': real_points,
            'real_points_indice': indices
        }

        np.save(output_path, data_dict)
        # torch.cuda.empty_cache() if you get OOM error 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--info_path', type=str, required=True)
    parser.add_argument('--config-file', type=str, default='c2_config/nuImages_CenterNet2_DLA_640_8x.yaml')
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if not os.path.isdir('data/nuScenes/samples/LIDAR_TOP_VIRTUAL'):
        os.mkdir('data/nuScenes/samples/LIDAR_TOP_VIRTUAL')

    if not os.path.isdir('data/nuScenes/sweeps/LIDAR_TOP_VIRTUAL'):
        os.mkdir('data/nuScenes/sweeps/LIDAR_TOP_VIRTUAL')

    main(args)
