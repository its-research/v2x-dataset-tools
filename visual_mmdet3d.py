import mmcv
import numpy as np
from mmdet3d.structures import CameraInstance3DBoxes
from mmdet3d.visualization import Det3DLocalVisualizer
from mmengine import load

info_file = load('datasets/single-vehicle-side/kitti_infos_train.pkl')

cam2img = np.array(
    info_file['data_list'][22]['images']['CAM2']['cam2img'], dtype=np.float32)
bboxes_3d = []
for instance in info_file['data_list'][22]['instances']:
    bboxes_3d.append(instance['bbox_3d'])
gt_bboxes_3d = np.array(bboxes_3d, dtype=np.float32)
gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d)
input_meta = {'cam2img': cam2img}

visualizer = Det3DLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')], save_dir='./outputs/')
img = mmcv.imread('datasets/single-vehicle-side/training/image_2/000024.jpg')
img = mmcv.imconvert(img, 'bgr', 'rgb')

visualizer.set_image(img)
visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, input_meta)
visualizer.add_image('demo', visualizer.get_image())

# bboxes_3d = []
# for instance in info_file['data_list'][0]['instances']:
#     bboxes_3d.append(instance['bbox_3d'])
# gt_bboxes_3d = np.array(bboxes_3d, dtype=np.float32)
# gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d)

# visualizer = Det3DLocalVisualizer()
# # set bev image in visualizer
# visualizer.set_bev_image()
# # draw bev bboxes
# visualizer.draw_bev_bboxes(gt_bboxes_3d, edge_colors='orange')
# visualizer.add_image('demo', visualizer.get_image())
