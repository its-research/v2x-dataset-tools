import torch
import numpy as np
import json
from .map_data import convert,is_range,uniq,convert2
import pandas
from itertools import permutations
from itertools import product



map_path = '/home/lxy/HiVT-v2x/data/maps/hdmap1.json'
with open(map_path,'rb') as fp :
    map = json.load(fp)

def lane_is_in_intersection(lane_id):
    return map['LANE'][lane_id]['is_intersection']

def get_lane_turn_direction(lane_id):
    return map['LANE'][lane_id]['turn_direction']

def lane_has_traffic_control_measure(lane_id):
    return map['LANE'][lane_id]["has_traffic_control"]


def get_lane_ids_in_xy_bbox(x,y,radius):
    keys = map['LANE'].keys()
    Lane_ids = []
    for key in keys :
        Lane_ids.append(key)
    for Lane_id in Lane_ids:
        centerline = map['LANE'][Lane_id]['centerline']
        for i in range(len(centerline)):
            ctr = convert(centerline[i])
            if is_range(ctr,x,y,radius):
                return Lane_id



def get_lane_features(node_inds,node_positions: torch.Tensor,origin: torch.Tensor,rotate_mat: torch.Tensor,
                      radius: float):
    
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids= []
    for node_position in node_positions:
        lane_ids.append(get_lane_ids_in_xy_bbox(node_position[0], node_position[1],  radius))
    lane_ids = uniq(lane_ids)
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        lane_centerline = map['LANE'][lane_id]['centerline']
        lane_centerline = np.array(convert2(lane_centerline))
        lane_centerline = torch.from_numpy(lane_centerline).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = lane_is_in_intersection(lane_id)
        turn_direction = get_lane_turn_direction(lane_id)
        lane_positions.append(lane_centerline[:-1])
        traffic_control = lane_has_traffic_control_measure(lane_id)
        lane_vectors.append(lane_centerline[1:]-lane_centerline[:-1])
        count = len(lane_centerline)-1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]
    
    return lane_vectors,is_intersections,turn_directions,traffic_controls,lane_actor_index,lane_actor_vectors
