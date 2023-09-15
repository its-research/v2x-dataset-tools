# 从源文件中读取数据

import os
import json
import argparse
import copy
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.map_api import MapAPI
from utils.cubic_spline import Spline2D


class Preporcessor(Dataset):
    def __init__(self, src_path, save_path, split='val'):
        super().__init__()
        self.vehicle_path = os.path.join(src_path, 'single-vehicle/trajectories', split)
        self.save_path = save_path

        self.files = os.listdir(self.vehicle_path)
        self.map_path = os.path.join(src_path, 'maps', 'hdmap1.json')

        self.map_api = MapAPI(src_path)

        self.split = split
        self.obs_horizon = 50
        self.pred_horizon = 50
        self.radius = 150

    def __getitem__(self, idx):
        data = self.read_agent_data(idx)
        data = self.get_agent_feat(data)
        data['graph'] = self.get_lane_graph(data)
        seq_id = self.files[idx][:-4]
        data['seq_id'] = seq_id
        df = pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )
        self.save(df, seq_id, self.save_path)

    def __len__(self):
        return len(os.listdir(self.vehicle_path))

    def save(self, df, seq_id, save_path=None):
        if not isinstance(df, pd.DataFrame):
            return

        if not save_path:
            save_path = os.path.join(os.path.split(self.vehicle_path)[0], "processed", self.split + "_intermediate", "raw")
        else:
            save_path = os.path.join(save_path, self.split + "_intermediate", "raw")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fname = f"features_{seq_id}.pkl"
        df.to_pickle(os.path.join(save_path, fname))
        # print("[Preprocessor]: Saving data to {} with name: {}...".format(dir_, fname))

    def read_agent_data(self, idx):
        file_path = os.path.join(self.vehicle_path, self.files[idx])
        df = pd.read_csv(file_path)

        agt_ts = np.sort(np.unique(df['timestamp']))
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i
        steps = [mapping[x] for x in df['timestamp'].values]
        steps = np.asarray(steps, np.int64)
        
        city = df['city'].values[0]
        trajs = df[['x', 'y']].values

        objs = df.groupby(['id', 'tag']).groups   
        keys = list(objs.keys())   
        obj_type = [x[1] for x in keys]

        agt_idx = obj_type.index('TARGET_AGENT')  
        idcs = objs[keys[agt_idx]]

        agt_traj = trajs[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        return data  

    def get_agent_feat(self, data):
        
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float32)
        pre = (orig - data['trajs'][0][self.obs_horizon-4]) / 2.0
        theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
        rot = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]], np.float32)  #旋转矩阵

        agt_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float32)   
        agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32) 

        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T

        lanes = self.map_api.get_lane_in_bbox(orig, self.radius)

        ctr_line_candts = []
        for id, v in lanes.items():
            ctrl = v['centerline']
            ctrl = np.matmul(rot, (ctrl - orig.reshape(-1, 2)).T).T
            ctr_line_candts.append(ctrl)

        tar_candts = self.lane_candidate_sampling(ctr_line_candts, [0, 0])

        if self.split == "test":
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)   
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        x_min, x_max, y_min, y_max = -self.radius, self.radius, -self.radius, self.radius
        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon-1 not in step:
                continue
            
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            has_pred = np.zeros(self.pred_horizon, np.bool_)
            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            post_step = step[future_mask] - self.obs_horizon
            post_traj = traj_nd[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = True

            # colect the observation
            obs_mask = step < self.obs_horizon
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]

            for i in range(len(step_obs)):
                if step_obs[i] == self.obs_horizon - len(step_obs) + i:
                    break
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:]

            if len(step_obs) <= 1:
                continue

            feat = np.zeros((self.obs_horizon, 3), np.float32)
            has_obs = np.zeros(self.obs_horizon, np.bool_)

            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1.0
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            feats.append(feat)
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        has_obss = np.asarray(has_obss, np.bool_)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool_)

        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offse_gt

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        return data

    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
        x_min, x_max, y_min, y_max = -self.radius, self.radius, -self.radius, self.radius
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        
        src_lanes = self.map_api.get_lane_in_bbox(data['orig'], radius * 1.5)
        lanes = dict()
        for id, lane in src_lanes.items():
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane['centerline'] - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                # """Getting polygons requires original centerline"""
                # polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                # polygon = copy.deepcopy(polygon)
                lane['centerline'] = centerline
                # lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane['centerline']
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            x = np.zeros((num_segs, 2), np.float32)
            if lane['turn_direction'] == 'LEFT':
                x[:, 0] = 1
            elif lane['turn_direction'] == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane['has_traffic_control'] * np.ones(num_segs, np.float32))
            intersect.append(lane['is_intersection'] * np.ones(num_segs, np.float32))

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count
        lane_idcs = np.concatenate(lane_idcs, 0)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['lane_idcs'] = lane_idcs

        return graph

    @staticmethod
    def get_ref_centerline(cline_list, pred_gt):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx

    @staticmethod
    def get_candidate_gt(target_candidate, gt_target):
        """
        find the target candidate closest to the gt and output the one-hot ground truth
        :param target_candidate, (N, 2) candidates
        :param gt_target, (1, 2) the coordinate of final target
        """
        displacement = gt_target - target_candidate
        gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidate.shape[0], 1))
        onehot[gt_index] = 1

        offset_xy = gt_target - target_candidate[gt_index]
        return onehot, offset_xy

    def lane_candidate_sampling(self, centerline_list, orig, distance=0.5):
        """the input are list of lines, each line containing"""
        if not centerline_list:
            return []
        candidates = []
        for lane_id, line in enumerate(centerline_list):
            sp = Spline2D(x=line[:, 0], y=line[:, 1])
            s_o, d_o = sp.calc_frenet_position(orig[0], orig[1])   # s: the longitudinal; d: the lateral
            s = np.arange(s_o, sp.s[-1], distance)
            ix, iy = sp.calc_global_position_online(s)
            candidates.append(np.stack([ix, iy], axis=1))
        candidates = np.unique(np.concatenate(candidates), axis=0)

        return candidates
if __name__ == "__main__":
    split = 'val'
    src_path = '/home/fsq/V2X-Seq-TFD-Example'   # /maps   ; /single-vehicle
    save_path = '/home/fsq/V2X-Seq-TFD-Example/processed'
    
    dataset = Preporcessor(src_path, save_path, split=split)
    for i in range(len(dataset)):
        dataset[i]
    # loader = DataLoader(
    #     dataset,
    #     batch_size=16,
    #     num_workers=1
    # )
    # total = len(dataset)
    # for i, _ in enumerate(loader):
    #     print(f'{i} / {total}')
