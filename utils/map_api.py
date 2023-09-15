import json
import os
import numpy as np


class MapAPI:
    def __init__(self, path):
        map_path = os.path.join(path, 'maps', 'hdmap1.json')
        with open(map_path, 'rb') as f:
            self.map = json.load(f)
        
        self.line_type = list(self.map.keys())
        self.lane_ids = list(self.map['LANE'].keys())

    def get_lane_ids_in_bbox(self, point, radius):
        return list(self.get_lane_in_bbox(point, radius).keys())

    def get_centerline(self, point, radius):
        return self.get_lane_in_bbox(point, radius).values()

    def get_lane_in_bbox(self, point, radius):
        ans = {}
        lanes = self.map['LANE']
        for k, v in lanes.items():
            v['centerline'] = self.str2np(v['centerline'])
            if self.in_radius(v, point, radius):
                v['left_boundary'] = self.str2np(v['left_boundary'])
                v['right_boundary'] = self.str2np(v['right_boundary'])
                ans[k] = v
        return ans


    def str2np(self, ls):
        if isinstance(ls[0], str):
            return np.stack([s[1:-1].split(',') for s in ls]).astype(np.float32)
        else:
            return ls

    def in_radius(self, v, orig, radius):
        line = v['centerline']
        x_o, y_o = orig
        x_max, x_min, y_max, y_min = np.max(line[:, 0]), np.min(line[:, 0]), np.max(line[:, 1]), np.min(line[:, 1])
        return x_o - radius < x_min < x_max < x_o + radius and y_o - radius < y_min < y_max < y_o + radius





if __name__ == "__main__":

    path = '/home/fsq/V2X-Seq-TFD-Example'
    mp = MapAPI(path)
    ctr = mp.get_centerline()