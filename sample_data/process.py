import json
import numpy as np 


class mapAPI:
    def __init__(self,path) -> None:
        with open(path, 'r') as f:
            self.map = json.load(f)

    def get_lane_id(self):
        return list(self.map['LANE'].keys())

    def get_lane(self, id):
        return LANE(self.map['LANE'][id])

    def get_centerline(self,lane_id):
        ctln = []
        for string in self.map['LANE'][lane_id]['centerline']:
            pos = string[1:-1].split(',')
            pos = [float(i) for i in pos]
            ctln.append(pos)
        return ctln

    def get_left_nbr(self,lane_id):
        
        pass

class LANE:
    def __init__(self, config):
        self.__dict__.update(config)

    def __setattr__(self, key, value):
        if key not in self.__dict__:
            raise AttributeError(key)
        self.__dict__[key] = value




if __name__ == "__main__":
    path = 'maps/hdmap1.json'

    mp = mapAPI(path)
    lane_id = mp.get_lane_id()
    for id in lane_id:
        ctln = mp.get_centerline(id)
        lane = mp.get_lane(id)



    