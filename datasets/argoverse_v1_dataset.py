from typing import List
import pandas as pd 
import os
from typing import Callable, Dict, List, Optional, Tuple, Union
from itertools import permutations
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
from .utils1 import get_lane_features
from utils import TemporalData

class ArgoverseV1Dataset(Dataset):

    def __init__(self,root: str,split: str,transform: Optional[Callable] = None,local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]#创建处理的数据
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self):
        for raw_path in tqdm(self.raw_paths):
            kwargs = precoess_data(self._split, raw_path, self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self):
        return len(self._raw_file_names)
    
    def get(self,idx):
        return torch.load(self.processed_paths[idx])

def precoess_data(split,path,radius):
    df = pd.read_csv(path)
    timestamps = list(np.sort(df['timestamp'].unique()))
    hisoreical_timestamps = timestamps[:50]
    hisoreical_df = df[df['timestamp'].isin(hisoreical_timestamps)]
    actors_ids = list(hisoreical_df['id'].unique())
    df = df[df['id'].isin(actors_ids)]
    num_nodes = len(actors_ids)

    av_df = df[df['tag'] == 'AV'].iloc
    av_index = actors_ids.index(av_df[0]['id'])
    agent_df = df[df['tag'] == 'TARGET_AGENT'].iloc
    agent_index = actors_ids.index(agent_df[0]['id'])
    city = df['city'].values[0]
    # make the scene centered at AV
    origin = torch.tensor([av_df[49]['x'],av_df[49]['y']],dtype=torch.float)
    av_hedding_vector = origin - torch.tensor([av_df[48]['x'],av_df[48]['y']],dtype=torch.float)
    theta = torch.atan2(av_hedding_vector[1],av_hedding_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta),-torch.sin(theta)],
                               [torch.sin(theta),torch.cos(theta)]])
    
    #initionlization
    x = torch.zeros(num_nodes,100,2,dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 100, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 50, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)


    for actor_id, actor_df in df.groupby('id'):
        node_idx = actors_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['timestamp']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 49]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 50:] = True
        xy = torch.from_numpy(np.stack([actor_df['x'].values, actor_df['y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 50, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 50:] = True


    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 50] = padding_mask[:, : 49] & ~padding_mask[:, 1: 50]

    positions = x.clone()
    x[:, 50:] = torch.where((padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 50, 2),
                            x[:, 50:] - x[:, 49].unsqueeze(-2))

    x[:, 1: 50] = torch.where((padding_mask[:, : 49] | padding_mask[:, 1: 50]).unsqueeze(-1),
                              torch.zeros(num_nodes, 49, 2),
                              x[:, 1: 50] - x[:, : 49])
    x[:, 0] = torch.zeros(num_nodes, 2)
    
    # get lane features at the current time step
    df_49 = df[df['timestamp'] == timestamps[49]]
    node_inds_49 = [actors_ids.index(actor_id) for actor_id in df_49['id']]
    node_positions_49 = torch.from_numpy(np.stack([df_49['x'].values, df_49['y'].values], axis=-1)).float()
    
    
    
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors)= get_lane_features( node_inds_49, node_positions_49, origin, rotate_mat, radius)

    y = None if split == 'test' else x[:, 50:]
    seq_id = os.path.splitext(os.path.basename(path))[0]
   
    return {
        'x': x[:, : 50],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


# traj_path = '/home/lxy/HiVT-v2x/data/train'
# raw_file_name = os.listdir(traj_path)
# file_names = [os.path.join(traj_path,f) for  f in raw_file_name]
# for path in file_names:
#     precoess_data(path)


