"""Visualize the dataset."""

import os
import os.path as osp

from simple_parsing import ArgumentParser

from v2x_datasets_tools.utils.arguments import DatasetsArguments
from v2x_datasets_tools.visualization.vis_label_in_image import vis_label_in_image

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(DatasetsArguments, dest='datasets_arguments')

    args = parser.parse_args()

    print(args.datasets_arguments)
    save_path = args.datasets_arguments.save_path
    dataset_type = args.datasets_arguments.dataset_type
    if not osp.exists(save_path):
        os.mkdir(save_path)
    data_root = ''

    # image visualization
    if dataset_type == 'DAIR-V2X-C':
        data_root = './datasets/cooperative-vehicle-infrastructure/vehicle-side'
    elif dataset_type == 'DAIR-V2X-I':
        data_root = './datasets/single-infrastructure-side'
    elif dataset_type == 'DAIR-V2X-V':
        data_root = './datasets/single-vehicle-side'

    vis_label_in_image(data_root, save_path)

    # point cloud visualization
