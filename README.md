# v2x-dataset-tools

## TODO

- [x] Run on samples
- [ ] Run on full dataset
- [ ] data visualization
- [ ] result visualization
- [ ] DDP base


## Prepare
* python == 3.8.8
* pytorch == 1.8.1
* torch-geometric == 1.7.2 (The version of related libraries are listed as follows.)
  * pytorch-cluster == 1.5.9          
  * pytorch-geometric == 1.7.2           
  * pytorch-scatter == 2.0.7           
  * pytorch-sparse == 0.6.10         
  * pytorch-spline-conv == 1.2.1
* pandas == 1.4.4
* tqdm == 4.60.0
* tensorboard
* (Optianl) [nvidia-apex](https://github.com/NVIDIA/apex) == 1.0

* [Argoverse-api](https://github.com/argoai/argoverse-api)

## HowTo
1. To prepare data:
    ```
    python data/get_data.py
    ```
2. To train the TNT model:
    ```
    python train_tnt.py
    ```

## Issues
There are some bugs in data/get_data.py
