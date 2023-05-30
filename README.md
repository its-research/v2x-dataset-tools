# v2x-dataset-tools

## TODO

- [x] 3D bbox visualization
- [x] 3D point cloud visualization
- [ ] 3D bev visualization
- [ ] rope3d visualization
- [ ] pred fusion with gt visialization
- [ ] V2X-Seq-SPD visualization
- [ ] V2X-Seq-TFD visualization
- [ ] A9 visualization

## Prepare

```shell
pip install -r requirements.txt
python setup.py install

pre-commit install
pre-commit run --all-files
```

### Example Datasets

[BaiduPan](https://pan.baidu.com/s/1Bj97xzdT6i6c-NBPxnsWRA?pwd=6puw)

```shell
ln -s your_path_to_v2x_datasets datasets
```

## HowTo

```shell
python visual.py

# python v2x_datasets_tools/visualization/vis_label_in_image.py
# python v2x_datasets_tools/visualization/vis_label_in_3d.py
```

## Notice

### linter

Run follow command before you commit:

```shell
bash ./dev/linter.sh
```

## Issues

### vtk mayavi

ERROR: No matching distribution found for vtk

```shell
conda install vtk mayavi 'numpy<1.24'
```

### pypcd

```shell
git clone <https://github.com/klintan/pypcd.git>
cd pypcd
python setup.py install
```

## Change log

## Reference

- [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
- [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)
- [mmdetection3d](https://github.com/openmmlab/mmdetection3d)
