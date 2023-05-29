# Convert Datasets

## To Kitti

mmdet3d==v1.1.0

### Dair-V2X-V

```shell
python v2x_datasets_tools/dataset_converter/dair2kitti.py --source-root datasets/single-vehicle-side --target-root datasets/single-vehicle-side --split-path configs/example-single-vehicle-split-data.json --label-type camera --sensor-view vehicle

python v2x_datasets_tools/dataset_converter/create_data.py kitti --root-path datasets/single-vehicle-side --out-dir datasets/single-vehicle-side --extra-tag kitti
```
