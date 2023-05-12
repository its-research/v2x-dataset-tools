# v2x-dataset-tools

## Prepare

```shell
pip install -r requirements.txt
python setup.py install 
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

### Dair-V2X-V

+ 该数据集的calib文件夹中的.json文件中, "cam_K"是相机内参, 而不是"P", 尽管它们非常相似, 而且数值也非常接近。
+ 该数据集内存在多个焦距和不同视角的相机图片, 相机内参会跟随每一个image文件变化。因此使用该数据集时, 请确认你使用的calib文件和image文件对应, 切勿为全部image文件只使用000000.txt的参数
+ 该数据集的文件序列是非连续的, 切勿直接使用00000 to lens(dataset)来取得文件名序列
+ 该数据集同时提供了针对lidar和camera进行优化后的label。经过测试, 使用lidar的label在camera的投影存在错误(但是投影矩阵是正确的, 只是label本身存在issues). 同样的, + 使用camera的label在lidar中也存在缺点。因此建议使用对应的label
+ label里还有一些其他物体, 例如你可以看到左侧有一些trafficcone

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

+ [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
+ [Where2comm](https://github.com/MediaBrain-SJTU/Where2comm)
