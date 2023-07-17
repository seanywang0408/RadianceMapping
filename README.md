## Boosting Point Clouds Rendering via Radiance Mapping

This is the official code of AAAI'23 paper **Boosting Point Clouds Rendering via Radiance Mapping** written in PyTorch.

## [Paper](https://arxiv.org/abs/2210.15107)

## Installation

```
conda create -n bpcr python=3.8
conda activate bpcr

conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install matplotlib
pip install opencv-python
pip install lpips
pip install git+https://github.com/francois-rozet/piqa
pip install tensorboard
pip install ConfigArgParse
pip install open3d

python setup.py install
```

## Data Preparation

The layout should look like this 

```
code
├── data
    ├── nerf_synthetic
    ├── dtu
    |   ├── dtu_110
    │   │   │── cams_1
    │   │   │── image
    │   │   │── mask
    │   │   │── npbgpp.ply
    |   ├── dtu_114
    |   ├── dtu_118
    ├── scannet
    │   │   │──0000
    |   │   │   │──color_select
    |   │   │   │──pose_select
    |   │   │   |──intrinsic
    |   │   │   |──00.ply
    │   │   │──0043
    │   │   │──0045
    ├── pc
    |   ├── nerf
    │   │   │── chair.ply
    │   │   │── drums.ply  

```

**NeRF-Synthetic**: Please download dataset from [NeRF](https://github.com/bmild/nerf) and put the unpacked files in ``./data/nerf_synthetic``. To generate point clouds, run [Point-NeRF](https://github.com/Xharlie/pointnerf) and save the point clouds in ``./data/pc/nerf``.
You can also download point clouds from [here](https://drive.google.com/drive/folders/1qcEk97RgwCAzzmXUTUXCGNsGzwYPicLA).

**DTU**: Please download images and masks from [IDR](https://github.com/lioryariv/idr) and camera parameters from [PatchmatchNet](https://github.com/FangjinhuaWang/PatchmatchNet). We use the point clouds provided by [npbg++](https://github.com/rakhimovv/npbgpp).

**ScanNet**: Please download data from [ScanNet](http://www.scan-net.org/) and run ``select_scan.py`` to select the frames. We use the point cloud provided by [ScanNet](http://www.scan-net.org/) for ``scene0000_00`` and point clouds provided by [npbg++](https://github.com/rakhimovv/npbgpp) for two other scenes. For ``scene0043_00``, the frames after 1000 are ignored because the camera parameters are ``-inf``.

### Rasterization

```
python run_rasterize.py --config=configs/chair.txt
```

Please change the config file to run other scenes. The fragments would be saved in ``./data/fragments``.

## Training

```
python main.py --config=configs/chair.txt
```

Before training, please ensure that the fragments of this scene already exist. The results would be saved in ``./logs``. You can also run tensorboard to observe training and testing

```
tensorboard --logdir=logs
```

## Acknowledgements and Citation

The code in rasterization borrows a lot from Pytorch3D.

If you find this project useful in your research, please cite the following papers:

```
Huang X, Zhang Y, Ni B, et al. Boosting point clouds rendering via radiance mapping[C]//Proceedings of the AAAI conference on artificial intelligence. 2023, 37(1): 953-961.
```

or in bibtex:

```bibtex
@inproceedings{huang2023boosting,
  title={Boosting point clouds rendering via radiance mapping},
  author={Huang, Xiaoyang and Zhang, Yi and Ni, Bingbing and Li, Teng and Chen, Kai and Zhang, Wenjun},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={37},
  number={1},
  pages={953--961},
  year={2023}
}
```
