# MTNeRF: Multi-Task Neural Radiance Fields

[**Project**](TBD) | [**Paper**](TBD) | [**YouTube**](TBD)

Pytorch implementation of NeRF, adapted to Dense Depth Supervised NeRF (DDSNeRF). DDSNeRF can improve the training of neural radiance fields by leveraging depth supervision of dense depth maps derived from DERS, IVDE, or any monocular depth estimation tools. It can be used to train NeRF models given only very few input views.

[Dense Depth-supervised NeRF: Fewer Views and Faster Training for Free](https://github.com/HamedRK89/DDSNeRF.git)

To submit to a conf paper

 [Hamed Razavi Khosroshahi](https://dunbar12138.github.io/)<sup>1</sup>,
 [Farnaz Faramarzi Lighvan]()<sup>2</sup>,


<sup>1</sup>ULB, <sup>2</sup>VUB 

---

We propose DDSNeRF (Dense Depth-supervised Neural Radiance Fields), a model for learning neural radiance fields that takes advantage of depth supervised by dense depth maps. Current NeRF methods require many images with known camera parameters -- . Most, if not all, NeRF pipelines make use of the former but ignore the latter. Our key insight is that such sparse 3D input can be used as an additional free signal during training.

---

## Installation

### Clone the repo:
```
git clone https://github.com/HamedRK89/MTNeRF.git
```

### Create a virtual environment
```
python -m venv venv
```

### Dependencies

install requirements:

install pytorch from its official website
```
pip install -r requirements.txt
```

### Data

Download the dataset [here]()

You can use any multi-view dataset that can be fit in "Data Augmentation" pipeline.

## How to Run?

### Generate camera poses using COLMAP

First, place your scene directory. See the following directory structure for an example:

```
├── data
│   ├── ULB_Toys_Table
│   ├── ├── images
│   ├── ├── ├── V00.png
│   ├── ├── ├── V01.png
```
To generate the poses and sparse point cloud:
```
python imgs2poses.py <your_scenedir>
```

Note: if you use this data format, make sure your `dataset_type` in the config file is set as `llff`.

### Training

```
python run_nerf.py --config configs/Ulb_Toys.txt --depth_supervision True
```
It will create an experiment directory in `./logs`, and store the checkpoints and rendering examples there.

You can create your own experiment configuration to try other datasets.

## Citation