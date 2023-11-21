# DAMO-StreamNet: Optimizing Streaming Perception for Autonomous Driving

DAMO-StreamNet is a novel streaming perception framework for real-time video object detection in autonomous driving scenarios. It builds upon state-of-the-art models like YOLO and LongShortNet to achieve optimized accuracy under strict latency constraints.

<p align="center">
<img width="600" src="https://github.com/zhiqic/DAMO-StreamNet/assets/65300431/c60358c6-80de-4366-8dfd-c07ecd4bdfbf">
</p>


## Key Features

- **Robust Neck Design**: Incorporates deformable convolution to enhance receptive fields and feature alignment.

- **Dual-Branch Structure**: Fuses semantic and temporal features for accurate motion prediction. 

- **Asymmetric Distillation**: Distills future knowledge from teacher to student network during training for performance gains.

- **Real-time Forecasting**: Continuously updates support frames for seamless streaming.


For more details, please see our full [IJCAI 2023 paper](https://arxiv.org/pdf/2303.17144.pdf).



## Usage

DAMO-StreamNet supports real-time detection of 8 classes relevant to autonomous driving:

- Person, Bicycle, Car, Motorcycle, Bus, Truck, Traffic Light, Stop Sign

See [ModelScope Documentation](https://docs.modelscope.cn/master/model_development/built_in_model/cv/video_object_detection/damo_streamnet.html) for code examples to run inference using our pretrained models.

## Model Zoo

| Model | Input Size | Velocity | sAP 0.5:0.95 | sAP50 | sAP75 | COCO Weights | Checkpoint |
| ----- | ---------- | -------- | ------------ | ----- | ----- | ------------ | ---------- |
| [DAMO-StreamNet-S](cfgs/streamnet_s.py) | 600x960 | 1x | 31.8 | 52.3 | 31.0 | [link](https://drive.google.com/file/d/1MdxFS7sp45oGc6CMqEnnvtG2ddQzI3s1/view?usp=sharing) | [link](https://drive.google.com/file/d/15Mi8ShE3PiVdEBMzfG2BlVkGFdWPNL19/view?usp=share_link) |
| [DAMO-StreamNet-M](cfgs/streamnet_m.py) | 600x960 | 1x | 35.5 | 57.0 | 36.2 | [link](https://drive.google.com/file/d/1vJIf9CPprdDWrcisg1kCg4vxVBuSZ_kH/view?usp=share_link) | [link](https://drive.google.com/file/d/1P3STvXZPpkzJB6EmsRc0RbSM0T_D0U1Q/view?usp=share_link) |  
| [DAMO-StreamNet-L](cfgs/streamnet_l.py) | 600x960 | 1x | 37.8 | 59.1 | 38.6 | [link](https://drive.google.com/file/d/10rWOhrPf68zUJNigRnjaBTitI0OEEPds/view?usp=share_link) | [link](https://drive.google.com/file/d/1V__om759s2vCXy5L8A1oP8qQqPbPms5A/view?usp=share_link) |
| [DAMO-StreamNet-L](cfgs/streamnet_l_1200x1920.py) | 1200x1920 | 1x | **43.3** | **66.1** | **44.6** | [link](https://drive.google.com/file/d/10rWOhrPf68zUJNigRnjaBTitI0OEEPds/view?usp=share_link) | [link](https://drive.google.com/file/d/17qRB7xIKkSH6RNCk0OF3XFTQO_WACA04/view?usp=share_link) |

Teacher models available [here](https://drive.google.com/drive/folders/1I0R68LqXt7yoUtJ-i1-uynW6dsKSO49Y?usp=sharing).

## Installation

Follow install guidelines from [StreamYOLO](https://github.com/yancie-yjr/StreamYOLO) and [LongShortNet](https://github.com/LiChenyang-Github/LongShortNet).

## Quick Start

### Dataset Preparation 

Follow [Argoverse-HD setup instructions](https://github.com/yancie-yjr/StreamYOLO#quick-start).

### Model Preparation

Organize downloaded models:

```
./models
├── checkpoints
│   ├── streamnet_l_1200x1920.pth
│   ├── streamnet_l.pth
│   ├── streamnet_m.pth
│   └── streamnet_s.pth
├── coco_pretrained_models
│   ├── yolox_l_drfpn.pth
│   ├── yolox_m_drfpn.pth
│   └── yolox_s_drfpn.pth  
└── teacher_models
    └── l_s50_still_dfp_flip_ep8_4_gpus_bs_8
        └── best_ckpt.pth
```

### Training 

```
bash run_train.sh
```

### Evaluation

```
bash run_eval.sh 
```

## Training Details

- 8 Epochs on Argoverse-HD
- SGD Optimizer with Linear LR Schedule
- Random Flip Augmentation  
- Multi-Scale Training

## References

Please cite our paper:

```
@article{DAMO_StreamNet,
  title={DAMO-StreamNet: Optimizing Streaming Perception in Autonomous Driving},
  author={Jun-Yan He, Zhi-Qi Cheng, Chenyang Li, Wangmeng Xiang, Binghui Chen, Bin Luo, Yifeng Geng, Xuansong Xie},
  journal={IJCAI},  
  year={2023}
}
```

DAMO-StreamNet builds on [YOLO](https://arxiv.org/abs/2104.10497), [LongShortNet](https://arxiv.org/abs/2203.17084) and [StreamYOLO](https://arxiv.org/abs/2203.11972).

## License

For academic research only. Please contact authors for commercial licensing.



