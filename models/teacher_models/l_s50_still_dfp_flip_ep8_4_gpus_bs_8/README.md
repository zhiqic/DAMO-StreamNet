# DAMO-StreamNet

## DAMO-StreamNet: Optimizing Streaming Perception in Autonomous Driving
<p align='center'>
  <img src='assets/framework.jpg' width='900'/>
</p>

Streaming perception is a vital aspect of autonomous driving, yet previous research has lacked systematic examination. To address this, we propose the optimized framework, DAMO-StreamNet, which incorporates recent advances from the YOLO series and conducts a comprehensive analysis of spatial and temporal perception mechanisms to provide a state-of-the-art solution. The key innovations of this work include: 1) Utilization of a robust neck structure incorporating deformable convolution, which improves the receptive field and feature alignment abilities. 2) Introduction of a dual-branch structure for extracting longer time-series information, resulting in improved prediction accuracy for motion states. 3) Distillation at the logits level, which aligns the logits of the teacher and studentmodels to the semantic space for more efficient optimization. 4) Realtime forecasting mechanism updates support frame features with the current frame before the next prediction in the inference phase to handle realtime streaming perception.



## Model Zoo

|Model |size |velocity | sAP<br>0.5:0.95 | sAP50 |sAP75| coco pretrained models | weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: | :----: |
|[DAMO-StreamNet-S](./cfgs/streamnet_s.py)    |600×960  |1x      |31.8     |52.3 | 31.0 | [link](./models/coco_pretrained_models/yolox_s_drfpn.pth) | [link](./models/checkpoints/streamnet_s.pth) |
|[DAMO-StreamNet-M](./cfgs/streamnet_m.py)    |600×960  |1x      |35.5     |57.0 | 36.2 | [link](./models/coco_pretrained_models/yolox_m_drfpn.pth) | [link](./models/checkpoints/streamnet_m.pth) |
|[DAMO-StreamNet-L](./cfgs/streamnet_l.py)    |600×960  |1x      |37.8     |59.1 | 38.6 | [link](./models/coco_pretrained_models/yolox_l_drfpn.pth) | [link](./models/checkpoints/streamnet_l.pth) |
|[DAMO-StreamNet-L](./cfgs/streamnet_l_1200x1920.py)   |1200×1920  |1x      | **43.3** | **66.1** | **44.6** | [link](./models/coco_pretrained_models/yolox_l_drfpn.pth) | [link](./models/checkpoints/streamnet_l_1200x1920.pth) |

Please find the teacher model [here](./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth).

## Quick Start

### Installation
You can refer to [StreamYOLO](https://github.com/yancie-yjr/StreamYOLO)/[LongShortNet](https://github.com/LiChenyang-Github/LongShortNet) to install the whole environments.

### Train
```shell
bash run_train.sh
```

### Evaluation
```shell
bash run_eval.sh
```


## Acknowledgment
Our implementation is mainly based on [StreamYOLO](https://github.com/yancie-yjr/StreamYOLO) and [LongShortNet](https://github.com/LiChenyang-Github/LongShortNet). We gratefully thank the authors for their wonderful works.


