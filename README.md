# DAMO-StreamNet

## DAMO-StreamNet: Optimizing Streaming Perception in Autonomous Driving
<p align='center'>
  <img src='assets/framework.jpg' width='900'/>
</p>

Real-time perception, or streaming perception, is a crucial aspect of autonomous driving that has yet to be thoroughly explored in existing research. To address this gap, we present DAMO-StreamNet, an optimized framework that combines recent advances from the YOLO series with a comprehensive analysis of spatial and temporal perception mechanisms, delivering a cutting-edge solution. The key innovations of DAMO-StreamNet are: (1) A robust neck structure incorporating deformable convolution, enhancing the receptive field and feature alignment capabilities. (2) A dual-branch structure that integrates short-path semantic features and long-path temporal features, improving motion state prediction accuracy. (3) Logits-level distillation for efficient optimization, aligning the logits of teacher and student networks in semantic space. (4) A real-time forecasting mechanism that updates support frame features with the current frame, ensuring seamless streaming perception during inference.~Our experiments demonstrate that DAMO-StreamNet surpasses existing state-of-the-art methods, achieving 37.8\% (normal size (600, 960)) and 43.3\% (large size (1200, 1920)) sAP without using extra data. This work not only sets a new benchmark for real-time perception but also provides valuable insights for future research. Additionally, DAMO-StreamNet can be applied to various autonomous systems, such as drones and robots, paving the way for real-time perception.



## Model Zoo

|Model |size |velocity | sAP<br>0.5:0.95 | sAP50 |sAP75| coco pretrained models | weights |
| ------        |:---: | :---:       |:---:     |:---:  | :---: | :----: | :----: |
|[DAMO-StreamNet-S](./cfgs/streamnet_s.py)    |600×960  |1x      |31.8     |52.3 | 31.0 | [link](https://drive.google.com/file/d/1MdxFS7sp45oGc6CMqEnnvtG2ddQzI3s1/view?usp=sharing) | [link](https://drive.google.com/file/d/15Mi8ShE3PiVdEBMzfG2BlVkGFdWPNL19/view?usp=share_link) |
|[DAMO-StreamNet-M](./cfgs/streamnet_m.py)    |600×960  |1x      |35.5     |57.0 | 36.2 | [link](https://drive.google.com/file/d/1vJIf9CPprdDWrcisg1kCg4vxVBuSZ_kH/view?usp=share_link) | [link](https://drive.google.com/file/d/1P3STvXZPpkzJB6EmsRc0RbSM0T_D0U1Q/view?usp=share_link) |
|[DAMO-StreamNet-L](./cfgs/streamnet_l.py)    |600×960  |1x      |37.8     |59.1 | 38.6 | [link](https://drive.google.com/file/d/10rWOhrPf68zUJNigRnjaBTitI0OEEPds/view?usp=share_link) | [link](https://drive.google.com/file/d/1V__om759s2vCXy5L8A1oP8qQqPbPms5A/view?usp=share_link) |
|[DAMO-StreamNet-L](./cfgs/streamnet_l_1200x1920.py)   |1200×1920  |1x      | **43.3** | **66.1** | **44.6** | [link](https://drive.google.com/file/d/10rWOhrPf68zUJNigRnjaBTitI0OEEPds/view?usp=share_link) | [link](https://drive.google.com/file/d/17qRB7xIKkSH6RNCk0OF3XFTQO_WACA04/view?usp=share_link) |

Please find the teacher model [here](https://drive.google.com/drive/folders/1I0R68LqXt7yoUtJ-i1-uynW6dsKSO49Y?usp=sharing).

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


