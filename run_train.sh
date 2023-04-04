# python tools/train_dil.py -f cfgs/streamnet_s \
#                           -c ./models/coco_pretrained_models/yolox_s_drfpn.pth \
#                           -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --experiment-name streamnet_s \
#                           -d 4 -b 16 --fp16

# python tools/train_dil.py -f cfgs/streamnet_m \
#                           -c ./models/coco_pretrained_models/yolox_m_drfpn.pth \
#                           -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --experiment-name streamnet_m \
#                           -d 4 -b 16 --fp16

# python tools/train_dil.py -f cfgs/streamnet_l \
#                           -c ./models/coco_pretrained_models/yolox_l_drfpn.pth \
#                           -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --experiment-name streamnet_l \
#                           -d 4 -b 8 --fp16

python tools/train.py -f cfgs/streamnet_l_1200x1920 \
                      -c ./models/coco_pretrained_models/yolox_l_drfpn.pth \
                      --experiment-name streamnet_l_1200x1920 \
                      -d 4 -b 8 --fp16