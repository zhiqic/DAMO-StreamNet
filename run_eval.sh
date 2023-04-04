python tools/eval.py -f cfgs/streamnet_s \
                     -c ./models/checkpoints/streamnet_s.pth \
                     -d 4 -b 32 --conf 0.01 --fp16

# python tools/eval.py -f cfgs/streamnet_m \
#                      -c ./models/checkpoints/streamnet_m.pth \
#                      -d 4 -b 32 --conf 0.01 --fp16

# python tools/eval.py -f cfgs/streamnet_l \
#                      -c ./models/checkpoints/streamnet_l.pth \
#                      -d 4 -b 8 --conf 0.01 --fp16

# python tools/eval.py -f cfgs/streamnet_l_1200x1920 \
#                      -c ./models/checkpoints/streamnet_l_1200x1920.pth \
#                      -d 4 -b 8 --conf 0.01 --fp16