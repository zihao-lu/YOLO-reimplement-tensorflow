import os
BATCH_SIZE=32
IMAGE_SIZE=448
CELL_SIZE=7
BOXES_PER_CELL=2
CLASS_NUM=20
CLASSES=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
DATA_PATH='/home/luzihao/zihao_dataset/VOCtrainval_'
LOG_DIR='./output/logdir/'

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
WEIGHT_INIT_PATH='./yolo/YOLO_small.ckpt'
WEIGHT_FILE='./output/checkpoint/yolo'
RESTORE_PATH='./output/checkpoint/'
if not os.path.exists(RESTORE_PATH):
    os.makedirs(RESTORE_PATH)

if not os.path.exists('./output/checkpoint/'):
    os.makedirs('./output/checkpoint/')
ITE_STEP=40000
DISPLAY_STEP=10
SAVE_STEP=1000
LOG_STEP=100
TRAIN_RESTORE=True

LAMBDA_COOR=5
LAMBDA_NOOBJ=0.5
ALPHA=0.1
MOMENTUM=0.9
DECAY_RATE=0.0005
LEARNING_RATE=0.00001
DROP_OUT_RATE=0.5
EPOCHES=135
KEEP_PRO=0.5
FLIPPED=True
COORD_SCALE=5
NOOBJ_SCALE=0.5

OBJECT_SCALE=1


CLASS_SCALE=1

GPU='0'
#test

THRESHOLD=0.2
IOU_THRESHOLD=0.5

