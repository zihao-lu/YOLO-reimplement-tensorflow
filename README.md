# YOLO-reimplement-tensorflow
yolov1  tensorflow python

My Environment：

	Ubuntu          16.04.3 LTS

	numpy           1.13.3

	tensorflow-gpu  1.4.1

	opencv-python   3.4.0.12

	lxml            4.0.0
	
TRAIN：

If you want to train the modle with PASCAL VOC, please the download the dataset by the link below:


https://pjreddie.com/projects/pascal-voc-dataset-mirror/

I use PASCAL VOC2007. After download it, extract the data from the compressed file and change its name into  VOCtrainval_   Please modify the DATA_PATH in yolo/config.py according to your enviorment. Mine is '/home/luzihao/zihao_dataset/VOCtrainval_'

Then， put the YOLO_small.ckpt into yolo/        Download it by this link :

https://drive.google.com/file/d/1Qt4ePFWa43eC8O684jyZwJRHSNLvw9l9/view?usp=sharing

Because i don't have the pre_trained modle from Imagenet, it's so hard to get it, i use the convolutional layers parameter in  YOLO_small.ckpt found from https://github.com/Mr-zihao/yolo_tensorflow ,Thanks!  Then 

	 cd  YOLO-reimplement-tensorflow            
 
	 python3 yolo/train.py
 
You can edit the parameters in yolo/config.py such as learning_rate, scale, threhold etc.

TEST:

Put the YOLO_small.ckpt into yolo/  and put the yolo-18000.data-00000-of-00001  into output/checkpoint/

download it by the link below:

https://drive.google.com/file/d/1H_YTpHnRz-zVSlyFioSid0VSKRsdYH_W/view?usp=sharing

There are two test modes, single image detection and real time video detection(A camera is needed). For detail, please read the code in yolo/test.py

Single image detection:

	cd  YOLO-reimplement-tensorflow

	python3 yolo/test.py
	
the result will exist in output/  

![image](https://github.com/Mr-zihao/YOLO-reimplement-tensorflow/blob/master/image/000001.jpg)![image](https://github.com/Mr-zihao/YOLO-reimplement-tensorflow/blob/master/image/000022.jpg)
