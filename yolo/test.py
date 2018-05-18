import sys
import os
sys.path.append(os.getcwd())
import tensorflow as tf
import yolo.config as cfg
from   yolo.yolo_net import Yolo_net
import numpy as np
import cv2
import glob
class Test(object):
    def __init__(self):
        self.net=Yolo_net(training=False)
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.index2class = dict(zip(range(self.num_classes), self.classes))
        self.class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 0), (255, 0, 255),(0, 255, 255),
                            (100, 255, 100), (255, 100, 100), (100, 100, 255), (100, 100, 0), (100, 0, 100),(0, 100, 100),
                            (200, 100, 0), (200, 0, 100), (100, 200, 0), (100, 0, 200), (0, 100, 200), (0, 200, 100),
                            (100, 200, 255)]
        self.image_size=cfg.IMAGE_SIZE
        self.cell_size=cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.confidence_boundary = self.cell_size * self.cell_size * self.boxes_per_cell
        self.boxes_boundary = self.confidence_boundary + self.cell_size * self.cell_size * self.boxes_per_cell * 4
        self.h_ratio=0
        self.w_ratio=0
        self.keep_pro = tf.placeholder(tf.float32)
        self.gpu_options=tf.GPUOptions()
        self.config=tf.ConfigProto(gpu_options=self.gpu_options)
        self.restore_path = cfg.RESTORE_PATH
        self.sess=tf.Session(config=self.config)
        self.init=tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver=tf.train.Saver()
        self.ckpt=tf.train.get_checkpoint_state(self.restore_path)
        self.threshold=cfg.THRESHOLD
        self.iou_threshold=cfg.IOU_THRESHOLD
        self.saver.restore(self.sess,self.ckpt.model_checkpoint_path)
    def detect(self,img):
        img_h=img.shape[0]
        img_w=img.shape[1]
        self.h_ratio=img_h/self.image_size
        self.w_ratio=img_w/self.image_size
        input=cv2.resize(img,(self.image_size,self.image_size))
        input=cv2.cvtColor(input,cv2.COLOR_BGR2RGB).astype(np.float32)
        input=(input/255.0)*2.0-1.0
        input=np.reshape(input,(1,self.image_size,self.image_size,3))

        output=self.sess.run(self.net.layers['fc_2'],feed_dict={self.net.data: input,self.net.keep_pro :1.0})
        # print(output.shape)
        confidence = np.reshape(output[0,0:self.confidence_boundary],
                                (self.cell_size, self.cell_size, self.boxes_per_cell))

        coordinate = np.reshape(output[0,self.confidence_boundary:self.boxes_boundary],
                                (self.cell_size, self.cell_size, self.boxes_per_cell, 4))

        classes=np.reshape(output[0,self.boxes_boundary:],(self.cell_size,self.cell_size,self.num_classes))
        offset = [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell
        offset = np.array(offset)
        offset = np.reshape(offset, [self.boxes_per_cell, self.cell_size, self.cell_size])
        x_offset = np.transpose(offset, [1, 2, 0])
        y_offset = np.transpose(x_offset, [1, 0, 2])

        ####         the probality of each class in per box      ###
        pro_mul_confi=np.zeros([self.cell_size,self.cell_size,self.boxes_per_cell,self.num_classes])



        box=np.zeros((self.cell_size,self.cell_size,self.boxes_per_cell,4))
        box[...,0]=(coordinate[..., 0] + x_offset) / self.cell_size
        box[...,1]=(coordinate[..., 1] + y_offset) / self.cell_size
        box[...,2]=np.square(coordinate[..., 2])
        box[...,3]=np.square(coordinate[..., 3])
        box=box*self.image_size
        box[..., 0] = box[..., 0] * self.w_ratio
        box[..., 2] = box[..., 2] * self.w_ratio
        box[..., 1] = box[..., 1] * self.h_ratio
        box[..., 3] = box[..., 3] * self.h_ratio
        for i in np.arange(self.boxes_per_cell):
            for j in np.arange(self.num_classes):
                pro_mul_confi[:,:,i,j]=np.multiply(confidence[:,:,i],classes[:,:,j])



########                         NMS               ###########

        max_pro_in_per_box=np.max(pro_mul_confi,axis=3)

        max_pro_index_per_box=np.argmax(pro_mul_confi,axis=3)


        score_filter=np.array(max_pro_in_per_box>=self.threshold).astype(np.int64)

        score_filter_tuple=np.nonzero(score_filter)

        filtered_max_pro_in_per_box=max_pro_in_per_box[score_filter_tuple[0],score_filter_tuple[1],score_filter_tuple[2]]

        filtered_box=box[score_filter_tuple[0],score_filter_tuple[1],score_filter_tuple[2]]

        filter_max_pro_index_per_box=max_pro_index_per_box[score_filter_tuple[0],score_filter_tuple[1],score_filter_tuple[2]]

        score_sort=np.argsort(filtered_max_pro_in_per_box)[::-1]


        boxes=[]
        for i in np.arange(len(score_sort)):

            index1=score_sort[i]
            if filtered_max_pro_in_per_box[index1]==0:
                continue
            box1=filtered_box[index1]
            boxes.append((box1,filter_max_pro_index_per_box[index1],filtered_max_pro_in_per_box[index1]))
            for j in np.arange(i+1,len(score_sort)):
                index2=score_sort[j]
                if filtered_max_pro_in_per_box[index2]==0:
                    continue
                box2=filtered_box[index2]
                if self.iou(box1,box2)>cfg.IOU_THRESHOLD:
                    if filter_max_pro_index_per_box[index1] == filter_max_pro_index_per_box[index2]:
                        filtered_max_pro_in_per_box[index2]=0

######                        NMS                     ##########

        return self.draw(boxes,img)











    def draw(self,boxes,img):
        if len(boxes)!=0:
            for item in boxes:
                x1 = int(item[0][0] - item[0][2] / 2.0)
                y1 = int(item[0][1] - item[0][3] / 2.0)
                x2 = int(item[0][0] + item[0][2] / 2.0)
                y2 = int(item[0][1] + item[0][3] / 2.0)
                x1 = np.minimum(np.maximum(x1, 0), img.shape[1])
                y1 = np.minimum(np.maximum(y1, 0), img.shape[0])
                x2 = np.minimum(np.maximum(x2, 0), img.shape[1])
                y2 = np.minimum(np.maximum(y2, 0), img.shape[0])
                cv2.rectangle(img,(x1,y1),(x2,y2),self.class_color[item[1]],2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, self.index2class[item[1]], (x1, y1 + 17), font, 1, self.class_color[item[1]], 2)
                cv2.putText(img, str(round(item[2], 2)), (x1, y1 + 47), font, 0.7, self.class_color[item[1]], 2)
                lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
        return img

    def iou(self,box1,box2):
        box1_tran=np.stack([box1[0]-box1[2]/2.0,
                            box1[1]-box1[3]/2.0,
                            box1[0]+box1[2]/2.0,
                            box1[1]+box1[3]/2.0,],
                          axis=-1)
        box2_tran = np.stack([box2[0] - box2[2] / 2.0,
                              box2[1] - box2[3] / 2.0,
                              box2[0] + box2[2] / 2.0,
                              box2[1] + box2[3] / 2.0, ],
                             axis=-1)
        lu=np.maximum(box1_tran[:2],box2_tran[:2])
        rd=np.minimum(box1_tran[2:],box2_tran[2:])

        intersaction=np.maximum(rd-lu,0.0)

        intersaction_area=intersaction[0]*intersaction[1]

        union_area=box1[2]*box1[3]+box2[2]*box2[3]-intersaction_area

        union_area=np.maximum(union_area,1e-5)
        return intersaction_area/union_area

if __name__=='__main__':


###       detect   a   singel   image        ###

    image = './000001.jpg'
    test = Test()
    img = cv2.imread(image)
    img = test.detect(img)
    cv2.imwrite('./output/' + image.split('/')[-1], img)
    print('detect', image, '...........................', 'done')




###    uncomment  the codes  below,   detect  from  camera             ###
    # test = Test()
    # cap=cv2.VideoCapture(0)
    # while(1):
    #     ret,frame=cap.read()
    #     img=test.detect(frame)
    #     cv2.imshow('capture',img)
    #
    #     if cv2.waitKey(1)&0xFF==ord('q'):
    #         break
    # cap.release()
