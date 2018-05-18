import os
import yolo.config as cfg
import numpy as np
import cv2
from xml.etree import ElementTree as ET
import tensorflow as tf


class Pasval_VOC(object):
    def __init__(self,phase,year):
        self.phase=phase
        self.year=year
        self.data_path=os.path.join(cfg.DATA_PATH+self.year,'VOCdevkit','VOC'+self.year)
        self.image_size=cfg.IMAGE_SIZE
        self.cell_size=cfg.CELL_SIZE
        self.classes=cfg.CLASSES
        self.class_num=len(self.classes)
        self.class_ind=dict(zip(self.classes,range(self.class_num)))
        self.flipped=cfg.FLIPPED
        self.h_ratio=0.0
        self.w_ratio=0.0
        self.batch_size=cfg.BATCH_SIZE
        self.data2tfrecord()
    def image_read(self,image_num):
        image_path=os.path.join(self.data_path,'JPEGImages',image_num+'.jpg')
        #print('..................Read image from {}...............'.format(image_path))
        image=cv2.imread(image_path)
        shape=image.shape
        self.h_ratio=self.image_size/shape[0]
        self.w_ratio=self.image_size/shape[1]
        image=cv2.resize(image,(self.image_size,self.image_size),interpolation=cv2.INTER_LINEAR)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image=image / 255.0 * 2 - 1# normlization to [-1 , 1]
        if self.flipped==True:
            return image,image[:,::-1,:]
        return image
    def make_lable(self,image_num):
        label = np.zeros([self.cell_size, self.cell_size, 5 + self.class_num], np.float32)
        label_path=os.path.join(self.data_path,'Annotations',image_num+'.xml')
        #print('.............. Load label from {}.............'.format(label_path))
        if self.flipped:
            label_flipped=np.zeros([self.cell_size, self.cell_size, 5 + self.class_num], np.float32)
        tree=ET.parse(label_path)
        root=tree.getroot()
        objects=root.findall('object')
        for object in objects:
            bndbox=object.find('bndbox')
            xmin=bndbox.find('xmin').text
            ymin=bndbox.find('ymin').text
            xmax=bndbox.find('xmax').text
            ymax=bndbox.find('ymax').text
            x1 = max(min(float(xmin) * self.w_ratio, self.image_size - 1), 0)
            y1 = max(min(float(ymin) * self.h_ratio, self.image_size - 1), 0)
            x2 = max(min(float(xmax) * self.w_ratio, self.image_size - 1), 0)
            y2 = max(min(float(ymax) * self.h_ratio, self.image_size - 1), 0)
           # print([x1,y1,x2,y2])
            class_index=self.class_ind[object.find('name').text.lower().strip()]
            #print(object.find('name').text.lower().strip())
            #print(class_index)
            box_x=(x1+x2)/2.0
            box_y=(y1+y2)/2.0
            box_w=x2-x1
            box_h=y2-y1
            box_x_ind=int(box_x / self.image_size * self.cell_size)
            box_y_ind=int(box_y / self.image_size * self.cell_size)
            if label[box_y_ind,box_x_ind,0]==1:
                continue
            label[box_y_ind,box_x_ind,0]=1
            label[box_y_ind,box_x_ind,1:5]=[box_x, box_y, box_w, box_h]
            label[box_y_ind,box_x_ind,5+class_index]=1
            if self.flipped:
                label_flipped[box_y_ind, box_x_ind, 0] = 1
                label_flipped[box_y_ind, box_x_ind, 1:5] = [self.image_size-1-box_x, box_y, box_w, box_h]
                label_flipped[box_y_ind, box_x_ind, 5 + class_index] = 1
        if self.flipped:
            return label, label_flipped[:,::-1,:]
        else:
            return label
    def data2tfrecord(self):
        trainval_text_path=os.path.join(self.data_path,'ImageSets','Main','trainval.txt')

        if self.flipped:

            if not os.path.exists('./train_flipped.tfrecord'):
                print('Start to make train_flipped.tfrecord')
                tfrecord_name = './train_flipped.tfrecord'
                writer = tf.python_io.TFRecordWriter(tfrecord_name)
                with open(trainval_text_path, 'r') as read:
                    lines = read.readlines()
                    for line in lines:
                        image_num = line[0:-1]
                        image,image_flipped=self.image_read(image_num=image_num)
                        label,label_flipped=self.make_lable(image_num=image_num)
                        image_string=image.tostring()
                        image_flipped_string=image_flipped.tostring()
                        label_string=label.tostring()
                        label_flipped_string=label_flipped.tostring()
                        example = tf.train.Example(features=tf.train.Features(
                            feature={'im_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                                     'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))}))
                        writer.write(example.SerializeToString())
                        example = tf.train.Example(features=tf.train.Features(
                            feature={'im_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_flipped_string])),
                                     'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_flipped_string]))}))
                        writer.write(example.SerializeToString())
                writer.close()
                print('Make train_flipped.tfrecord............Done')
        else:
            if not os.path.exists('./train.tfrecord'):
                print('Start to make train.tfrecord')
                tfrecord_name = './train.tfrecord'
                writer = tf.python_io.TFRecordWriter(tfrecord_name)
                with open(trainval_text_path, 'r') as read:
                    lines = read.readlines()
                    for line in lines:
                        image_num = line[0:-1]
                        image = self.image_read(image_num=image_num)
                        label = self.make_lable(image_num=image_num)
                        image_string = image.tostring()
                        label_string = label.tostring()
                        example = tf.train.Example(features=tf.train.Features(
                            feature={'im_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                                     'label' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_string]))}))
                        writer.write(example.SerializeToString())
                writer.close()
                print('Make train.tfrecord....................Done')
    def read_batches_from_tfrecord(self):
        if self.flipped:
            tfrecord_name='./train_flipped.tfrecord'
            print('Read Data from train_flipped.tfrecord')
        else:
            tfrecord_name = './train.tfrecord'
            print('Read Data from train.tfrecord')

        filename_queue = tf.train.string_input_producer([tfrecord_name])

        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        img_features = tf.parse_single_example(
                                                serialized_example,
                                                features={
                                                       'im_raw': tf.FixedLenFeature([], tf.string),
                                                       'label': tf.FixedLenFeature([],tf.string)
                                                       })

        image = tf.decode_raw(img_features['im_raw'], tf.float32)

        label = tf.decode_raw(img_features['label'], tf.float32)

        image = tf.reshape(image, [self.image_size, self.image_size, 3])

        label = tf.reshape(label, [self.cell_size, self.cell_size, 5 + self.class_num])

        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=self.batch_size, capacity=2000,min_after_dequeue=100)

        return image_batch, label_batch












