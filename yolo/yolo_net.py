from yolo.Network import Network
import tensorflow as tf
from easydict import EasyDict as edict
import yolo.config as cfg
import numpy as np
class Yolo_net(Network):
    def __init__(self, training=True):
        Network.__init__(self)
        self.phase=training
        self.classes=cfg.CLASSES
        self.num_classes=len(self.classes)
        self.image_size=cfg.IMAGE_SIZE
        self.cell_size=cfg.CELL_SIZE
        self.boxes_per_cell=cfg.BOXES_PER_CELL
        self.out_size=self.cell_size*self.cell_size*(self.num_classes+self.boxes_per_cell*5)
        self.data = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3])
        self.layers.data=self.data
        self.confidence_boundary=self.cell_size*self.cell_size*self.boxes_per_cell
        self.boxes_boundary=self.confidence_boundary+self.cell_size*self.cell_size*self.boxes_per_cell*4
        self.batch_size=cfg.BATCH_SIZE
        self.coord_scale=cfg.COORD_SCALE
        self.noobj_scale=cfg.NOOBJ_SCALE
        self.object_scale=cfg.OBJECT_SCALE
        self.class_scale=cfg.CLASS_SCALE
        self.weight_decay=cfg.DECAY_RATE
        self.build_network()
        if self.phase:
            self.label = tf.placeholder(tf.float32, shape=[None, self.cell_size, self.cell_size, 5 + self.num_classes])
            self.calculate_loss()
            self.total_loss=tf.losses.get_total_loss(False)
            tf.summary.scalar('total_loss',self.total_loss)

    def build_network(self):
        (self.feed('data')
         .cov(7, 7, 64, 2, 2, name='conv1_1',  init_from_modle=False, trainalble=True)
         .pool(2, 2, 2, 2, name='pool1')
         .cov(3, 3, 192, 1, 1, name='conv2_1', init_from_modle=False, trainalble=True)
         .pool(2, 2, 2, 2, name='pool2')
         .cov(1, 1, 128, 1, 1, name='conv3_1', init_from_modle=False, trainalble=True)
         .cov(3, 3, 256, 1, 1, name='conv3_2', init_from_modle=False, trainalble=True)
         .cov(1, 1, 256, 1, 1, name='conv3_3', init_from_modle=False, trainalble=True)
         .cov(3, 3, 512, 1, 1, name='conv3_4', init_from_modle=False, trainalble=True)
         .pool(2, 2, 2, 2, name='pool3')
         .cov(1, 1, 256, 1, 1, name='conv4_1', init_from_modle=False, trainalble=True)
         .cov(3, 3, 512, 1, 1, name='conv4_2', init_from_modle=False, trainalble=True)
         .cov(1, 1, 256, 1, 1, name='conv4_3', init_from_modle=False, trainalble=True)
         .cov(3, 3, 512, 1, 1, name='conv4_4', init_from_modle=False, trainalble=True)
         .cov(1, 1, 256, 1, 1, name='conv4_5', init_from_modle=False, trainalble=True)
         .cov(3, 3, 512, 1, 1, name='conv4_6', init_from_modle=False, trainalble=True)
         .cov(1, 1, 256, 1, 1, name='conv4_7', init_from_modle=False, trainalble=True)
         .cov(3, 3, 512, 1, 1, name='conv4_8', init_from_modle=False, trainalble=True)
         .cov(1, 1, 512, 1, 1, name='conv4_9', init_from_modle=False, trainalble=True)
         .cov(3, 3, 1024, 1, 1,name='conv4_10',init_from_modle=False, trainalble=True)
         .pool(2, 2, 2, 2, name='pool4')
         .cov(1, 1, 512, 1, 1, name='conv5_1', init_from_modle=False, trainalble=True)
         .cov(3, 3, 1024, 1, 1, name='conv5_2',init_from_modle=False, trainalble=True)
         .cov(1, 1, 512, 1, 1, name='conv5_3', init_from_modle=False, trainalble=True)
         .cov(3, 3, 1024, 1, 1, name='conv5_4',init_from_modle=False, trainalble=True)
         .cov(3, 3, 1024, 1, 1, name='conv5_5',init_from_modle=False, trainalble=True)
         .cov(3, 3, 1024, 2, 2, name='conv5_6',init_from_modle=False, trainalble=True)
         .cov(3, 3, 1024, 1, 1, name='conv6_1',init_from_modle=False, trainalble=True)
         .cov(3, 3, 1024, 1, 1, name='conv6_2',init_from_modle=False, trainalble=True)
         .fc(4096, name='fc_1', drop_out=True)
         .fc(self.out_size, name='fc_2'))

        ###   methods chain   ###
    def calculate_loss(self):

        predict = self.layers['fc_2']
        predict_confidence = tf.reshape(predict[ : ,: self.confidence_boundary],
                                     [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
        predict_boxes = tf.reshape(predict[: , self.confidence_boundary:self.boxes_boundary],
                                   [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
        predict_classes = tf.reshape(predict[: , self.boxes_boundary:],
                                        [self.batch_size, self.cell_size, self.cell_size, self.num_classes])

        #..... label  .....

        label_confidence = tf.reshape(self.label[..., 0], [self.batch_size, self.cell_size, self.cell_size, 1])

        label_boxes = tf.reshape(self.label[..., 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])

        # 把label中的坐标 对图像大小做归一化
        label_boxes = tf.tile(label_boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size

        label_classes = tf.reshape(self.label[..., 5:],
                                   [self.batch_size, self.cell_size, self.cell_size, self.num_classes])

        offset = [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell
        offset = np.array(offset)
        offset = np.reshape(offset, [self.boxes_per_cell, self.cell_size, self.cell_size])
        x_offset = np.transpose(offset, [1, 2, 0])
        x_offset = tf.reshape(tf.constant(x_offset, dtype=tf.float32),
                              [1, self.cell_size, self.cell_size, self.boxes_per_cell])
        x_offset = tf.tile(x_offset, [self.batch_size, 1, 1, 1])
        y_offset = tf.transpose(x_offset, [0, 2, 1, 3])

        # Normlize the predicted box by the image size for calculating the IOU
        predict_boxes_norm_for_image = tf.stack([(predict_boxes[..., 0] + x_offset) / self.cell_size,
                                                 (predict_boxes[..., 1] + y_offset) / self.cell_size,
                                                  tf.square(predict_boxes[..., 2]),
                                                  tf.square(predict_boxes[..., 3])],
                                                  axis=-1)
        # Transform the box into left upper  corner (x1,y1)  right down corner (x2,y2)
        label_boxes_4 = tf.stack([label_boxes[..., 0] - label_boxes[..., 2] / 2.0,
                                  label_boxes[..., 1] - label_boxes[..., 3] / 2.0,
                                  label_boxes[..., 0] + label_boxes[..., 2] / 2.0,
                                  label_boxes[..., 1] + label_boxes[..., 3] / 2.0],
                                  axis=-1)

        predict_boxes_4 = tf.stack([predict_boxes_norm_for_image[..., 0] - predict_boxes_norm_for_image[..., 2] / 2.0,
                                    predict_boxes_norm_for_image[..., 1] - predict_boxes_norm_for_image[..., 3] / 2.0,
                                    predict_boxes_norm_for_image[..., 0] + predict_boxes_norm_for_image[..., 2] / 2.0,
                                    predict_boxes_norm_for_image[..., 1] + predict_boxes_norm_for_image[..., 3] / 2.0],
                                    axis=-1)
        # get the intersetion box
        left_upper = tf.maximum(predict_boxes_4[..., :2], label_boxes_4[..., :2])
        right_down = tf.minimum(predict_boxes_4[..., 2:], label_boxes_4[..., 2:])
        intersection = tf.maximum(0.0, right_down - left_upper)


        area_intersection = intersection[..., 0] * intersection[..., 1]  #[batch_size,cell_size,cell_size,boxes_per_cell]
        union_area = tf.maximum(predict_boxes_norm_for_image[..., 2] * predict_boxes_norm_for_image[..., 3]
                                + label_boxes[..., 2] * label_boxes[..., 3] - area_intersection, 1e-10)
        iou = tf.clip_by_value(area_intersection / union_area, 0.0, 1.0)

        predictor = tf.reduce_max(iou, axis=3, keep_dims=True)

        predictor = tf.cast((iou >= predictor), tf.float32) * label_confidence  # label_confidence:  [batch_size, cell_size,cell_size,1]

        not_predictor = tf.ones_like(predictor, dtype=tf.float32) - predictor

        label_boxes_for_deta = tf.stack([label_boxes[..., 0] * self.cell_size - x_offset
                                            , label_boxes[..., 1] * self.cell_size - y_offset
                                            , tf.sqrt(label_boxes[..., 2])
                                            , tf.sqrt(label_boxes[..., 3])], axis=-1)

        coordinate_deta=(predict_boxes-label_boxes_for_deta)*tf.expand_dims(predictor,4)

        coordinate_loss=tf.reduce_mean(tf.reduce_sum(tf.square(coordinate_deta),axis=[1,2,3,4]))*self.coord_scale

        object_deta=(iou-predict_confidence)*predictor

        noobj_deta=predict_confidence*not_predictor

        object_loss=tf.reduce_mean(tf.reduce_sum(tf.square(object_deta),axis=[1,2,3]))*self.object_scale

        noobj_loss=tf.reduce_mean(tf.reduce_sum(tf.square(noobj_deta),axis=[1,2,3]))*self.noobj_scale

        class_deta=(predict_classes-label_classes)*label_confidence

        class_loss=tf.reduce_mean(tf.reduce_sum(tf.square(class_deta),axis=[1,2,3]))*self.class_scale

        tf.losses.add_loss(coordinate_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobj_loss)
        tf.losses.add_loss(class_loss)


        tf.summary.scalar('coordinate_loss', coordinate_loss)
        tf.summary.scalar('object_loss',object_loss)
        tf.summary.scalar('noobj_loss',noobj_loss)
        tf.summary.scalar('class_loss',class_loss)



















