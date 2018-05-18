import sys
import os
sys.path.append(os.getcwd())
from yolo.yolo_net import Yolo_net
from yolo.tfrecord_data_prepare import Pasval_VOC
import tensorflow as tf
import yolo.config as cfg
import numpy as np
import pdb
import time
#import os
class Train_Sovler(object):
    def __init__(self,restore=cfg.TRAIN_RESTORE):
        self.net=Yolo_net(training=True)
        self.restore=restore
        self.learning_rate=cfg.LEARNING_RATE
        self.max_iter_step=cfg.ITE_STEP
        self.start_step=0
        self.display_step=cfg.DISPLAY_STEP
        self.save_step=cfg.SAVE_STEP
        self.log_step=cfg.LOG_STEP
        self.log_dir=cfg.LOG_DIR
        self.weight_file=cfg.WEIGHT_FILE
        self.restore_path=cfg.RESTORE_PATH
        self.momentum=cfg.MOMENTUM
        self.global_step=tf.train.create_global_step()
        self.saver=tf.train.Saver(max_to_keep=400)
        self.summary_op=tf.summary.merge_all()
        self.gpu_options = tf.GPUOptions()
        self.config = tf.ConfigProto(gpu_options=self.gpu_options)
        self.sess = tf.Session(config=self.config)
        self.train_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=self.momentum).minimize(self.net.total_loss, global_step=self.global_step)
        self.writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
        #self.optimizer=tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=self.momentum)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        if self.restore:
            ckpt=tf.train.get_checkpoint_state(self.restore_path)
            print('Restoreing from {}......'.format(ckpt.model_checkpoint_path))
            stem = os.path.basename(ckpt.model_checkpoint_path)
            restore_iter = int(stem.split('.')[0].split('-')[-1])
            self.start_step=restore_iter
            self.sess.run(self.global_step.assign(restore_iter))
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
        


    def close_sess(self):
        self.sess.close()


if __name__ == '__main__':
    train = Train_Sovler()
    data = Pasval_VOC(phase='train', year='2007')
    image_batch, label_batch = data.read_batches_from_tfrecord()
    

   #  variable_names=[v.name for v in tf.global_variables()]
   #  for variable in variable_names:
   # #     value=train.sess.run(variable)
   #       print(variable)
   #     print(value.shape)
    #value=train.sess.run(variale_names)
    #print(variale_names)
    # print(value[0])
    # print(value[0].shape)
    # train.close_sess()

    coordinate = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coordinate, sess=train.sess)
    print('Strat training......................')

    for step in range(train.start_step+1, train.max_iter_step+1):
        start_time = time.time()

       # print(train.sess.run(train.global_step))
        if coordinate.should_stop():
            break
        image, label = train.sess.run([image_batch, label_batch])
        fetch_list = [train.train_op, train.net.total_loss,train.net.layers['fc_2']]
        feed_dict = {train.net.data: image, train.net.label: label, train.net.keep_pro: cfg.KEEP_PRO}
        _, total_loss,fc2 = train.sess.run(fetches=fetch_list, feed_dict=feed_dict)
        end_time = time.time()

        if step % train.save_step == 0:
           save_path= train.saver.save(train.sess, train.weight_file, global_step=step)
           print('Save modle into {}....'.format(save_path))
        if step % train.log_step == 0:
            summary = train.sess.run(train.summary_op, feed_dict=feed_dict)
            train.writer.add_summary(summary, global_step=step)
        if step % train.display_step == 0:

            per_iter_time=end_time-start_time
            print("step:   {:.0f}          total_loss:  {:.5f}            learning_rate: {:.7f}       {:.2f} s/iter".format(step,total_loss,train.learning_rate,per_iter_time))

       #else:
       #    pass

        #if step == 24000:
        #    train.learning_rate = 0.0001
            #train.optimizer=tf.train.MomentumOptimizer(learning_rate=train.learning_rate,momentum=train.momentum)
        #elif step==33600:
        #    train.learning_rate = 0.00001
       # #else:
        #    pass

    coordinate.join(threads)
    coordinate.request_stop()
    train.close_sess()







