from easydict import EasyDict as edict
import tensorflow as tf
import yolo.config as cfg
import pdb
def layer(op):
    def layer_decorated(self,*args,**kwargs):
        name=kwargs['name']
        if len(self.inputs)==0:
            raise RuntimeError("NO Input Avaliable  for this layer{}".format(name))
        elif len(self.inputs)==1:
            layer_input=self.inputs[0]
        else:
            layer_input=self.inputs
        layer_out=op(self,layer_input,*args,**kwargs)
        self.layers[name]=layer_out
        self.feed(layer_out)
        return self
    return layer_decorated


class Network(object):
    def __init__(self):
        self.inputs=[]
        self.layers=edict()
        self.keep_pro=tf.placeholder(tf.float32)
        self.conv_num_per_group = [0, 0, 1, 2, 6, 16, 22]
        self.weight_init_file=cfg.WEIGHT_INIT_PATH
        self.reader=tf.train.NewCheckpointReader(self.weight_init_file)
        

    def feed(self,*args):
        assert len(args)!=0
        self.inputs=[]
        for layer in args:
            if isinstance(layer,str):
                try:
                    layer=self.layers[layer]
                except:
                    print(self.layers.keys())
                    raise KeyError("Unknown layer name{}".format(layer))
            self.inputs.append(layer)
        return self
    @layer
    def cov(self,input,kernal_h,kernal_w,channel_out,strides_h,strides_w,name,relu=True,
            padding='SAME',trainalble=True,alpha=0.1,init_from_modle=False):
        assert padding in ('SAME','VAILD')
        channel_in = input.get_shape()[-1]
        if init_from_modle:
            with tf.variable_scope(name):
                weights,bias=self.get_weights(name=name)
                kernal=tf.get_variable('weights',
                                       initializer=weights,regularizer=tf.nn.l2_loss,trainable=trainalble)
                biases=tf.get_variable('biases',initializer=bias,trainable=trainalble,regularizer=tf.nn.l2_loss)
        else:
            with tf.variable_scope(name):
                kernal=tf.get_variable('weights',shape=[kernal_h,kernal_w,channel_in,channel_out],
                                       initializer=tf.contrib.layers.xavier_initializer(),trainable=trainalble)
                biases=tf.get_variable('biases',shape=[channel_out],initializer=tf.constant_initializer(0.0),trainable=trainalble,regularizer=tf.nn.l2_loss)
        conv=tf.nn.conv2d(input,kernal,[1,strides_h,strides_w,1],padding)
        if relu:
            return tf.nn.leaky_relu(tf.nn.bias_add(conv,biases),alpha=alpha,name='leaky_relu')
        return tf.nn.bias_add(conv,biases)
    @layer
    def pool(self, input, kernal_h, kernal_w, strides_h, strides_w, name, padding='SAME', max_ave='MAX'):
        assert padding in ('SAME','VAILD')
        if max_ave=='MAX':
            return tf.nn.max_pool(input,[1,kernal_h,kernal_w,1],[1,strides_h,strides_w,1],padding,name=name)
        else:
            return tf.nn.avg_pool(input,[1,kernal_h,kernal_w,1],[1,strides_h,strides_w,1],padding,name=name)
    # def leaky_relu(self,x,alpha=0.1):
    #     return tf.maximum(alpha*x,x)
    @layer
    def fc(self,input,output_num,name,relu=True,trainable=True,alpha=0.1,drop_out=False):
        shape=input.get_shape()
        if len(shape) == 4:
            size = shape[1].value * shape[2].value * shape[3].value
        else:
            size = shape[-1].value
        with tf.variable_scope(name):
            w=tf.get_variable('weights',shape=[size,output_num],trainable=trainable,initializer=tf.contrib.layers.xavier_initializer(),regularizer=tf.nn.l2_loss)
            b=tf.get_variable('biases',shape=[output_num],trainable=trainable,initializer=tf.constant_initializer(0.0),regularizer=tf.nn.l2_loss)
            x=tf.reshape(input,[-1,size])
            x=tf.nn.bias_add(tf.matmul(x,w),b)
            if relu:
                if drop_out:

                    return tf.nn.dropout(tf.nn.leaky_relu(x,alpha=alpha),keep_prob=self.keep_pro)
                else:
                    return tf.nn.leaky_relu(x,alpha=alpha)
            else:
                if drop_out:

                    return tf.nn.dropout(x,keep_prob=self.keep_pro)
                else:
                    return x

    def get_weights(self,name):
        group = int(name.split('_')[0][-1])
        layer = int(name.split('_')[-1])
        if group == 1:
            weights = self.reader.get_tensor('Variable')
            bias    = self.reader.get_tensor('Variable_1')
        else:
            weights = self.reader.get_tensor('Variable_' + str(self.conv_num_per_group[group] * 2 + (layer - 1) * 2))
            bias    = self.reader.get_tensor('Variable_' + str(self.conv_num_per_group[group] * 2 + (layer - 1) * 2 + 1))
        weights = tf.convert_to_tensor(weights)
        bias = tf.convert_to_tensor(bias)
        return weights, bias






