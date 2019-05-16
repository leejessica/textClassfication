# coding:utf8
import time
import warnings
import tensorflow as tf
import numpy as np
import json
import os

tfmt = '%m%d_%H%M%S'


class config(object):
    """
    Not all the configuration parameters is effective, In actual operation, obtain parameters as required
    """

    # data path

    char_embedding_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/char_embedding.npz'
    word_embedding_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/word_embedding.npz'
    train_data_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/train.npz'
    labels_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/labels.json'
    test_data_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/test.npz'

    with open(word_embedding_path, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(char_embedding_path, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(labels_path, "r") as fh:
        labels=np.array(json.load(fh), dtype=np.float32)
    # with open(train_data_path, "r") as fh:
    #     train_eval_file = json.load(fh)
    # with open(config.dev_eval_file, "r") as fh:
    #     dev_eval_file = json.load(fh)
    # with open(config.dev_meta, "r") as fh:
    #     meta = json.load(fh)



    # Set universal hyperparameter
    is_demo=False
    batch_size = 256
    epoches = 64
    max_sequence_length = 400
    use_word = True
    use_char = True
    word_dim = 300
    char_dim = 100
    num_classes = 100
    learning_rate = 1e-3
    decay_rate_big = 0.02
    drop_keep_pro = 0.5
    initializer = tf.random_normal_initializer(stddev=0.1)
    clip_gradients = 5.0
    multi_label_flag=False
    decay_rate=0.96
    decay_steps=100
    epoches=10000
    is_demo=True
    period=100
    checkpoint=1000
    model_dir='model'
    data_dir='data'
    log_dir='log'


    # ------------------------------------------------------------------
    # Set hyperparameter for textCNN
    num_filters = 200  # number of convolutional filters in one layer
    filter_sizes = [1, 2, 3, 4, 5]
    linear_hidden_size = 2000  # number of cell in fully-connected layer
    pool_size=2

    textCNN_path=os.path.join(model_dir, "textCNN")
    textCNN_log=os.path.join(log_dir, "textCNN")


    num_classes = 1999  # number of classes
    embedding_dim = 256  # embedding size

    kmax_pooling = 2  # k
    hidden_size = 256  # LSTM hidden size
    num_layers = 2  # LSTM layers
    inception_dim = 512  # inception的卷积核数

    # vocab_size = 11973 # num of chars
    vocab_size = 411720  # num of words
    kernel_size = 3  # 单尺度卷积核
    kernel_sizes = [2, 3, 4]  # 多尺度卷积核
    title_seq_len = 50  # 标题长度,word为30 char为50
    content_seq_len = 250  # 描述长度 word为120 char为250
    type_ = 'word'  # word 和char
    all = False  # 模型同时训练char和word

    embedding_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/char_embedding.npz'  # Embedding
    train_data_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/train.npz'  # train
    labels_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/labels.json'  # labels
    test_data_path = '/mnt/7/zhihu/ieee_zhihu_cup/data/test.npz'  # test
    result_path = 'csv/' + time.strftime(tfmt) + '.csv'
    shuffle = True  # 是否需要打乱数据
    num_workers = 4  # 多线程加载所需要的线程数目
    pin_memory = True  # 数据从CPU->pin_memory—>GPU加速
    batch_size = 128

    env = time.strftime(tfmt)  # Visdom env
    plot_every = 10  # 每10个batch，更新visdom等

    max_epoch = 100
    lr = 5e-3  # 学习率
    lr2 = 1e-3  # embedding层的学习率
    min_lr = 1e-5  # 当学习率低于这个值，就退出训练
    lr_decay = 0.99  # 当一个epoch的损失开始上升lr = lr*lr_decay
    weight_decay = 0  # 2e-5 # 权重衰减
    weight = 1  # 正负样本的weight
    decay_every = 3000  # 每多少个batch 查看一下score,并随之修改学习率

    model_path = None  # 如果有 就加载
    optimizer_path = 'optimizer.pth'  # 优化器的保存地址

    debug_file = '/tmp/debug2'  # 若该文件存在则进入debug模式
    debug = False

    gpu1 = False  # 如果在GPU1上运行代码,则需要修改数据存放的路径
    floyd = False  # 服务如果在floyd上运行需要修改文件路径
    zhuge = False  # 服务如果在zhuge上运行,修改文件路径

    ### multimode 用到的
    model_names = ['MultiCNNTextBNDeep', 'CNNText_inception', 'RCNN', 'LSTMText', 'CNNText_inception']
    model_paths = ['checkpoints/MultiCNNTextBNDeep_0.37125473788', 'checkpoints/CNNText_tmp_0.380390420742',
                   'checkpoints/RCNN_word_0.373609030286', 'checkpoints/LSTMText_word_0.381833388089',
                   'checkpoints/CNNText_tmp_0.376364647145']  # ,'checkpoints/CNNText_tmp_0.402429167301']
    static = False  # 是否训练embedding
    val = False  # 跑测试集还是验证集?

    fold = 1  # 数据集fold, 0或1 见 data/fold_dataset.py
    augument = True  # 是否进行数据增强

    ###stack
    model_num = 7
    data_root = "/data/text/zhihu/result/"
    labels_file = "/home/a/code/pytorch/zhihu/ddd/labels.json"
    val = "/home/a/code/pytorch/zhihu/ddd/val.npz"


def parse(self, kwargs, print_=True):
    '''
    根据字典kwargs 更新 config参数
    '''
    for k, v in kwargs.iteritems():
        if not hasattr(self, k):
            raise Exception("opt has not attribute <%s>" % k)
        setattr(self, k, v)

        ###### 根据程序在哪台服务器运行,自动修正数据存放路径 ######
    if self.gpu1:
        self.train_data_path = '/mnt/zhihu/data/train.npz'
        self.test_data_path = '/mnt/zhihu/data/%s.npz' % ('val' if self.val else 'test')
        self.labels_path = '/mnt/zhihu/data/labels.json'
        self.embedding_path = self.embedding_path.replace('/mnt/7/zhihu/ieee_zhihu_cup/', '/mnt/zhihu/')

    if self.floyd:
        self.train_data_path = '/data/train.npz'
        self.test_data_path = '/data/%s.npz' % ('val' if self.val else 'test')
        self.labels_path = '/data/labels.json'
        self.embedding_path = '/data/char_embedding.npz'
    if self.zhuge:
        self.train_data_path = './ddd/train.npz'
        self.test_data_path = './ddd/%s.npz' % ('val' if self.val else 'test')
        self.labels_path = './ddd/labels.json'
        self.embedding_path = './ddd/char_embedding.npz'

    ### word和char的长度不一样 ##
    if self.type_ == 'word':
        self.vocab_size = 411720  # num of words
        self.title_seq_len = 30
        self.content_seq_len = 120
        self.embedding_path = self.embedding_path.replace('char', 'word') if self.embedding_path is not None else None

    if self.type_ == 'char':
        self.vocab_size = 11973  # num of words
        self.title_seq_len = 50
        self.content_seq_len = 250

    if self.model_path:
        self.embedding_path = None

    if print_:
        print('user config:')
        print('#################################')
        for k in dir(self):
            if not k.startswith('_') and k != 'parse' and k != 'state_dict':
                print
                k, getattr(self, k)
        print('#################################')
    return self


def state_dict(self):
    return {k: getattr(self, k) for k in dir(self) if not k.startswith('_') and k != 'parse' and k != 'state_dict'}


config.parse = parse
config.state_dict = state_dict
opt = config()
