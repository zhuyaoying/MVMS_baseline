# -*- coding: utf-8 -*-
'''
@time: 2021/4/16 18:45

@ author:
'''

class Config:

    seed = 10

    # path
    datafolder = '../../data/CPSC/'

    #
    experiment = 'cpsc'

    # for train
    '''
    MyNet6View, resnet1d_wang, xresnet1d101, inceptiontime, fcn_wang, lstm, lstm_bidir, vit, mobilenetv3_small
    '''
    # PatchTST, TimesNet, iTransformer, MambaSL (Linux+CUDA only)
    model_name = 'MyNet6View'

    model_name2 = 'MyNet'

    # ECG sequence length (100 Hz → 1000, 500 Hz → 5000)
    seq_len = 1000

    # dropout rate for TSL baseline models
    dropout = 0.1

    batch_size = 64

    max_epoch = 100

    lr = 0.001

    device_num = 1

    # eg: MyNet6View_all_checkpoint_best_tpr.pth
    checkpoints = 'MyNet6View_exp0_checkpoint_best_auc.pth'

    # knowledge distillation param
    alpha = 0.5
    temperature = 2


config = Config()
