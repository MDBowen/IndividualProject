import argparse
import os
import torch
import random
import numpy as np
from Autoformer.utils.tools import dotdict


def return_parser():

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')#模型id
    parser.add_argument('--model', type=str, required=True, default='Autoformer',#选择模型
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')#数据类型
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')#数据文件夹路径
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')#具体文件
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')#预测类别
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')#不太懂 OT好像代表Output Target,要预测的单变量
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')#保存模型

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')#输入序列长度
    parser.add_argument('--label_len', type=int, default=48, help='start token length')#这个label_len未完全搞懂
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')#输出序列长度

    # model define
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')#Reformer专用属性
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')#Reformer专用属性
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')#encoder input size
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')#decoder input size
    parser.add_argument('--c_out', type=int, default=7, help='output size')#输出长度
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')#dimension of model
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')#num of heads 
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')#num of encoder layers
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')#num of decoder layers
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')#dimension of fcn
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')#窗口滑动平均数
    parser.add_argument('--factor', type=int, default=1, help='attn factor')#attn factor不太理解
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)#是否在encoder里面使用知识蒸馏
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')#dropout
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')#time features encoding不太能get到
    parser.add_argument('--activation', type=str, default='gelu', help='activation')#激活函数default=gelu
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')#encoder的output_attention是否输出
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')#是否预测未见的未来数据,也就是是否进行推理的意思

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')# num_workers是加载数据(batch)的线程数目
    parser.add_argument('--itr', type=int, default=2, help='experiments times')#实验次数
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')#就是epoch
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')#bathsize
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')#patience: 当early stop被激活(如发现loss相比上一个epoch训练没有下降)，则经过patience个epoch后停止训练
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')#lr
    parser.add_argument('--des', type=str, default='test', help='exp description')#test
    parser.add_argument('--loss', type=str, default='mse', help='loss function')#loss is mse
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')#adjust learning-rate
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)#使用自动混合精度训练

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # args = parser.parse_args()
    args = dotdict()
    args.target = 'OT'
    args.des = 'test'
    args.dropout = 0.05
    args.num_workers = 10
    args.gpu = 0
    args.lradj = 'type1'
    args.devices = '0'
    args.use_gpu = False
    args.use_multi_gpu = False
    # if args.use_gpu and args.use_multi_gpu: #是否使用多卡的判断
    #     args.dvices = args.devices.replace(' ', '')
    #     device_ids = args.devices.split(',')
    #     args.device_ids = [int(id_) for id_ in device_ids]
    #     args.gpu = args.device_ids[0]
    args.freq = 'h'
    args.checkpoints = './checkpoints/'
    args.bucket_size = 4
    args.n_hashes = 4
    args.is_trainging = True
    args.root_path = './dataset/ETT-small/'
    args.data_path ='ETTh1.csv' 
    args.model_id='ETTh1_96_24'
    args.model = 'Autoformer'
    args.data = 'ETTh1'
    args.features = 'M'
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 24
    args.e_layers = 2
    args.d_layers = 1
    args.n_heads = 8
    args.factor = 1
    args.enc_in = 7
    args.dec_in =7
    args.c_out = 7
    args.d_model = 512
    args.des = 'Exp'
    args.itr = 1
    args.d_ff = 2048
    args.moving_avg = 25
    args.factor = 1
    args.distil = True
    args.output_attention = False
    args.patience= 3
    args.learning_rate = 0.0001
    args.batch_size = 32 
    args.embed = 'timeF'
    args.activation = 'gelu'
    args.use_amp = False
    args.loss = 'mse'
    args.train_epochs = 10
    print('Args in experiment:')
    print(args)
    return args

