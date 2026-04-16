
from utils.tools import dotdict
import torch
from exp.exp_main import Exp_Main

def create_config(model, feature_dim, seq_len=96, pred_len=24, label_len=48, epochs=10, batch_size=32, learning_rate=0.0001):

    
    args = dotdict()

    args.is_training = 1
    args.model_id = 'test'
    args.model = model
    args.data = None
    args.root_path = None
    args.data_path = None
    args.features = 'M'
    args.target = 'M'
    args.frequency = 'D'
    args.freq = 'd'
    args.checkpoints = './autoformer_checkpoints/'

    args.seq_len = seq_len
    args.label_len = label_len
    args.pred_len = pred_len

    args.enc_in = feature_dim
    args.dec_in = feature_dim
    args.c_out = feature_dim
    args.d_model = 512
    args.n_heads = 8 
    args.e_layers = 2
    args.d_layers=1
    args.d_ff = 2048 
    args.moving_avg = 50 
    args.faction = 1 
    args.distil = True
    args.dropout = 0.1
    args.activation = 'gelu'
    args.scale = True

    args.output_attention = True 
    args.do_predict = None
    
    args.num_workers = 1 
    args.itr = None 
    args.train_epochs = epochs
    args.batch_size = batch_size
    args.patience = 3 
    args.learning_rate = learning_rate # default is 0.0001
    args.des = 'train'
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False 

    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0 if args.use_gpu else None
    args.use_multi_gpu = False
    args.devices = None  

    args.embed = 'timeF'
    args.factor = 3
    

    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, 0)
    

    return args, setting 

def get_model(**kwargs):

    model_name = kwargs.get('model_name')
    feature_dim = kwargs.get('feature_dim')
    seq_len = kwargs.get('seq_len', 96)
    pred_len = kwargs.get('pred_len', 24)
    label_len = kwargs.get('label_len', 48)
    epochs = kwargs.get('epochs', 10)
    batch_size = kwargs.get('batch_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.0001)
    
    args, setting = create_config(model_name, feature_dim, seq_len, pred_len, label_len, epochs, batch_size, learning_rate)

    if feature_dim is None:
        raise ValueError('feature_dim must be provided in model_kwargs')
    

    if model_name in ['Transformer', 'Autoformer', 'Informer']:
        from models.Transformer import Model
        model = Exp_Main(args)

    elif model_name == 'Dense':
        from models.denseModel import DenseModel
        model = DenseModel(args)
    else:
        raise ValueError(f'Model {model_name} not recognized. Must be one of Transformer, Autoformer, Informer, Dense')

    return model