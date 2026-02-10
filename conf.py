

from utils.tools import dotdict

def get_single_asset_config(root_path, data_path, data_name):

    args, setting = get_train_config()

    args.data = 'yh_fin'
    args.root_path = root_path
    args.data_path = data_path

    #change if we want indicators
    args.enc_in = 1
    args.dec_out = 1
    args.c_out = 1

    return args, setting

def get_train_config():
    args = dotdict()

    args.is_training = 1
    args.model_id = 'test'
    args.model = 'Autoformer'
    args.data = 'sp100'
    args.root_path = 'data'
    args.data_path = 'sp100_combined_close.csv'
    args.features = 'M'
    args.target = 'M'
    args.frequency = 'D'
    args.freq = 'd'
    args.checkpoints = './autoformer_checkpoints/'

    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 24

    args.enc_in = 98
    args.dec_in =98
    args.c_out =98
    args.d_model = 512
    args.n_heads = 8 
    args.e_layers = 2
    args.d_layers=1
    args.d_ff = 2048 
    args.moving_avg = 50 
    args.faction = 1 
    args.distil = True
    args.dropout = 0.05  
    args.activation = 'gelu'

    args.output_attention = True 
    args.do_predict = None
    
    args.num_workers = 1 
    args.itr = None 
    args.train_epochs = 10
    args.batch_size = 32
    args.patience = 3 
    args.learning_rate = 0.0001
    args.des = 'train'
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False 

    args.use_gpu = False
    args.gpu = None 
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