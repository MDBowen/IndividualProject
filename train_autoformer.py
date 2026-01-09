
from utils.tools import dotdict
from exp.exp_main import Exp_Main as Exp

def get_test_config():
    args = dotdict()

    args.is_training = 1
    args.model_id = 'test'
    args.model = 'Autoformer'
    args.data = 'sp100_combined_close'
    args.root_path = 'get_data'
    args.data_path = 'sp100_combined_close.csv'
    args.features = 'M'
    args.target = 'M'
    args.frequency = 'd'
    args.checkpoints = '/.autoformer_checkpoints/'

    args.sequence_len = 96
    args.argument = label_len = 48
    args.pred_len = 24

    args.enc_in = None
    args.dec_in = None
    args.c_out = None
    args.d_model = 512
    args.n_heads = 8 
    args.e_layers = 2
    args.d_layers=1
    args.d_ff = None 
    args.moving_avg = 50 
    args.faction = 1 
    args.distil = True
    args.dropout = 0.05  
    args.embed = None # because unnedded for the sp100 data loader, would be 'timeF'

    args.activation = 'gelu'

    args.output_attention = True 
    args.do_predict = None
    
    args.num_workers = 1 
    args.itr = None 
    args.train_epochs = 10
    args.batch_size = 32 
    args.patience = 3 
    args.learning_rate = 0.0001
    args.des = 'test'
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False 

    args.use_gpu = False
    args.gpu = None 
    args.use_multi_gpu = False
    args.devices = None  
    

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


if __name__ == '__main__':

    args, setting = get_test_config()
    
    exp = Exp(args)
    print(f'Started training for {args.train_epochs}')
    exp.train(setting)
    print(f'Training call done, who knows if it worked?')


