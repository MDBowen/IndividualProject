
import torch
import numpy as np
import time

from sklearn.preprocessing import StandardScaler

from data_provider.data_factory import data_provider

class DenseModel(torch.nn.Module): 
    def __init__(self, args, hidden_layer_sizes = None):
        super(DenseModel, self).__init__()

        self.pred_len = args.pred_len
        self.feature_size = args.enc_in
        self.batch_size = args.batch 
        self.seq_len = args.seq_len

        self.input_shape = (self.seq_len, self.feature_size)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [1024, 512, 256]

        l = hidden_layer_sizes

        input_size = self.feature_size * self.seq_len
        output_size = self.feature_size * self.pred_len

        self.basepath = 'checkpoints'
        self.model_folder = 'denseModel'
        self.default_name = 'dense_model_checkpoint'
        self.default_path = self.basepath + '/' + self.model_folder + '/' + self.default_name + '.pth'

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, l[0]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(l[0], l[1]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(l[1], l[2]),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(l[2], output_size)   
        )

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.scaler = StandardScaler()

    def forward(self, x):
        y = self.model(x)
        if len(x.shape)<3:
            return y.reshape((1, self.pred_len, self.feature_size))

        return y.reshape((x.shape[0], self.pred_len, self.feature_size))
    
    def _predict(self, x, y, x_m, y_m):
        self.model.eval()
        with torch.no_grad():
            pred = self.forward(x)

        return pred, None

    def save_params(self, name= None, filepath = None):
        if filepath is None:
            filepath = self.basepath + '/' +  self.model_folder

        if name is None:
            name = self.default_name

        import os
        os.makedirs(filepath, exist_ok=True)

        filepath = filepath + '/' + name + '.pth'

        torch.save(self.model.state_dict(), filepath)

        print(f'Weights of {self.model_folder} saved to {filepath} ')

    def load_params(self, path, name = None, device = 'cpu'):

        if name is None:
            name = self.default_name

        path = path + '/' + name + '.pth'

        state_dict = torch.load(path, map_location = device)
        self.model.load_state_dict(state_dict)

        print(f'Weights loaded from {path}')


def train_dense(args, load_path = None, save_path = None):

    train_set, train_loader = data_provider(args, flag = 'train')
    df, df_raw = train_set.get_data_frame()
    print(f'Training Dense from {df_raw['date'].min()} to {df_raw['date'].max()} ')

    epochs = args.train_epochs

    model = DenseModel(args)
    model.scaler = train_set.scaler

    loss_fn = model.loss
    optimizer = model.optimizer

    train_steps = len(train_loader)


    if load_path is not None:
        try:
            model.load_params(path = load_path)
            return model
        except:
            print(f'Loading model from {load_path} failed, training the model instead')

    print(f'Training dense for {epochs} epochs')

    for epoch in range(epochs):
        train_loss = []
        model.train(True)

        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

            optimizer.zero_grad()

            batch_x = batch_x.float()
            batch_y = batch_y.float()
            output = model(batch_x)

            output = output[:, -model.pred_len:, :]
            batch_y = batch_y[:, -model.pred_len:, :]

            loss = loss_fn(output, batch_y)

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss = np.average(train_loss)

        print("Epoch: {} time: {}".format(epoch + 1, time.time() - epoch_time))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss,))

    if save_path is not None:
        model.save_params(filepath = save_path)

    model.model.eval()

    return model

