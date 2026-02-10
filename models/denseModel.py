
import torch
import numpy as np
import time

from sklearn.preprocessing import StandardScaler

class DenseModel(torch.nn.Module): 
    def __init__(self, feature_size, pred_len, batch_size, hidden_layer_sizes=None):
        super(DenseModel, self).__init__()

        self.pred_len = pred_len
        self.feature_size = feature_size
        self.batch_size = batch_size 

        self.input_shape = (pred_len, feature_size)

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [512, 256, 128]

        l = hidden_layer_sizes

        seq_len = 96

        input_size = feature_size * seq_len
        output_size = feature_size * pred_len

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

    def save_params(self, name= None, filepath = None):
        if filepath is None:
            filepath = self.basepath + '/' +  self.model_folder

        if name is None:
            name = self.default_name

        filepath = filepath + '/' + name + '.pth'

        print(f'Weights of {self.model_folder} saved to {filepath} ')

    def load_params(self, path, device = 'cpu'):

        state_dict = torch.load(path, map_location = device)
        self.model.load_state_dict(state_dict)

        print(f'Weights loaded from {path}')


def train_dense(train_loader, feature_size=98, pred_len=24, batch_size=32, epochs=10, save = False, load_path = None, save_path = None):

    model = DenseModel(feature_size, pred_len, batch_size)

    loss_fn = model.loss
    optimizer = model.optimizer

    train_steps = len(train_loader)

    if load_path is not None:
        model.load_params(path = load_path)

        return model

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

    return model

