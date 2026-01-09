import torch
import numpy as np 
import torch.nn as nn 
from torch import optim

import os
import time

import warnings

class train_framework:
    def __init__(self, args, model):
        self.args = args
        self.model = model

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def train(self, data_loader):
        pass

    def predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input 
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs