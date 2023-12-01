import torch
import torch.nn as nn

class LinearWActivation(nn.Module): #class for the input and hidden layer
    def __init__(self, in_features, out_features, dropout_p, batch_norm, affine):
        super(LinearWActivation,self).__init__()
        self.batch_norm = batch_norm    #if batch_norm == True then inmplement batch-normalization
        if self.batch_norm == True:
            self.m = nn.BatchNorm1d(in_features, affine=affine)
        self.f = nn.Linear(in_features,out_features)  #in_features = input_dimension, linear transformation
        self.d = nn.Dropout(p=dropout_p)    #dropout reguralization
        self.a = nn.ReLU()  #activation function
    def forward(self,x):    #forward function that define the path of the data
        if self.batch_norm == True:
            return self.a(self.d(self.f(self.m(x))))
        return self.a(self.d(self.f(x)))

class TorchDNN(nn.Module):
    """Create a DNN to extract posteriors that can be used for HMM decoding
    Parameters:
        input_dim (int): Input features dimension
        output_dim (int): Number of classes
        num_layers (int): Number of hidden layers
        batch_norm (bool): Whether to use BatchNorm1d after each hidden layer
        hidden_dim (int): Number of neurons in each hidden layer
        dropout_p (float): Dropout probability for regularization
    """

    def __init__(
        self, input_dim, output_dim, num_layers=2, batch_norm=True, hidden_dim=256, dropout_p=0.2, affine=False
    ):
        super(TorchDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers_in = [self.input_dim] + num_layers*[hidden_dim]  #construct such hidden layers as we defined
        layers_out = num_layers*[hidden_dim] + [self.output_dim]
        self.f = nn.Sequential(
            *[LinearWActivation(in_feats, out_feats, dropout_p, batch_norm, affine) 
             for in_feats, out_feats in zip(layers_in,layers_out)])
        self.clf = nn.Linear(self.output_dim, self.output_dim)  #outout layer, only linear transformation
        
    def forward(self, x):
        return self.clf(self.f(x))