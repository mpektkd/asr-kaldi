# Train a torch DNN for Kaldi DNN-HMM model

import math
import sys

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dnn.torch_dataset import TorchSpeechDataset
from dnn.torch_dnn import TorchDNN
from dnn.torch_dnn import LinearWActivation

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')
# CONFIGURATION #

NUM_LAYERS = 2      #hyperparameters of the neural net
HIDDEN_DIM = 256
USE_BATCH_NORM = True
DROPOUT_P = .2
EPOCHS = 50
PATIENCE = 3
ETA = 1e-1
BATCH_SIZE=128
AFFINE=True #define if the parameters of batch_normalization are learnable or not
if len(sys.argv) < 2:
    print("USAGE: python timit_dnn.py <PATH/TO/CHECKPOINT_TO_SAVE.pt>")

BEST_CHECKPOINT = sys.argv[1]


#define the paths
TRAIN_ALIGNMENT_DIR = "exp/tri_ali_train"
DEV_ALIGNMENT_DIR = "exp/tri_ali_dev"
TEST_ALIGNMENT_DIR = "exp/tri_ali_test"

class EarlyStopping:#class for the early stopping reguralization
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path=BEST_CHECKPOINT):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta  #definition of the minimum tolerance
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None: #check if it is the first epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:  #if there is no advance then increase counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:   #if counter == patience then stop
                self.early_stop = True
        else:
            self.best_score = score #else save the best model till now
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)
        self.val_loss_min = val_loss
    
    def stopping(self):
        return self.early_stop

def train(net, criterion, optimizer, train_loader, dev_loader, epochs=50, patience=3):
    """Train model using Early Stopping and save the checkpoint for
    the best validation loss
    """
    # TODO: IMPLEMENT THIS FUNCTION
    early = EarlyStopping(patience=patience, verbose=True, path=BEST_CHECKPOINT)
    for epoch in tqdm(range(epochs)):
        net.train() #gradients on
        for i, data in tqdm(enumerate(train_loader)):   #for loop for pass the mini-batches into net
            X_batch, y_batch = data
            optimizer.zero_grad() # ALWAYS USE THIS!! 
            out = net(X_batch)  
            loss = criterion(out, y_batch)  #compute the loss
            loss.backward() #compute the gradients
            optimizer.step()    #update the weights

        net.eval()#evaluation mode
        running_average_loss = 0
        acc = 0
        n_samples = 0
        with torch.no_grad(): 
            for i, data in tqdm(enumerate(dev_loader)):
                X_batch, y_batch = data
                out = net(X_batch)
                loss = criterion(out, y_batch)
                val, y_pred = out.max(1)
                acc += (y_batch == y_pred).sum().detach().item() # get accuracy
                n_samples += BATCH_SIZE
                running_average_loss += loss.detach().item()
            print(acc/n_samples)
            early.__call__(running_average_loss, net)   #call the object early for checking the advance
        if early.stopping() == True:    #if true then stop the training to avoid overfitting
                break


trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
validset = TorchSpeechDataset('./', DEV_ALIGNMENT_DIR, 'dev')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')

scaler = StandardScaler()
scaler.fit(trainset.feats)

trainset.feats = scaler.transform(trainset.feats)
validset.feats = scaler.transform(validset.feats)
testset.feats = scaler.transform(testset.feats)

feature_dim = trainset.feats.shape[1]
n_classes = int(trainset.labels.max() - trainset.labels.min() + 1)

net = TorchDNN( #create object
    feature_dim,
    n_classes,
    num_layers=NUM_LAYERS,
    batch_norm=USE_BATCH_NORM,
    hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P,
    affine=AFFINE
)
net.to(DEVICE)
print(f"The network architecture is: \n {net}")
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)    #make tyhe data loadble(shuffle them) to pass them into net
dev_loader = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.SGD(net.parameters(), lr=ETA) #stochastic gradient descent for optimizer
criterion = nn.CrossEntropyLoss()   #CE loss is the common-used for classifiers

train(net, criterion, optimizer, train_loader, dev_loader, epochs=EPOCHS, patience=PATIENCE)
