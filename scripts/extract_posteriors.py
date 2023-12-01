import math
import os
import sys

import kaldi_io
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dnn.torch_dataset import TorchSpeechDataset
from dnn.torch_dnn import TorchDNN
from dnn.torch_dnn import LinearWActivation

if len(sys.argv) < 3:
    print("USAGE: python extract_posteriors.py <MY_TORCHDNN_CHECKPOINT> <OUTPUT_DIR>")

CHECKPOINT_TO_LOAD = sys.argv[1]
OUT_DIR = sys.argv[2]

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
OUTPUT_ARK_FILE = os.path.join(OUT_DIR, "posteriors.ark")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#define the paths
TRAIN_ALIGNMENT_DIR = "exp/tri_ali_train"
TEST_ALIGNMENT_DIR = "exp/tri_ali_test"


def extract_logits(net, test_loader):       #pass the test-set into net to extract logits
    """Runs through the  test_loader and returns a
    tensor containing the logits (forward output) for each sample in the test set
    """
    # TODO: IMPLEMENT THIS FUNCTION
    net.eval() #evaluation mode
    acc = 0
    n_samples = 0   #initialize the variables
    logits = None
    with torch.no_grad():   #switch off the gradients becuase we are in evaluation mode
        for i, data in tqdm(enumerate(test_loader)):
            X_batch, y_batch = data
            out = net(X_batch)
            val, y_pred = out.max(1)    #take the max probability and the corresponding subphones(hidden state)
            acc += (y_batch == y_pred).sum().detach().item() # get accuracy
            n_samples += BATCH_SIZE
            if i == 0:
                logits = out.clone()
                continue
            logits = torch.cat((logits, out), 0)
        print(acc/n_samples)    #get accutracy to see aproximately the quality of our classifier
    return logits



trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')


scaler = StandardScaler()
scaler.fit(trainset.feats)  #standarization for better fitting

testset.feats = scaler.transform(testset.feats)
BATCH_SIZE = testset.__len__()  #define the mini-batch of the test-set

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

feature_dim = trainset.feats.shape[1]   #input dim is the dimension of the features vector
n_classes = int(trainset.labels.max() - trainset.labels.min() + 1)  #number of subphones(hidden states)

net=torch.load(CHECKPOINT_TO_LOAD, map_location="cpu").to(DEVICE)   #losd the best model and define the place of execution

logits = extract_logits(net, test_loader)


post_file = kaldi_io.open_or_fd(OUTPUT_ARK_FILE, 'wb')

start_index = 0
testset.end_indices[-1] += 1

for i, name in enumerate(testset.uttids):   #stroing the logits in format that kaldi can understand
    out = logits[start_index:testset.end_indices[i]].cpu().numpy()
    start_index = testset.end_indices[i]
    kaldi_io.write_mat(post_file, out, testset.uttids[i])
