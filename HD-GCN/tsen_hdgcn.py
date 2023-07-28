import random
import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from CDR.hdgcn.HDGCN import Model
from feeders.feeder_ntu import Feeder
from graph.ntu_rgb_d_hierarchy import Graph

model_path = '/cvhci/temp/prgc/hdgcn_filtered/CDR/output_40/results_cdr/model.pt'

# load dataset
def load_data():
    test_data_path="/cvhci/temp/prgc/st-gcn/data/clean/xview/test_data.npy"
    test_label_path="/cvhci/temp/prgc/st-gcn/data/clean/xview/test_label.pkl"

    test_dataset = Feeder(data_path=test_data_path, label_path=test_label_path)

    return test_dataset

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Evaluate the Model
def evaluate(test_loader, model1):
    model1.eval()  # Change model to 'eval' mode.
    # correct1 = 0
    # total1 = 0
    results = []
    labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.cuda()
            logits1 = model1(data)
            results.append(logits1.cpu().numpy())
            labels.append(label.cpu().numpy())
    
    return np.concatenate(results), np.concatenate(labels)
            
    #         outputs1 = F.softmax(logits1, dim=1)
    #         _, pred1 = torch.max(outputs1.data, 1)
    #         total1 += labels.size(0)
    #         correct1 += (pred1.cpu() == labels.long()).sum()

    #     acc1 = 100 * float(correct1) / float(total1)

    # return acc1
    

model = Model(graph=Graph(labeling_mode= 'spatial',CoM=1),compute_flops= True)
model.load_state_dict(torch.load(model_path))
model.fcn = nn.Sequential()
model.cuda()

test_loader = torch.utils.data.DataLoader(dataset=load_data(),
                                            batch_size=16,
                                            num_workers=3,
                                            drop_last=False,
                                            shuffle=False,
                                            worker_init_fn=init_seed)

results, labels = evaluate(test_loader, model)

results = np.array(object=results, dtype=np.float32)