import time
import glob
import numpy as np
import pandas as pd
import spacy
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch._autograd_functions
import torch.autograd as autograd
import torch.nn.functional as f

device = "cpu"
DIMS = 1824
num_epochs = 100

test_tensor = torch.rand(DIMS)
train_data = torch.load("train_data.pt") 
quantIdList = list(set(train_data["quantId"]))

class Classifier(nn.Module):
    def __init__(self, emb_size=768, att_size=144):
        super().__init__()
        # Inputs to hidden layers linear transformation
        self.quant_hidden = nn.Linear(emb_size, 256)
        self.ent_hidden = nn.Linear(emb_size, 256)
        self.quant_att = nn.Linear(2*att_size, 256)
        self.ent_att = nn.Linear(2*att_size, 256)

        self.relation = nn.Linear(256,256)
        
    def forward(self, x):
        quant_rep = f.tanh(self.quant_hidden(x[:, 768:1536]) + self.quant_att(x[:, 1536:]) )
        ent_rep = f.tanh( self.ent_hidden(x[:, :768]) + self.ent_att(x[:, 1536:]) )

        ent = self.relation(ent_rep)
        prob = f.sigmoid(torch.dot(quant_rep, ent))
        return prob

def train_model(entityRelClassifier, propertyRelClassifier, thresholdRelClassifier, criterion = nn.BCELoss()):
    entityRelClassifier.train()
    propertyRelClassifier.train()
    thresholdRelClassifier.train()
    optimizer1 = optim.SGD( entityRelClassifier.parameters() , lr=0.01)
    optimizer2 = optim.SGD( propertyRelClassifier.parameters() , lr=0.01)
    optimizer3 = optim.SGD( thresholdRelClassifier.parameters() , lr=0.01)
    
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0
        outputs = []
        
        for each_quant in quantIdList:
            #list of same quantId same batch
            curr_batch = train_data.loc[train_data['quantId'] == each_quant]
            
            qId = list(curr_batch['quantId'])
            eId = list(curr_batch['entId'])
            x = list(curr_batch['X'])
            entLabel = list(curr_batch['entLabel'])
            propLabel = list(curr_batch['propLabel'])

            temp_list = []
            for lis in x:
                temp_list.append(lis.tolist())
            x = torch.tensor(temp_list)
            print(x.shape)
            
            optimizer1.zero_grad()
            #entity-quantity relation classifier
            entityRelOutput = entityRelClassifier(torch.FloatTensor(x))
            propertyRelOutput = propertyRelClassifier(torch.FloatTensor(x))
            thresholdRelOutput = thresholdRelClassifier(torch.FloatTensor(x))
            #target label
            # positive samples: 
            return
            ent_label_tensor = torch.zeros(output.shape)
            if entLabel == 1:
                ent_label_tensor = torch.ones(output.shape)

            loss = criterion( output, ent_label_tensor )
            loss.backward()
            optimizer1.step()
            running_loss += loss.item()
                #model.train()
            
        print( "Epoch %d, time = %0.4f, Loss = %0.4f"%( epoch+1, time.time() - start, running_loss ))

entityModel = Classifier().to(device)
propertyModel = Classifier().to(device)
thresholdModel = Classifier().to(device)
train_model(entityModel, propertyModel, thresholdModel)
