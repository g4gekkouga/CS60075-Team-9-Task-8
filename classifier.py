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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

num_epochs = 100

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
        quant_rep = torch.tanh(self.quant_hidden(x[:, 768:1536]) + self.quant_att(x[:, 1536:]) )
        ent_rep = torch.tanh( self.ent_hidden(x[:, :768]) + self.ent_att(x[:, 1536:]) )

        ent = self.relation(ent_rep)
        prob = torch.sigmoid(torch.bmm(quant_rep.reshape(-1,1,256), ent.reshape(-1,256,1)).reshape(-1,1))
        return prob
        #shape of prob is (batch_len, 1)

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
            eId = np.array(curr_batch['entId'])

            x = list(curr_batch['X'])
            x = torch.stack(x,dim=0)
            
            entLabel = torch.tensor(list(curr_batch['entLabel']), dtype=torch.int32)
            propLabel = torch.tensor(list(curr_batch['propLabel']), dtype=torch.int32)
            
            optimizer1.zero_grad()
            #entity-quantity relation classifier
            entityRelOutput = entityRelClassifier(x)
            propertyRelOutput = propertyRelClassifier(x)
            thresholdRelOutput = thresholdRelClassifier(x)
            #target label
            # positive samples: 
            
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
