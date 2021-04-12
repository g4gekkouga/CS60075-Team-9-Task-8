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

from torch.nn import LogSoftmax

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

num_epochs = 1000
chkpt_dir = "./checkpoints/"

train_data = torch.load("train_data.pt") 
quantIdList = list(set(train_data["quantId"]))

##########################################################

def adaptive_loss(o_e,o_th,o_p, ent_labels, prop_labels):
    loss_fn = LogSoftmax(dim=-1)

    l_e = torch.squeeze(o_e, dim=1)
    l_p = torch.squeeze(o_p, dim=1)
    l_th = torch.squeeze(o_th,dim=1)

    logits = [l_e, l_p, l_th]
    logits = torch.stack(logits, dim=1)

    th_labels = torch.squeeze(torch.zeros(ent_labels.shape), dim=1)
    
    labels = [ torch.squeeze(ent_labels, dim=1), torch.squeeze(prop_labels,dim=1), th_labels]
    labels = torch.stack(labels, dim=1)

    th_label = torch.zeros(labels.shape)
    th_label[:, -1] = 1.0

    n_mask = 1-labels
    p_mask = labels + th_label

    # For Loss 1
    logit1 = logits - (1-p_mask)*1e30
    loss1 = -((loss_fn(logit1)*labels).sum(1))

    # For Loss 2
    logit2 = logits - (1-n_mask)*1e30
    loss2 = -((loss_fn(logit2)*th_label).sum(1))

    loss = loss1 + loss2
    loss = loss.mean()
    return loss




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

def train_model(entityRelClassifier, propertyRelClassifier, thresholdRelClassifier, criterion=adaptive_loss ):
    entityRelClassifier.train()
    propertyRelClassifier.train()
    thresholdRelClassifier.train()
    optimizer1 = optim.SGD( entityRelClassifier.parameters() , lr=0.005)
    optimizer2 = optim.SGD( propertyRelClassifier.parameters() , lr=0.005)
    optimizer3 = optim.SGD( thresholdRelClassifier.parameters() , lr=0.005)
    
    for epoch in range(1, 1+num_epochs):
        start = time.time()
        best_loss = 1e30
        running_loss = 0
        outputs = []
        
        for each_quant in quantIdList:
            #list of same quantId same batch
            curr_batch = train_data.loc[train_data['quantId'] == each_quant]

            x = list(curr_batch['X'])
            x = torch.stack(x,dim=0)
            # x = x.to(device)

            batch_len = x.shape[0]
            
            entLabel = torch.tensor(list(curr_batch['entLabel']), dtype=torch.float).reshape(-1,1)
            propLabel = torch.tensor(list(curr_batch['propLabel']), dtype=torch.float).reshape(-1,1)
            
            entityRelOutput = entityRelClassifier(x)
            propertyRelOutput = propertyRelClassifier(x)
            thresholdRelOutput = thresholdRelClassifier(x)
            #target label
            # positive samples: 

            loss = criterion(
                entityRelOutput,
                thresholdRelOutput,
                propertyRelOutput,
                entLabel,
                propLabel
            )

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()

            running_loss += loss
            
        if epoch%100 == 0:
            ent_filepath = chkpt_dir + "ent_model_"  + str(epoch/100)+".pt"
            prop_filepath = chkpt_dir + "prop_model_"  + str(epoch/100)+".pt"
            thresh_filepath = chkpt_dir + "thresh_model_" + str(epoch/100)+".pt"
            torch.save(entityRelClassifier.state_dict(), ent_filepath)
            torch.save(propertyRelClassifier.state_dict(), prop_filepath)
            torch.save(thresholdRelClassifier.state_dict(), thresh_filepath)
            
            
        print( "Epoch %d, time = %0.4f, Loss = %0.4f"%( epoch+1, time.time() - start, running_loss/len(quantIdList) ))


if __name__ == "__main__":
    entityModel = Classifier().to(device)
    propertyModel = Classifier().to(device)
    thresholdModel = Classifier().to(device)
    train_model(entityModel, propertyModel, thresholdModel)
