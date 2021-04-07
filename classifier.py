import time
import glob
import numpy as np
import pandas as pd
import spacy
import copy
import sys
# !{sys.executable} -m pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch._autograd_functions
import torch.autograd as autograd
import torch.nn.functional as f

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = "cpu"

num_epochs = 100

test_tensor = torch.rand(DIMS)
train_data = torch.load("train_data.pt") 
#print(train_data)

class Classifier(nn.Module):
    def _init_(self, emb_size=768, att_size=144):
        super()._init_()
        # Inputs to hidden layers linear transformation

        self.quant_hidden = nn.Linear(emb_size, 256)
        self.ent_hidden = nn.Linear(emb_size, 256)
        self.quant_att = nn.Linear(2*att_size, 256)
        self.ent_att = nn.Linear(2*att_size, 256)

        self.relation = nn.Linear(256,256)
        
    def forward(self, x):
        quant_rep = f.tanh(self.quant_hidden(x[768:1536]) + self.quant_att(x[1536:]) )
        ent_rep = f.tanh( self.ent_hidden(x[:768]) + self.ent_att(x[1536:]) )

        ent = self.relation(ent_rep)
        prob = f.sigmoid(torch.dot(quant_rep, ent))
        return prob

# trainloader = torch.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True)

def train_model(model, criterion = nn.BCELoss()):
    model.train()
    optimizer = optim.SGD( model.parameters() , lr=0.001)
    
    for epoch in range(num_epochs):
        start = time.time()
        running_loss = 0
        outputs = []

        for _, series in train_data.iterrows():
            qId, eId, x, entLabel, propLabel = series
            optimizer.zero_grad()
            output = model(x)
            outputs.append(output)
            #target label
            # positive samples: 
            ent_label_tensor = torch.zeros(output.shape)
            if entLabel == 1:
                ent_label_tensor = torch.ones(output.shape)
        
            loss = criterion(output, ent_label_tensor)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            model.train()
            
        print( "Epoch %d, time = %0.4f, Loss = %0.4f"%( epoch+1, time.time() - start, running_loss ))

model = Classifier().to(device)
train_model(model)