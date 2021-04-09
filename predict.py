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
from classifier import Classifier
import sys

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

num_epochs = 1000
chkpt_dir = "./checkpoints/"

def get_prediction(entityModel, propertyModel, thresholdModel, test_data, test_quantIds):
    prediction_output = []

    for each_quant in test_quantIds:
        #enter quantity to results
        curr_batch = test_data.loc[test_data['uniqueId'] == each_quant]
        #extracting data of a single quantity
        annotSet = list(curr_batch['annotSet'])[0]
        qId = list(curr_batch['quantId'])
        docId = list(curr_batch['docId'])
        entId = list(curr_batch['entId'])
        quantStartOffSet = list(curr_batch['quantStartOffset'])
        quantText = list(curr_batch['quantText'])
        entStartOffset  = list(curr_batch['entStartOffset'])
        entText = list(curr_batch['entText'])
        x_list = list(curr_batch['X'])
        index_list = list(curr_batch.index)

        # print("{} {}".format(docId, qId))
        # quantity data
        pred_row = curr_batch.loc[index_list[0]]
        prediction_output.append(
        [pred_row['docId'], pred_row['annotSet'], "Quantity", pred_row['quantStartOffset'], pred_row['quantStartOffset'] + len(pred_row['quantText']), str(pred_row['annotSet'])+"-1", pred_row['quantText'], "{\"unit\": \"kg\"}"] ) 

        x = torch.stack(x_list,dim=0).to(device)
        entityRelProb = entityModel(x)
        propertyRelProb = propertyModel(x)
        thresholdRelProb = thresholdModel(x)

        entityRelList = entityRelProb.tolist()
        propertyRelList = propertyRelProb.tolist()
        thresholdRelList = thresholdRelProb.tolist()

        entity_flag = 0 
        entity_threshold_prob_ratio = 0
        entity_index = -1
        for i in range(len(index_list)):
            if( entityRelList[i][0] < thresholdRelList[i][0] ):
                continue
            
            entity_flag = 1
            curr_ratio = entityRelList[i][0]/thresholdRelList[i][0]
            if entity_threshold_prob_ratio < curr_ratio:
                entity_index = i
                entity_threshold_prob_ratio = curr_ratio

        #no entity and property for this quantity
        if entity_flag == 0:
            continue 

        pred_row = curr_batch.loc[index_list[entity_index]]
        prediction_output.append(
        [pred_row['docId'], pred_row['annotSet'], "MeasuredEntity", pred_row['entStartOffset'], pred_row['entStartOffset'] + len(pred_row['entText']), str(pred_row['annotSet'])+"-2", pred_row['entText'], "{\"HasQuantity\": "+ "\"{}".format(pred_row['annotSet']) +"-1\"}" ] ) 

        propertyRelList.pop(entity_index)
        index_list.pop(entity_index)

        property_flag = 0
        property_threshold_prob_ratio = 0
        property_index = -1
        for i in range(len(index_list)):
            if( propertyRelList[i][0] < thresholdRelList[i][0] ):
                continue

            property_flag = 1
            curr_ratio = propertyRelList[i][0]/thresholdRelList[i][0]
            if property_threshold_prob_ratio < curr_ratio:
                property_index = i
                property_threshold_prob_ratio = curr_ratio
            
        if property_flag == 1:
            pred_row = curr_batch.loc[index_list[property_index]]
            prediction_output.append(
            [pred_row['docId'], pred_row['annotSet'], "MeasuredProperty", pred_row['entStartOffset'], pred_row['entStartOffset'] + len(pred_row['entText']), str(pred_row['annotSet'])+"-3", pred_row['entText'], "{\"HasQuantity\": "+ "\"{}".format(pred_row['annotSet']) +"-1\"}" ] ) 

        

    prediction_output = pd.DataFrame(prediction_output, columns = ["docId", "annotSet", "annotType", "startOffset", "endOffset", "annotId", "text", "other"])
    return prediction_output

entityRelClassifier = Classifier().to(device)
propertyRelClassifier = Classifier().to(device)
thresholdRelClassifier = Classifier().to(device)
test_data = torch.load("test_data.pt")

identifierList = []
for _, row in test_data.iterrows():
    identifierList.append( str(row["docId"]) + "_" + str(row["quantId"]) )
test_data["uniqueId"] = identifierList
test_quantIds = list(set(test_data["uniqueId"]))

ent_filepath = chkpt_dir + sys.argv[1]
prop_filepath = chkpt_dir + sys.argv[2]
thresh_filepath = chkpt_dir + sys.argv[3]

entityRelClassifier.load_state_dict(torch.load(ent_filepath))
propertyRelClassifier.load_state_dict(torch.load(prop_filepath))
thresholdRelClassifier.load_state_dict(torch.load(thresh_filepath))

# print("models loaded")

prediction_df = get_prediction(entityRelClassifier, propertyRelClassifier, thresholdRelClassifier, test_data, test_quantIds )

docId_list = list(set(prediction_df["docId"]))
for docId in docId_list:
    curr_doc = prediction_df.loc[prediction_df['docId'] == docId]
    file_name = "./results/"+docId+".tsv"
    curr_doc.to_csv(file_name, sep="\t", index=False)


