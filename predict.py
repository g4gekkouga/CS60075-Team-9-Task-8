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
import classifier

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

num_epochs = 1000
chkpt_dir = "./checkpoints/"

def get_prediction(entityModel, propertyModel, thresholdModel, test_data, test_quantIds):
    prediction_output = []

    annotSet = 0
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
        # quantity data
        prediction_output.append(
        [str(docId[0]), str(annotSet), "Quantity", str(quantStartOffSet[0]), str( int(quantStartOffSet[0]) + len(quantText[0])), str(annotSet)+"-1", str(quantText[0]), ""] ) 

        x = torch.stack(x_list,dim=0)
        entityRelProb = entityModel(x)
        propertyRelProb = propertyModel(x)
        thresholdRelProb = thresholdModel(x)

        entityRelList = entityRelProb.tolist()
        propertyRelList = propertyRelProb.tolist()
        thresholdRelList = thresholdRelProb.tolist()

        entity_flag = 0 
        entity_threshold_prob_ratio = 0
        entity_index = -1
        for i in range(len(curr_batch)):
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

        prediction_output.append(
        [str(docId[entity_index]), str(annotSet), "MeasuredEntity", str(entStartOffset[entity_index]), str( int(entStartOffset[entity_index]) + len(entText[entity_index])), str(annotSet)+"-2", str(entText[entity_index])] ) 


        property_flag = 0
        property_threshold_prob_ratio = 0
        property_index = -1
        for i in range(len(curr_batch)):
            if( propertyRelList[i][0] < thresholdRelList[i][0] ):
                continue

            property_flag = 1
            curr_ratio = propertyRelList[i][0]/thresholdRelList[i][0]
            if property_threshold_prob_ratio < curr_ratio:
                property_index = i
                property_threshold_prob_ratio = curr_ratio
            
        if property_flag == 1:
            prediction_output.append(
            [str(docId[property_index]), str(annotSet), "MeasuredProperty", str(entStartOffset[property_index]), str( int(entStartOffset[property_index]) + len(entText[property_index])), str(annotSet)+"-3", str(entText[property_index])] ) 

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
# ent_filepath = chkpt_dir + "ent_model_" + str(int(start/1e6)) + "_" + str(epoch/100)+".pt"
# prop_filepath = chkpt_dir + "prop_model_" + str(int(start/1e6)) + "_" + str(epoch/100)+".pt"
# thresh_filepath = chkpt_dir + "thresh_model_" + str(int(start/1e6)) + "_" + str(epoch/100)+".pt"

# entityRelClassifier.load_state_dict(torch.load(ent_filepath))
# propertyRelClassifier.load_state_dict(torch.load(prop_filepath))
# thresholdRelClassifier.load_state_dict(torch.load(thresh_filepath))

prediction_df = get_prediction(entityRelClassifier, propertyRelClassifier, thresholdRelClassifier, test_data, test_quantIds )

docId_list = list(set(prediction_df["docId"]))
for docId in docId_list:
    curr_doc = prediction_df.loc[prediction_df['docId'] == docId]
    file_name = "./results/"+docId+".tsv"
    curr_doc.to_csv(file_name, sep="\t", index=False)


