# CS60075-Team-9-Task-8

python >= 3.7
`pip install -r requirements.txt`
`python -m spacy download en_core_web_sm`

## Preprocessing

The command to generate tensors from train data is `python3 preprocess.py train`.  
This generates ***train_data.pt*** in the same directory and it contains the train data in its most ready-to-use form i.e., X vector and y label.  
The command to generate tensors from test data is `python3 preprocess.py test`.  
This generates ***test_data.pt*** in the same directory and it contains the test data i.e., X vector.  

After generating the preprocessed data, you need to make sure that the checkpoints/ subdirectory exists where the trained model will be saved.  

## Training

Before training, make sure that the train data is in the subdirectory train/text/.  
To train the model, run the command `python3 classifier.py`.  
This trains the model for 1000 epochs and the model will be checkpointed every 100 epochs in the checkpoints/ directory.  

## Predict

Before predicting, make sure that the test data is in the subdirectory test/text/.  
To predict, run the command `python3 predict.py <ent-model-name> <prop-model-name> <thresh-model-name>`.

* ent-model-name is the name of the file in the checkpoint directory ent_model_\<it_no\>.pt

* prop-model-name is the name of the file in the checkpoint directory prop_model_\<it_no\>.pt

* thresh-model-name is the name of the file in the checkpoint directory thresh_model_\<it_no\>.pt

Where it_no is the number of iterations the model is trained divided by 10.  
In our case , the best choice is `python3 predict.py ent_model_10.0.pt prop_model_10.0.pt thresh_model_10.0.pt`

It generates the tsv files in the results/ subdirectory.  
