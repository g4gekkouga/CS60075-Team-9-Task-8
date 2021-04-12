# Conditional Random Fields based model for subtask-1 and subtask-3

python >= 3.7\
`pip install -r requirements.txt`\
`python -m spacy download en_core_web_sm`

## Executing the code

## Subtask - 1

A python notebook `Task1.ipynb` is includes with included cell outputs \
All the preprocess, train and predic code is included in a single file `Task1.py`\
To execute the file : `python Task1.py`\
The output TSV files will be saved in `results_task1` folder\
`task1_evaluation.txt` includes the output on executing evaluation script given in the official git repository on our results

## Subtask - 3

A python notebook `Task3.ipynb` is includes with included cell outputs \
All the preprocess, train and predic code is included in a single file `Task3.py`\
To execute the file : `python Task3.py`\
The output TSV files will be saved in `results_task3` folder\
`task3_evaluation.txt` includes the output on executing evaluation script given in the official git repository on our results

 
## Other Files

`train`, `trial` and `eval` are the official data folders from the MeasEval git repository \
`final_result.png` - Evaluation script output on the final codalab submission