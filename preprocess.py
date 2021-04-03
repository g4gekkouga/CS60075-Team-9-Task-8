import glob
import numpy as np
import pandas as pd

train_raw_files = glob.glob("train/text/*.txt")
train_tsv_files = glob.glob("train/tsv/*.tsv")

each_file_df = []
for tsv_file in train_tsv_files:
    each_file_df.append(pd.read_csv(tsv_file, sep="\t", header=0))

train_tsv_dataframe = pd.concat(each_file_df)

def get_ent_quant_dfs(tsv_df):
    entities_df = []
    quantities_df = []
    
    for _, tsv_row in tsv_df.iterrows():
        if tsv_row['annotType'] == 'Qualifier':
            continue

        temp_list = []
        temp_list.append(tsv_row['docId'])

        temp_list.append(tsv_row['annotSet'])
        temp_list.append(tsv_row['startOffset'])
        temp_list.append(tsv_row['endOffset'])
        temp_list.append(tsv_row['text'])

        if tsv_row['annotType'] == 'Quantity':
            quantities_df.append(temp_list)

        elif tsv_row['annotType'] == 'MeasuredProperty' or tsv_row['annotType'] == 'MeasuredEntity':
            entities_df.append(temp_list)

    entities_df = pd.DataFrame(entities_df, columns=['docId','annotSet','startOffset','endOffset','text'])
    quantities_df = pd.DataFrame(quantities_df, columns=['docId','annotSet','startOffset','endOffset','text'])

    return entities_df, quantities_df

ent_df, quant_df = get_ent_quant_dfs(train_tsv_dataframe)
# print(ent_df)
# print(quant_df)