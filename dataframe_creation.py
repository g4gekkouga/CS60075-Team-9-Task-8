import os
import glob

train_raw_files = glob.glob("train/text/*.txt")
train_tsv_files = glob.glob("train/tsv/*.tsv")
test_raw_files = glob.glob("test/text/*.txt")

def raw_files_to_df(train_raw_fies):
    '''converting raw training files to dataframes'''
    #dict to be exported as dataframe
    documents_of_interest = {
        'document_name' : [],
        'sentence' : [],
        'potential_measurement' : []
    }
    #filling the dict
    for raw_file in train_raw_files:
        with open(raw_file, 'r') as file:
            print(file.ne.split('/')[2])
            break


    return
    # return modified_dataframe

raw_files_to_df(train_raw_files)