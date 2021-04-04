import glob
import numpy as np
import pandas as pd
import spacy
import torch
from Task1 import raw_text_to_df
from transformers import BertTokenizer, BertModel

def insert_star(x, sentence):
    i = 1
    if x[0][0] > x[1][0]:
        i = 0

    while len(x[0]) > 0 or len(x[1]) > 0:
        if i == 1:
            sentence = sentence[:x[1][0]] + " *" + sentence[x[1][0]:]
            x[1].pop(0)

        else:
            sentence = sentence[:x[0][0]] + "* " + sentence[x[0][0]:]
            x[0].pop(0)

        if len(x[0]) > 0 and len(x[1]) > 0:
            if (x[0][0] > x[1][0]):
                i = 0
            else:
                i = 1
        else:
            if len(x[0]) > 0:
                i = 0
            
            if len(x[1]) > 0:
                i = 1

    return sentence

def get_ent_quant_dfs(text_df, tsv_df):
    entities_df = []
    quantities_df = []

    for _, text_row in text_df.iterrows():

        # list of words and their spans
        sentence_tag_placeholders = []
        for word in text_row["sentence"]:
            sentence_tag_placeholders.append(
                ["O", (word.idx, (word.idx + len(word)))]
            )
        
        document_name = text_row['document_name']
        doc_id = tsv_df["docId"] == document_name

        # insert * in sentence

        list_of_before_star_offsets = []
        list_of_after_star_offsets = []

        for _, annot_row in tsv_df[doc_id].iterrows():
            list_of_before_star_offsets.append(annot_row["startOffset"] - 2)
            list_of_after_star_offsets.append(annot_row["endOffset"])

        list_of_before_star_offsets.sort(reverse=True)
        list_of_after_star_offsets.sort(reverse=True)

        x = [list_of_before_star_offsets, list_of_after_star_offsets]
        sentence = text_row["sentence"]

        sentence = insert_star(sentence)

        # using BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        marked_sentence = "[CLS] " + sentence + " [SEP]"

        tokenized_text = tokenizer.tokenize(marked_sentence)

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

           

            


        

        # for _, annot_row in tsv_df[doc_id].iterrows():
        #     annotType = annot_row["annotType"]

        #     if annotType == "Qualifier":
        #         continue

        #     for i, item in enumerate(sentence_tag_placeholders):
        #         if (
        #             annot_row["startOffset"] 
        #             <= item[1][0] 
        #             < annot_row["endOffset"]
        #         ) or (
        #             annot_row["startOffset"] 
        #             < item[1][1]
        #             <= annot_row["endOffset"]
        #         ):
        #             if annotType == "Quantity":
        #                 temp_list = []

    
    # for _, tsv_row in tsv_df.iterrows():
    #     if tsv_row['annotType'] == 'Qualifier':
    #         continue

    #     temp_list = []
    #     temp_list.append(tsv_row['docId'])

    #     temp_list.append(tsv_row['annotSet'])
    #     temp_list.append(tsv_row['startOffset'])
    #     temp_list.append(tsv_row['endOffset'])
    #     temp_list.append(tsv_row['text'])

    #     if tsv_row['annotType'] == 'Quantity':
    #         quantities_df.append(temp_list)

    #     elif tsv_row['annotType'] == 'MeasuredProperty' or tsv_row['annotType'] == 'MeasuredEntity':
    #         entities_df.append(temp_list)

    entities_df = pd.DataFrame(entities_df, columns=['docId','annotSet','startOffset','endOffset','text'])
    quantities_df = pd.DataFrame(quantities_df, columns=['docId','annotSet','startOffset','endOffset','text'])

    return entities_df, quantities_df

# returns contextual representations of entities and quantities
def get_train_data(train_tsv_dataframe):
    ent_df, quant_df = get_ent_quant_dfs(train_tsv_dataframe)

    ent_quant_cross = []    # columns are docID  ent_text  quant_text  label

    for i in range(len(ent_df)):
        for j in range(len(quant_df)):
            if ent_df['docId'][i] == quant_df['docId'][j]:
                temp_list = []
                temp_list.append(ent_df['docId'][i])
                temp_list.append(ent_df['text'][i])
                temp_list.append(quant_df['text'][j])
                if ent_df['annotSet'][i] == quant_df['annotSet'][j]:
                    temp_list.append(1)                               # 1 implies a yes instance
                else:
                    temp_list.append(0)
                
                ent_quant_cross.append(temp_list)
                
    ent_quant_cross_df = pd.DataFrame(ent_quant_cross, columns = ['docID', 'ent_text', 'quant_text', 'label'])

    # print(ent_quant_cross_df)
    # print(np.count_nonzero(np.array(ent_quant_cross_df['same_sentence']) == 0))
    # print(np.count_nonzero(np.array(ent_quant_cross_df['same_sentence']) == 1))


############# MAIN ##############
lang_model = spacy.load("en_core_web_sm")

train_raw_files = glob.glob("train/text/*.txt")
train_tsv_files = glob.glob("train/tsv/*.tsv")

each_file_df = []
for tsv_file in train_tsv_files:
    each_file_df.append(pd.read_csv(tsv_file, sep="\t", header=0))

train_tsv_dataframe = pd.concat(each_file_df)

train_text_dataframe = raw_text_to_df(train_raw_files)