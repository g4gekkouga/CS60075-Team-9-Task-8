import glob
import numpy as np
import pandas as pd
import spacy
import copy
import torch
from Task1 import raw_text_to_df
from transformers import BertTokenizer, BertModel

def get_train_data(text_df, tsv_df):
    train_data = []

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained('bert-base-uncased',
                                output_hidden_states = True, # Whether the model returns all hidden-states.
                                output_attentions= True
                                )

    model.eval()

    for _, text_row in text_df.iterrows():
        # list of words and their spans
        sentence_tag_placeholders = []
        for word in text_row["sentence"]:
            sentence_tag_placeholders.append(
                [str(word), (word.idx, (word.idx + len(word)))]
            )
        
        document_name = text_row['document_name']
        doc_id = tsv_df["docId"] == document_name

        # extract the entity and quantity spans of those present in the current sentence
        entity_start = []
        entity_end = []
        quantity_start = []
        quantity_end = []

        for _, annot_row in tsv_df[doc_id].iterrows():
            if (
                annot_row["startOffset"] 
                >= sentence_tag_placeholders[0][1][0] 
            ) and (
                annot_row["endOffset"]
                < sentence_tag_placeholders[-1][1][1]
            ):          # if the annotation is completely in the sentence
                
                if annot_row["annotType"] == "Quantity":
                    quantity_start.append([annot_row["annotSet"],annot_row["startOffset"]])
                    quantity_end.append([annot_row["annotType"], annot_row["endOffset"]])

                if annot_row["annotType"] == "MeasuredProperty" or annot_row["annotType"] == "MeasuredEntity":
                    entity_start.append([annot_row["annotSet"],annot_row["startOffset"]])
                    entity_end.append([annot_row["annotType"], annot_row["endOffset"]])

        # sort the entity and quantity lists according to their spans
        quantity_start = sorted(quantity_start, key=lambda v:v[1])
        quantity_end = sorted(quantity_end, key=lambda v:v[1])

        entity_start = sorted(entity_start, key=lambda v:v[1])
        entity_end = sorted(entity_end, key=lambda v:v[1])

        # remember token number for the *s
        curr_ent_start = 0
        curr_ent_end = 0
        curr_quant_start = 0
        curr_quant_end = 0

        tokenized_text = ['[CLS]']
        for word_info in sentence_tag_placeholders:
            tokens = tokenizer.tokenize(word_info[0])

            #check for matching entities and store their start token number
            if word_info[1][0] == entity_start[curr_ent_start][1]:
                token_num = len(tokenized_text)
                tokens = ['*'] + tokens
                entity_start[curr_ent_start].append(token_num)
                curr_ent_start += 1

            if word_info[1][1] == entity_end[curr_ent_end][1]:
                tokens = tokens + ['*']
                curr_ent_end += 1

            #check for matching quantities and store their start token number
            if word_info[1][0] == quantity_start[curr_quant_start][1]:
                token_num = len(tokenized_text)
                tokens = ['*'] + tokens
                quantity_start[curr_quant_start].append(token_num)
                curr_quant_start += 1

            if word_info[1][1] == quantity_end[curr_quant_end][1]:
                tokens = tokens + ['*']
                curr_quant_end += 1

            tokenized_text.extend(tokens)

        # make dataframes for entity and quantity token numbers
        entity_df = pd.DataFrame(entity_start, columns=['annotSet', 'startOffset', 'tokenNo'])
        quantity_df = pd.DataFrame(quantity_start, columns=['annotSet', 'startOffset', 'tokenNo'])
        
        # use BERT to get embeddings and attention
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensor)

            hidden_states = outputs[2]
            attentions = outputs[3]
            

        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs_sum = []

        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:],dim=0)

            token_vecs_sum.append(sum_vec)
        
        token_vecs_sum = torchs.stack(token_vecs_sum,dim=0)
        # token_vecs_sum is of dimension (#tokens,768)

        attentions = torch.stack(attentions, dim=0)
        attentions = torch.squeeze(attentions, dim=1)
        attentions = attentions.permute(2,3,0,1)

        # attentions dim is (#tokens, #tokens, #layers, #attention_head)


        # getting the cartesian product of entities and quantities
        for _, ent_row in entity_df.iterrows():
            for _, quant_row in quantity_df.iterrows():
                ent_emb = token_vecs_sum[ent_row["tokenNo"]]
                quant_emb = token_vecs_sum[quant_row["tokenNo"]]

                att1 = torch.flatten(attentions[ent_row["tokenNo"], quant_row["tokenNo"]])
                att2 = torch.flatten(attentions[quant_row["tokenNo"], ent_row["tokenNo"]])

                att = torch.cat((att1, att2))

                input_vec = torch.cat((ent_emb, quant_emb, att))

                if ent_row["annotSet"] == quant_row["annotSet"]:
                    train_data.append([input_vec, 1])
                else:
                    train_data.append([input_vec, 0])

    train_data = pd.DataFrame(train_data, columns=["X","label"])
    return train_data

############# MAIN ##############
lang_model = spacy.load("en_core_web_sm")

train_raw_files = glob.glob("train/text/*.txt")
train_tsv_files = glob.glob("train/tsv/*.tsv")

each_file_df = []
for tsv_file in train_tsv_files:
    each_file_df.append(pd.read_csv(tsv_file, sep="\t", header=0))

train_tsv_dataframe = pd.concat(each_file_df)

train_text_dataframe = raw_text_to_df(train_raw_files)

train_data = get_train_data(train_text_dataframe[:8], train_tsv_dataframe[:9])
train_data.to_csv("./trial_train.csv", sep='\t')