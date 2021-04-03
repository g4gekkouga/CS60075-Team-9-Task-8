#!/usr/bin/env python
# coding: utf-8

# In[102]:


import glob
import re
from collections import Counter

from quantities.units.area import D
import pandas as pd
import sklearn_crfsuite
import spacy
import quantities
from quantities import units
from quantities.unitquantity import UnitQuantity as UQ
from sklearn_crfsuite import metrics
from pprint import pprint

import scipy
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# In[103]:


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[104]:


tags = {'QUANT' : 'Quantity', 'ME' : 'MeasuredEntity', 'MP' : 'MeasuredProperty'}
lang_model = spacy.load('en_core_web_sm')


# In[105]:


units_list = []

train_raw_files = glob.glob("train/text/*.txt")
train_tsv_files = glob.glob("train/tsv/*.tsv")
test_raw_files = glob.glob("trial/txt/*.txt")
test_tsv_files = glob.glob("trial/tsv/*.tsv")
eval_raw_files = glob.glob("eval/text/*.txt")


# In[106]:


def raw_text_to_df(raw_files):
    """converting raw training files to dataframes"""
    # dict to be exported as dataframe
    documents_of_interest = {
        'document_name': [],
        'sentence': [],
        'entities': [],
        'np': []
    }
    # filling the dict
    for raw_file in raw_files:
        with open(raw_file, "r") as file:
            doc_name = file.name.split("/")[2]
            doc_name = doc_name.split('.')[0]
            file_content = lang_model(file.read())
            for sentence in file_content.sents:
                sentence_pos_tags = [word.tag_ for word in sentence]
                documents_of_interest['document_name'].append(doc_name)
                documents_of_interest['sentence'].append(sentence)
                entities = []
                for measurement in sentence.ents:
                    entities.append((measurement.label_,(measurement.start, measurement.end)))
                documents_of_interest['entities'].append(
                    entities
                )

                noun_phrases = []

                for chunk in sentence.noun_chunks:
                    noun_phrases.append((chunk.text, (chunk.start, chunk.end)))
                documents_of_interest['np'].append(noun_phrases)
                    
        # break
    dataframe = pd.DataFrame(
        documents_of_interest,
        columns=['document_name', 'sentence', 'entities', 'np'],
    )
    # pprint(dataframe)
    return dataframe


# In[107]:


def get_text_labels(text_dataframe, tsv_dataframe):
    labels = []
    for _, text_row in text_dataframe.iterrows():
        sentence_tag_placeholders = []
        for word in text_row['sentence']:
            sentence_tag_placeholders.append(
                ['O', (word.idx, (word.idx + len(word)))]
            )
        # O means not a quantity QUANT means quantity
        document_name = text_row['document_name']
#         print(document_name)
        doc_id = tsv_dataframe['docId'] == document_name
        for _, annot_row in tsv_dataframe[doc_id].iterrows():
            annotType = annot_row['annotType']
            if annotType == 'Qualifier':
                continue

            for key, value in tags.items():
                if annotType == value:
                    annotType = key
                    break
            
            for i, item in enumerate(sentence_tag_placeholders):
                if item[0] != 'O':
                    continue
                
                if (annot_row['startOffset'] <= item[1][0] < annot_row['endOffset']) or (annot_row['startOffset'] < item[1][1] <= annot_row['endOffset']):
                    sentence_tag_placeholders[i][0] = annotType
    
        labels.append([label for label, _ in sentence_tag_placeholders])
                
    return labels


# In[108]:


def get_units() :
    units_list = ['%'] # Add possible unit symbols
    for key, value in units.__dict__.items():
        if isinstance(value, UQ):
            if key not in units_list :
                units_list.append(key.lower())
            if value.name not in units_list :
                units_list.append(value.name.lower())
        
    return units_list

def is_unit(token):
    return token.lower_ in units_list or token.lemma_ in units_list


# In[109]:


def features_word(word, entities, nouns, length, pos):
    
    features = {
        'bias': 1.0,
        'lower': word.lower_,
        'lemma': word.lemma_,
        'upper': word.is_upper,
        'title': word.is_title,
        'digit': word.is_digit,
        'numlike': word.like_num,
        'unit': is_unit(word),
        'postag': word.tag_,
        'dep': word.dep_
    }
    
    for entity in entities:
        if entity[1][0] <= word.i < entity[1][1]:
            features['entity'] = entity[0]
            break
    
    for noun in nouns:
        if noun[1][0] <= word.i < noun[1][1]:
            features['np'] = list(noun[0])
            break
    
    if pos >= 1 :
        new_word = word.nbor(-1)
        features.update({
            '-1:lower': new_word.lower_,
            '-1:lemma': new_word.lemma_,
            '-1:upper': new_word.is_upper,
            '-1:title': new_word.is_title,
            '-1:digit': new_word.is_digit,
            '-1:numlike': new_word.like_num,
            '-1:unit': is_unit(new_word),
            '-1:postag': new_word.tag_,
            '-1:dep': new_word.dep_
        })
        
        
    if pos <= length-2 :
        new_word = word.nbor(1)
        features.update({
            '+1:lower': new_word.lower_,
            '+1:lemma': new_word.lemma_,
            '+1:upper': new_word.is_upper,
            '+1:title': new_word.is_title,
            '+1:digit': new_word.is_digit,
            '+1:numlike': new_word.like_num,
            '+1:unit': is_unit(new_word),
            '+1:postag': new_word.tag_,
            '+1:dep': new_word.dep_
        })
    
    return features


# In[110]:


def features_sentence(sentence, entities, nouns):
    sentence_features = []
    for i in range(0, len(sentence)) :
        word_features = features_word(sentence[i], entities, nouns, len(sentence), i)
        sentence_features.append(word_features)
    return sentence_features


# In[111]:


def write_predictions_to_tsv(text_dataframe, y_pred, dirname):
    pass


# In[112]:


def find_closest(quantspan, start, end):
    ind = -1
    min_dist = 100000
    for i in range(len(quantspan)):
        dist = 100000
        if (start > quantspan[i][1]) :
            dist = start - quantspan[i][1]
        else :
            dist = quantspan[i][0] - end
        if dist <= min_dist :
            min_dist = dist
            ind = i
    return ind


# In[113]:


def merge_tags(output_list, quant_list, mp_list, me_list, quantspan, mpspan, mespan):
    
    if (len(quant_list) <= 0) :
        return output_list
    
    me_ind = [-1] * len(quant_list)
    mp_ind = [-1] * len(quant_list)
    
    for i in range(len(mpspan)):
        closest = find_closest(quantspan, mpspan[i][0], mpspan[i][1])
        mp_ind[closest] = i
    
    for i in range(len(mespan)):
        closest = find_closest(quantspan, mespan[i][0], mespan[i][1])
#         print(len(quant_list), '--', closest)
        me_ind[closest] = i
    
    for i in range(len(quant_list)):
#         quant_list[i][1] = annot_set_index
#         quant_list[i][5] = annot_index
        output_list.append(quant_list[i])
        if mp_ind[i] != -1:
            mp_list[mp_ind[i]][1] = quant_list[i][1]
#             mp_list[mp_ind[i]][5] = annot_index
            output_list.append(mp_list[mp_ind[i]])
        if me_ind[i] != -1:
            me_list[me_ind[i]][1] = quant_list[i][1]
#             me_list[me_ind[i]][5] = annot_index
            output_list.append(me_list[me_ind[i]])
#         annot_set_index += 1
    return output_list


# In[114]:


def write_predictions_to_tsv(text_dataframe, y_pred, dirname):
    tsv_columns = [
        "docId",
        "annotSet",
        "annotType",
        "startOffset",
        "endOffset",
        "annotId",
        "text",
        "other"
    ]
    # save the results in appropriate format in a new dataframe
    row_id = 0
    output_list = []
    prev_file = ""
    annot_set_index = 1
    annot_index = 1
    for i, text_row in text_dataframe.iterrows():
        quant_list = []
        mp_list = []
        me_list = []
        quantspan = []
        mpspan = []
        mespan = []
        file_name = text_row['document_name']
        if i > 0 and file_name != prev_file:
            result_df = pd.DataFrame(output_list, columns=tsv_columns)
            result_df.to_csv(dirname + prev_file + '.tsv', sep="\t", index=False)
            output_list = []
            annot_set_index = 1
            annot_index = 1

        pred_pos = y_pred[row_id]
        row_id += 1
        sentence = text_row['sentence']
        
        word_ind = 0
        while word_ind < len(pred_pos) :
            if pred_pos[word_ind] != 'QUANT':
                word_ind += 1
                continue
            
            start_ind = word_ind
            while word_ind < len(pred_pos) and pred_pos[word_ind] == 'QUANT':
                word_ind += 1
            end_ind = word_ind - 1
            
            quant_text = sentence.doc[sentence[start_ind].i : sentence[end_ind].i + 1]
            quant_list.append([file_name, annot_set_index, tags['QUANT'], sentence[start_ind].idx, sentence[end_ind].idx + len(sentence[end_ind]), annot_index, quant_text, ""])
            quantspan.append([sentence[start_ind].i, sentence[end_ind].i])
            annot_set_index += 1
            annot_index += 1
            
        word_ind = 0
        while word_ind < len(pred_pos) :
            if pred_pos[word_ind] != 'MP':
                word_ind += 1
                continue
            
            start_ind = word_ind
            while word_ind < len(pred_pos) and pred_pos[word_ind] == 'MP':
                word_ind += 1
            end_ind = word_ind - 1
            
            mp_text = sentence.doc[sentence[start_ind].i : sentence[end_ind].i + 1]
            mp_list.append([file_name, 0, tags['MP'], sentence[start_ind].idx, sentence[end_ind].idx + len(sentence[end_ind]), annot_index, mp_text, ""])
            mpspan.append([sentence[start_ind].i, sentence[end_ind].i])
            annot_index += 1
            
        word_ind = 0
        while word_ind < len(pred_pos) :
            if pred_pos[word_ind] != 'ME':
                word_ind += 1
                continue
            
            start_ind = word_ind
            while word_ind < len(pred_pos) and pred_pos[word_ind] == 'ME':
                word_ind += 1
            end_ind = word_ind - 1
            
            me_text = sentence.doc[sentence[start_ind].i : sentence[end_ind].i + 1]
            me_list.append([file_name, 0, tags['ME'], sentence[start_ind].idx, sentence[end_ind].idx + len(sentence[end_ind]), annot_index, me_text, ""])
            mespan.append([sentence[start_ind].i, sentence[end_ind].i])
            annot_index += 1
        
        output_list = merge_tags(output_list, quant_list, mp_list, me_list, quantspan, mpspan, mespan)
#         print("Here", annot_set_index)
        prev_file = file_name
        
    result_df = pd.DataFrame(output_list, columns=tsv_columns)
    result_df.to_csv(dirname + prev_file + '.tsv', sep="\t", index=False)
    return


# In[115]:


# write_predictions_to_tsv(train_text_dataframe, y_train, 'sample_tsv/')


# In[122]:


units_list = get_units()

train_text_dataframe = raw_text_to_df(train_raw_files)
train_text_dataframe.to_csv("./CSV/train_text_dataframe.csv")

each_file_df = []
for tsv_file in train_tsv_files:
    each_file_df.append(pd.read_csv(tsv_file, sep="\t", header=0))

train_tsv_dataframe = pd.concat(each_file_df)
train_tsv_dataframe.to_csv("./CSV/train_tsv_dataframe.csv")

test_text_dataframe = raw_text_to_df(test_raw_files)
test_text_dataframe.to_csv("./CSV/test_text_dataframe.csv")

each_file_test = []
for tsv_file in test_tsv_files:
    each_file_test.append(pd.read_csv(tsv_file, sep="\t", header=0))

test_tsv_dataframe = pd.concat(each_file_test)
test_tsv_dataframe.to_csv("./CSV/test_tsv_dataframe.csv")


# In[123]:


X_train = []
for _, row in train_text_dataframe.iterrows() :
    features = features_sentence(row['sentence'], row['entities'], row['np'])
    X_train.append(features)
    
y_train = get_text_labels(train_text_dataframe, train_tsv_dataframe)


# In[124]:


X_test = []
for _, row in test_text_dataframe.iterrows() :
    features = features_sentence(row['sentence'], row['entities'], row['np'])
    X_test.append(features)
    
y_test = get_text_labels(test_text_dataframe, test_tsv_dataframe)


# In[125]:


X_train = X_train + X_test
y_train = y_train + y_test


# In[126]:


eval_text_dataframe = raw_text_to_df(eval_raw_files)
X_eval = []
for _, row in eval_text_dataframe.iterrows() :
    features = features_sentence(row['sentence'], row['entities'], row['np'])
    X_eval.append(features)


# In[127]:


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)


# In[128]:


labels = list(crf.classes_)
labels.remove('O')


# In[129]:


y_pred = crf.predict(X_eval)

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)


# In[130]:


write_predictions_to_tsv(eval_text_dataframe, y_pred, 'results_task3/')


# In[131]:


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)

params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.5), 
}

f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted', labels=labels)

rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=50,
                        scoring=f1_scorer)

rs.fit(X_train, y_train)


# In[ ]:


crf = rs.best_estimator_
y_pred = crf.predict(X_eval)
# print(metrics.flat_classification_report(
#     y_test, y_pred, labels=sorted_labels, digits=3
# ))


# In[ ]:


write_predictions_to_tsv(eval_text_dataframe, y_pred, 'results_task3_opt/')


# In[ ]:




