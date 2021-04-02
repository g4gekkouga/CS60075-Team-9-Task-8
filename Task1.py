#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8

# In[132]:


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

# from sklearn_crfsuite import metrics

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[133]:


tags = {"QUANT": "Quantity"}
lang_model = spacy.load("en_core_web_sm")


# In[197]:


units_list = []

train_raw_files = glob.glob("train/text/*.txt")
train_tsv_files = glob.glob("train/tsv/*.tsv")
# test_raw_files = glob.glob("trial/txt/*.txt")
test_raw_files = glob.glob("test/text/*.txt")
# test_tsv_files = glob.glob("trial/tsv/*.tsv")


# In[181]:


def raw_text_to_df(raw_files):
    """converting raw training files to dataframes"""
    # dict to be exported as dataframe
    documents_of_interest = {
        "document_name": [],
        "sentence": [],
        "entities": [],
        "np": [],
    }
    # filling the dict
    for raw_file in raw_files:
        with open(raw_file, "r") as file:
            # print(file.name.split('/')[2])
            doc_name = file.name.split("/")[2]
            doc_name = doc_name.split(".")[0]
            file_content = lang_model(file.read())
            for sentence in file_content.sents:
                sentence_pos_tags = [word.tag_ for word in sentence]
                if "CD" in sentence_pos_tags:
                    documents_of_interest["document_name"].append(doc_name)
                    documents_of_interest["sentence"].append(sentence)
                    entities = []
                    for measurement in sentence.ents:
                        entities.append(
                            (
                                measurement.label_,
                                (measurement.start, measurement.end),
                            )
                        )
                    documents_of_interest["entities"].append(entities)

                    noun_phrases = []

                    for chunk in sentence.noun_chunks:
                        noun_phrases.append(
                            (chunk.text, (chunk.start, chunk.end))
                        )
                    documents_of_interest["np"].append(noun_phrases)

        # break
    dataframe = pd.DataFrame(
        documents_of_interest,
        columns=["document_name", "sentence", "entities", "np"],
    )
    # pprint(dataframe)
    return dataframe


# In[188]:


def get_text_labels(text_dataframe, tsv_dataframe):
    labels = []
    for _, text_row in text_dataframe.iterrows():
        sentence_tag_placeholders = []
        for word in text_row["sentence"]:
            sentence_tag_placeholders.append(
                ["O", (word.idx, (word.idx + len(word)))]
            )
        # O means not a quantity QUANT means quantity
        document_name = text_row["document_name"]
        doc_id = tsv_dataframe["docId"] == document_name
        for _, annot_row in tsv_dataframe[doc_id].iterrows():
            annotType = annot_row["annotType"]
            if annotType == "Qualifier":
                continue

            if annotType == "Quantity":
                annotType = "QUANT"
            else:
                annotType = "O"

            for i, item in enumerate(sentence_tag_placeholders):
                if item[0] != "O":
                    continue

                if (
                    annot_row["startOffset"]
                    <= item[1][0]
                    < annot_row["endOffset"]
                ) or (
                    annot_row["startOffset"]
                    < item[1][1]
                    <= annot_row["endOffset"]
                ):
                    if annotType != "O":
                        sentence_tag_placeholders[i][0] = annotType

        labels.append([label for label, _ in sentence_tag_placeholders])

    return labels


# In[137]:


def get_units():
    units_list = []  # Add possible unit symbols
    for key, value in units.__dict__.items():
        if isinstance(value, UQ):
            if key not in units_list:
                units_list.append(key.lower())
            if value.name not in units_list:
                units_list.append(value.name.lower())

    return units_list


# In[138]:


def is_unit(token):
    return token.lower_ in units_list or token.lemma_ in units_list


# In[139]:


def features_word(word, entities, nouns, length, pos):

    features = {
        "bias": 1.0,
        "lower": word.lower_,
        "lemma": word.lemma_,
        "upper": word.is_upper,
        "title": word.is_title,
        "digit": word.is_digit,
        "numlike": word.like_num,
        "unit": is_unit(word),
        "postag": word.tag_,
        "dep": word.dep_,
    }

    for entity in entities:
        if entity[1][0] <= word.i < entity[1][1]:
            features["entity"] = entity[0]
            break

    for noun in nouns:
        if noun[1][0] <= word.i < noun[1][1]:
            features["np"] = list(noun[0])
            break

    if pos >= 1:
        new_word = word.nbor(-1)
        features.update(
            {
                "-1:lower": new_word.lower_,
                "-1:lemma": new_word.lemma_,
                "-1:upper": new_word.is_upper,
                "-1:title": new_word.is_title,
                "-1:digit": new_word.is_digit,
                "-1:numlike": new_word.like_num,
                "-1:unit": is_unit(new_word),
                "-1:postag": new_word.tag_,
                "-1:dep": new_word.dep_,
            }
        )

    if pos <= length - 2:
        new_word = word.nbor(1)
        features.update(
            {
                "+1:lower": new_word.lower_,
                "+1:lemma": new_word.lemma_,
                "+:upper": new_word.is_upper,
                "+:title": new_word.is_title,
                "+:digit": new_word.is_digit,
                "+:numlike": new_word.like_num,
                "+:unit": is_unit(new_word),
                "+:postag": new_word.tag_,
                "+:dep": new_word.dep_,
            }
        )

    return features


# In[140]:


def features_sentence(sentence, entities, nouns):
    sentence_features = []
    for i in range(0, len(sentence)):
        word_features = features_word(
            sentence[i], entities, nouns, len(sentence), i
        )
        sentence_features.append(word_features)
    return sentence_features


# In[198]:

def write_predictions_to_tsv(test_text_dataframe, y_pred):
    pass


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

# now operating on the actual test data instead of the trial data hence commenting out.
# each_file_test = []
# for tsv_file in test_tsv_files:
#     each_file_test.append(pd.read_csv(tsv_file, sep="\t", header=0))

# test_tsv_dataframe = pd.concat(each_file_test)
# test_tsv_dataframe.to_csv("./CSV/test_tsv_dataframe.csv")


# In[199]:


X_train = []
for _, row in train_text_dataframe.iterrows():
    features = features_sentence(row["sentence"], row["entities"], row["np"])
    X_train.append(features)


# In[200]:


y_train = get_text_labels(train_text_dataframe, train_tsv_dataframe)


# In[201]:


X_test = []
for _, row in test_text_dataframe.iterrows():
    features = features_sentence(row["sentence"], row["entities"], row["np"])
    X_test.append(features)


# In[202]:


# y_test = get_text_labels(test_text_dataframe, test_tsv_dataframe)


# In[204]:


# crf = sklearn_crfsuite.CRF(
#     algorithm="lbfgs",
#     c1=0.1,
#     c2=0.1,
#     max_iterations=100,
#     all_possible_transitions=True,
# )
# crf.fit(X_train, y_train)


# In[162]:


# labels = list(crf.classes_)
# labels.remove("O")


# In[163]:


# y_pred = crf.predict(X_test)

# sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

# print(
#     metrics.flat_classification_report(
#         y_test, y_pred, labels=sorted_labels, digits=3
#     )
# )


# In[48]:


# Training the model

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs", max_iterations=100, all_possible_transitions=True
)

params_space = {
    "c1": scipy.stats.expon(scale=0.5),
    "c2": scipy.stats.expon(scale=0.05),
}

labels = ["QUANT"]
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))


f1_scorer = make_scorer(
    metrics.flat_f1_score, average="weighted", labels=labels
)

rs = RandomizedSearchCV(
    crf, params_space, cv=3, verbose=1, n_jobs=-1, n_iter=50, scoring=f1_scorer
)

rs.fit(X_train, y_train)


# In[52]:


crf = rs.best_estimator_
y_pred = crf.predict(X_test)

# documents_of_interest = {
#     "document_name": [],
#     "sentence": [],
#     "entities": [],
#     "np": [],
# }

# for i, row in test_text_dataframe.iterrows():
#     print(row["document_name"])
#     print(row["sentence"])
#     print(y_pred[i])

write_predictions_to_tsv(test_text_dataframe, y_pred)

# print(
#     metrics.flat_classification_report(
#         y_test, y_pred, labels=sorted_labels, digits=3
#     )
# )
