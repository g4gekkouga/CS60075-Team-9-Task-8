import os
from numpy import isin, mask_indices
import sklearn_crfsuite
import glob
import re
import pandas as pd
import spacy
import quantities
from pprint import pprint

nlp = spacy.load("en_core_web_sm")

# # the units to be extracted
# units = []

# training data
train_raw_files = glob.glob("train/text/*.txt")
train_tsv_files = glob.glob("train/tsv/*.tsv")
test_raw_files = glob.glob("test/text/*.txt")
# test_tsv_files = glob.glob('test/tsv/*.tsv')

typemap = {
    "QUANT": "Quantity",
    "ME": "MeasuredEntity",
    "MP": "MeasuredProperty",
    "QUAL": "Qualifier",
}


def convert_raw_to_df(train_raw_files):
    """Converts the training raw files into dataframes"""
    documents_of_interest = {
        "document_name": [],
        "sentence": [],
        "potential_measurements": [],
        "noun_phrases": [],
    }
    for raw_file in train_raw_files:
        with open(raw_file, "r") as file:
            doc_name = file.name.split("/")[2]
            file_content = nlp(file.read())
            for sentence in file_content.sents:
                sentence_pos_tags = [word.tag_ for word in sentence]
                if "CD" in sentence_pos_tags:
                    documents_of_interest["document_name"].append(doc_name)
                    documents_of_interest["sentence"].append(sentence)

                    potential_measurements = []
                    # print(sentence)
                    # pprint(sentence.ents)
                    for measurement in sentence.ents:
                        potential_measurements.append(
                            [
                                measurement.label_,
                                measurement.start,
                                measurement.end,
                            ]
                        )

                    noun_phrases = []
                    for chunk in sentence.noun_chunks:
                        noun_phrases.append(
                            [chunk.text, chunk.start, chunk.end]
                        )

                    documents_of_interest["potential_measurements"].append(
                        potential_measurements
                    )
                    documents_of_interest["noun_phrases"].append(noun_phrases)
        break
    dataframe = pd.DataFrame(
        documents_of_interest,
        columns=[
            "document_name",
            "sentence",
            "potential_measurements",
            "noun_phrases",
        ],
    )
    pprint(dataframe)
    return dataframe


def main():
    # convert training raw files to dataframes for easier usage
    train_text_dataframe = convert_raw_to_df(train_raw_files)
    train_text_dataframe.to_csv("./train_text_dataframe.csv")
    exit()
    # convert training tsv files into dataframe for easier usage
    individual_file_df = []
    for tsv_file in train_tsv_files:
        individual_file_df.append(pd.read_csv(tsv_file, sep="\t", header=0))
    train_tsv_dataframe = pd.concat(individual_file_df)
    train_tsv_dataframe.to_csv("./train_tsv_dataframe.csv")
    # pprint(train_tsv_dataframe)
    # exit()

    # do the above for test set
    test_text_dataframe = convert_raw_to_df(test_raw_files)
    test_text_dataframe.to_csv("./test_text_dataframe.csv")

    #xtrain
    #ytrain
    #xtest
    #ypred

    exit()


if __name__ == "__main__":
    main()
