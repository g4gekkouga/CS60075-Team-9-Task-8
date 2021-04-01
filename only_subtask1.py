import os

# from tkinter.ttk import LabeledScale
import pandas as pd
import glob
from quantities.units.area import D
import spacy
from pprint import pprint
import quantities
from quantities import units
from functools import lru_cache

model = spacy.load("en_core_web_sm")


# typemap = {
#     "QUANT": "Quantity",
#     "ME": "MeasuredEntity",
#     "MP": "MeasuredProperty",
#     "QUAL": "Qualifier",
# }


def raw_text_to_df(train_raw_fies):
    """converting raw training files to dataframes"""
    # dict to be exported as dataframe
    documents_of_interest = {
        "document_name": [],
        "sentence": [],
        "potential_measurements": [],
    }
    # filling the dict
    for raw_file in train_raw_files:
        with open(raw_file, "r") as file:
            # print(file.name.split('/')[2])
            doc_name = file.name.split("/")[2]
            file_content = model(file.read())
            for sentence in file_content.sents:
                sentence_pos_tags = [word.tag_ for word in sentence]
                if "CD" in sentence_pos_tags:
                    documents_of_interest["document_name"].append(doc_name)
                    documents_of_interest["sentence"].append(sentence)
                    potential_measurements = []
                    for measurement in sentence.ents:
                        potential_measurements.append(
                            [
                                measurement.label_,
                                measurement.start,
                                measurement.end,
                            ]
                        )
                    documents_of_interest["potential_measurements"].append(
                        potential_measurements
                    )
        # break
    dataframe = pd.DataFrame(
        documents_of_interest,
        columns=["document_name", "sentence", "potential_measurements"],
    )
    # pprint(dataframe)
    return dataframe


# documents_of_interest = {
#     "document_name": [],
#     "sentence": [],
#     "potential_measurements": [],
# }

# ['docId' 'annotSet' 'annotType' 'startOffset' 'endOffset' 'annotId' 'text' 'other']


def get_train_labels(train_text_dataframe, train_tsv_dataframe):
    labels = []
    for _, text_row in train_text_dataframe.iterrows():
        print(text_row)
        sentence_tag_placeholders = []
        for word in text_row["sentence"]:
            sentence_tag_placeholders.append(
                ["O", word, word.idx, word.idx + len(word)]
            )
        # O means not a quantity QUANT means quantity
        document_name = text_row["document_name"]
        match_doc_id = (
            train_text_dataframe["document_name"]
            == train_tsv_dataframe["docId"]
        )
        print(match_doc_id)
        break
        # for _, tsv_row in train_tsv_dataframe.iterrows():

    pass


units_list = []

train_raw_files = glob.glob("train/text/*.txt")
train_tsv_files = glob.glob("train/tsv/*.tsv")
test_raw_files = glob.glob("test/text/*.txt")


def get_units(model):
    units_list = []
    # quantities.unitquantity.UnitQuantity
    for key, value in units.__dict__.items():
        if isinstance(value, quantities.unitquantity.UnitQuantity):
            i = 0
            if key.lower() in model.Defaults.stop_words:
                i = i + 1

            # if value.name.lower() in model.Defaults.stop_words :
            #     i = i + 1

        print(i)


# for key, val in u.__dict__.items():
#     if isinstance(val, type(u.l)):
#         if key not in units and key.lower() not in nlp.Defaults.stop_words:
#             units.append(key.lower())

#         if val.name not in units and val.name.lower() not in nlp.Defaults.stop_words:
#             units.append(val.name.lower())


def main():

    tag = {"QUANT": "Quantity"}
    # model = spacy.load('en_core_web_sm')

    # units_list = get_units(model)

    train_text_dataframe = raw_text_to_df(train_raw_files)
    train_text_dataframe.to_csv("./CSV/train_text_dataframe.csv")

    # convert training tsv files into dataframe for easier usage
    each_file_df = []
    for tsv_file in train_tsv_files:
        each_file_df.append(pd.read_csv(tsv_file, sep="\t", header=0))

    train_tsv_dataframe = pd.concat(each_file_df)
    # print(train_tsv_dataframe.columns.values)
    # quit()
    train_tsv_dataframe.to_csv("./CSV/train_tsv_dataframe.csv")
    # do the above for test set

    test_text_dataframe = raw_text_to_df(test_raw_files)
    test_text_dataframe.to_csv("./CSV/test_text_dataframe.csv")

    # X_train = @amshu
    Y_train = get_train_labels(train_text_dataframe, train_tsv_dataframe)
    pass


if __name__ == "__main__":
    main()