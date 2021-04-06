import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Here is the * sentence * I want embeddings for."
marked_text = "[CLS] " + text + " [SEP]"
marked_text = "I am"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))

segments_ids = [1] * len(tokenized_text)

print (segments_ids)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,         # Whether the model returns all hidden-states.
                                  output_attentions=True
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)

    # attentions = outputs[3]
    print(len(outputs[3]), outputs[3][0].size())

print(outputs[3])