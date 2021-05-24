### run: python eval.py project
### Evaluates fine-tuned BERT model of 'project' (argv[1])
import os
import sys
import pandas as pd
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

project = sys.argv[1]
output_dir = './model_save/' + project + '/'
# Load a trained model and vocabulary that you have fine-tuned
model = model_class.from_pretrained(output_dir)
tokenizer = tokenizer_class.from_pretrained(output_dir)

# Copy the model to the GPU.
device = torch.device("cuda")
model.to(device)

# Load the dataset into a pandas dataframe.
df = pd.read_csv("../dataset/" + project + ".csv")
df.fillna(" ", inplace = True)                                  #replace NULL values with space

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
titles = df.title.values
descr = df.description.values
labels = df.storypoint.values
labels = labels / 100
sentences = titles + ' ' + descr

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []

# For every sentence...
for sent in sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                   )
    
    input_ids.append(encoded_sent)

MAX_LEN = 500
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                          dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask) 

# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set
def mre(preds, labels):
    mre = abs((labels - preds) / labels)
    #mre = numpy.divide(numpy.abs(numpy.subtract(labels, preds)), labels)
    return mre 

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

flat_predictions = [item[0] for sublist in predictions for item in sublist]
flat_true_labels = [item for sublist in true_labels for item in sublist]
result_dict = {"Predictions": flat_predictions, "True labels": flat_true_labels}
results_df = pd.DataFrame(result_dict)
print(results_df)

#MMRE:
print("MMRE:")
tmp = 0
for i, item in enumerate(flat_predictions):
  tmp += mre(item,flat_true_labels[i])
print(tmp / len(flat_true_labels))

results_df.to_csv('results_' + project + '.csv') 
print('    DONE.')