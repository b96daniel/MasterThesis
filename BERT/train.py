### run: python train.py project
### Fine tunes BERT with the selected dataset (argv[1] = project)
import os
import sys
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import datetime
import random

########################### Local functions ####################################
# Function to calculate the accuracy of our predictions vs labels
def mre(preds, labels):
    mre = abs((labels - preds) / labels)
    #mre = np.divide(np.abs(np.subtract(labels, preds)), labels)
    return mre                              

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))              
##################################################################################  
####################### Loading dataset and pretrained model #####################
project = sys.argv[1]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
df = pd.read_csv("../dataset/" + project + ".csv")               
df.fillna(" ", inplace = True)                          #replace NULL values with space
titles = df.title.values
descr = df.description.values
labels = df.storypoint.values
texts = titles + ' ' + descr                                

input_ids = []

# For every sentence...
for sent in texts:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                        # This function also supports truncation and conversion
                        # to pytorch tensors, but we need to do padding, so we
                        # can't use these features.
                        #max_length = 128,          # Truncate all sentences.
                        #return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.
    input_ids.append(encoded_sent)

# Padding:
MAX_LEN = 80
print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
# Pad our input tokens with value 0.
# "post" indicates that we want to pad and truncate at the end of the sequence,
# as opposed to the beginning.
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="int64", 
                          value=0, truncating="post", padding="post")
print('\nDone.')

# Create attention masks
attention_masks = []
# For each sentence...
for sent in input_ids:    
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]    
    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

# Use train_test_split to split our data into train and validation sets (90-10%)
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            test_size=0.1, shuffle = False)     
# Do the same for the masks.
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                             test_size=0.1, shuffle = False)

# Convert all inputs and labels into torch tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels).to(torch.float)                   #labels need to be float for regression
validation_labels = torch.tensor(validation_labels).to(torch.float)
train_labels =  train_labels / 100                  
validation_labels = validation_labels / 100

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)                                      

# The DataLoader needs to know our batch size for training
# For fine-tuning BERT on a specific task, recommended batch size is 16 or 32.
batch_size = 16

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)                               #instead RandomSampler
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 1, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
model.cuda()
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# Probably 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),  lr = 2e-5,  eps = 1e-8)

# Number of training epochs (authors recommend between 2 and 4)
epochs = 10
# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


#####################################################################################################
####################################### Training Loop ###############################################

device = torch.device("cuda")
# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []
MMREs = []
minMMRE = 100

print('Training...')
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
   
    # Perform one full pass over the training set.
    print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. The call to `train` just changes the *mode*,
    # it doesn't *perform* the training. `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        # As we unpack the batch, we'll also copy each tensor to the GPU using the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels         
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()
        
        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step() 
        ###################### End of batch ############################         
    
    # Store the loss value for plotting the learning curve.
    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)

    print("\n  Average training loss: {0:.4f}".format(total_loss / len(train_dataloader) ))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    # ========================================
    #               Validation
    # ========================================
    print("\nRunning Validation...")
    #t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    eval_mmre = 0
    nb_eval_steps = 0
    predictions , true_labels = [], []

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]
        
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        b_label_ids = b_labels.to('cpu').numpy()

        # Calculate MMRE for this batch of test sentences.
        b_preds = [item[0] for item in logits]      
        #MMRE:
        print("MMRE:")
        tmp = 0
        for i, item in enumerate(b_preds):
          tmp += mre(item,b_label_ids[i])        
        print(tmp / len(b_label_ids))
        eval_mmre += (tmp / len(b_label_ids))
        
        # Track the number of batches
        nb_eval_steps += 1
        # Store predictions and true labels
        predictions.append(b_preds)
        true_labels.append(b_label_ids)
        ################# End of batch ######################

    # Report the final MMRE for this validation epoch.
    MMRE = eval_mmre / nb_eval_steps
    print(" MMRE: {0:.3f}".format(MMRE))
    MMREs.append(MMRE)

    # Save best validation log of predictions, labels 
    if MMRE < minMMRE:
      minMMRE = MMRE
      flat_predictions = [item for sublist in predictions for item in sublist]
      flat_true_labels = [item for sublist in true_labels for item in sublist]
      val_dict = {"Predictions": flat_predictions, "True labels": flat_true_labels}
      val_df = pd.DataFrame(val_dict)
      val_df.to_csv("log/" + project + "_validation.csv")
    ###################### End of epoch #######################

result_dict = {"Test MMRE": MMREs, "Avg. Training Loss": loss_values}
log_df = pd.DataFrame(result_dict)
log_df.to_csv("log/" + project + "_log.csv")
print("\nTraining complete!")
################################### End of training loop ########################################
#################################################################################################

######################################## Save model #############################################
# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save/' + project + '/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))