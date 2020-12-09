#### Run: python sprinplanning.py project sprint_size
#####################################################
import os
import sys
import numpy
import pandas as pd
import cPickle
import gzip
from keras.models import model_from_json
from keras.models import Model
from keras.layers import *
from keras.callbacks import *
import prepare_backlog
from create_model import *

#### System arguments ####
project = sys.argv[1]           #project dataset
dataPre = sys.argv[2]           #projet owner / pretrain data
sprint_size = sys.argv[3]       # Estimated team velocity

#### Loading model ####
pretrain = 'fixed_lm'
hid_dim = 10
path = project + '_lstm_highway_dim' + str(hid_dim) + '_reginphid_pre' + pretrain + '_poolmean'
model_path = 'Deep-SE/classification/models/' + path + '.json'
param_path = 'Deep-SE/classification/bestModels/' + path + '.hdf5'
fModel = open(model_path)                                               #load model architecture from json file in /models
model = model_from_json(fModel.read(), custom_objects =  {'PoolingSeq': PoolingSeq})    
model.summary()
model.load_weights(param_path)                                           #loading parameters from hdf5 files in /bestModels
print('Model loaded')

#### Loading dataset ####
f = gzip.open( 'Deep-SE/data/' + sys.argv[1] + '.pkl.gz', 'rb')
train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = cPickle.load(f) 
print train_y
title = train_t + valid_t + test_t
description = train_d + valid_d + test_d
#modifying titles and desscriptions by limiting their vocab and length, and creating masks for them
#types: numpy arrays
title, title_mask, descr, descr_mask = prepare_backlog.prepare_data(title, description)

# Load embedding weights if pretrained model is used
if pretrain == 'x':
    emb_weight = None
elif 'w2v' in pretrain:
    pretrain_path = 'word2vec_' + project + '_dim' + str(hid_dim) + '.pkl'
    emb_weight = prepare_backlog.load_w2v_weight(pretrain_path)
else:
    if 'lm' in pretrain: lm = 'lm'
    else: lm = ''
    pretrain_path = 'lstm2v_' + dataPre + '_dim' + str(hid_dim)
    emb_weight = prepare_backlog.load_weight(pretrain_path)

if 'fixed' in pretrain:
    title, descr = prepare_backlog.to_features([title, descr], emb_weight)

##########################################################################
################### Estimating story-points ##############################

print('Estimation:')
x = [title, title_mask, descr, descr_mask]               #a list of numpy arrays  
sp_pred = model.predict(x, batch_size=x[0].shape[0])
print(sp_pred)

# Construct product backlog containing estimated story-points
df=pd.read_csv('Deep-SE/data/' + project + '.csv')
df['storypoint'] = sp_pred[:, 0]
print('Dataframe with estimated story-points:')
print(df)
df.to_csv('estimation_' + project + '.csv')

#########################################################################
######################## Sprint planning ################################

rand_prios = np.random.randint( 0, 11, size = (len(df.index), 1) )
df['priority'] = rand_prios
sp_per_prio = df.storypoint / df.priority
df['effort_per_prio'] = sp_per_prio
# Suggest refinement where the issue is too large compared to team velocity
df['Refinement'] = "-"
thresh = 0.25 * float(sprint_size)
df.loc[df['storypoint'] > thresh, 'Refinement'] = "Refine"   

# Sorting
efficient_backlog = df.sort_values('effort_per_prio', axis = 0, ascending = True, inplace = False) 
prio_backlog = df.sort_values('priority', axis = 0, ascending = False, inplace = False)

# Creating efficient sprint backlog
sp_sum = 0
ind = 0

while int(sp_sum) < int(sprint_size):
    sp_sum+= efficient_backlog.iloc[ind,3]  #summing storypoints
    ind+= 1
efficient_sprint = efficient_backlog[0:ind-1]
print(efficient_sprint)
efficient_sprint.to_csv('sprint_' + project + '_size' + str(sprint_size) + '.csv')

# Creating prioritized sprint backlog
sp_sum=0
ind=0

while int(sp_sum) < int(sprint_size):
    sp_sum+= prio_backlog.iloc[ind,3]      #summing storypoints    
    ind+= 1
prio_sprint = prio_backlog[0:ind-1]
print(prio_sprint)