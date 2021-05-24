import cPickle
import numpy
import gzip
from keras.models import model_from_json
from NCE import *

def arg_passing(argv):
    # -data: dataset
    # -saving: log & model saving file
    # -dim: dimension of embedding
    # -reg: dropout: inp or hid or both

    i = 1
    arg_dict = {'-data': 'usergrid',
                '-dataPre': 'apache',
                '-saving': 'apache',
                '-seed': 1234,
                '-dim': 10,
                '-reg': '', # '', 'inp', 'hid' or 'inp_hid'
                '-seqM': 'lstm', # 'lstm', 'gru', 'rnn'
                '-nnetM': 'highway', #'dense', 'highway', 'resnet' - 'resnet' is not available now
                '-vocab': 500, # should be small
                '-pool': 'mean',
                '-ord': 0, # 0: categorical classification, 1: ordinal classification
                '-pretrain': 'x',
                '-len': 100
                }

    while i < len(argv) - 1:
        arg_dict[argv[i]] = argv[i+1]
        i += 2

    arg_dict['-dim'] = int(arg_dict['-dim'])
    arg_dict['-vocab'] = int(arg_dict['-vocab'])
    arg_dict['-seed'] = int(arg_dict['-seed'])
    arg_dict['-ord'] = int(arg_dict['-ord'])
    arg_dict['-len'] = int(arg_dict['-len'])
    return arg_dict


def load(path):
    f = gzip.open(path, 'rb')
    train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y = cPickle.load(f)    
    return train_t, train_d, train_y, valid_t, valid_d, valid_y, test_t, test_d, test_y

def prepare_data(title, descr, vocab_size=1000, max_len=100):
    def create_mask(seqs):
        new_seqs = []
        #limiting number of words in titles or descriptions?
        for idx, s in enumerate(seqs):
            new_s = [w for w in s if w < vocab_size]
            if len(new_s) == 0: new_s = [0]
            new_seqs.append(new_s)

        seqs = new_seqs

        lengths = [min(max_len, len(s)) for s in seqs]                  #lengths of text in titles or descr, limiting it to max_len
        maxlen = max_len
        n_samples = len(lengths)                                        #number of titles or descriptions?

        x = numpy.zeros((n_samples, maxlen)).astype('int64')            #zero array of len(lengths) * max_len
        mask = numpy.zeros((n_samples, maxlen)).astype('float32')

        for i, s in enumerate(seqs):
            l = lengths[i]                                              #length of text in ith title or descr
            mask[i, :l] = 1                                             #creating array of ones: ith row with length(i) number of ones
            x[i, :l] = s[:l]
            x[i, :l] += 1

        return x, mask                      

    #returning modified title, descr and some kind of masks for them? 
    title, title_mask = create_mask(title)
    descr, descr_mask = create_mask(descr)                              

    return title, title_mask, descr, descr_mask                         #numpy arrays

def to_features(list_seqs, emb_weight):
    vocab, dim = emb_weight.shape
    weight = numpy.zeros((vocab + 1, dim)).astype('float32')
    weight[1:] = emb_weight

    list_feats = []
    for seqs in list_seqs:
        n_samples, seq_len = seqs.shape
        feat = weight[seqs.flatten()].reshape([n_samples, seq_len, dim])
        list_feats.append(feat)
    return list_feats

def load_weight(path):
    model_path = 'NCE/models/' + path + '.json'
    param_path = 'NCE/bestModels/' + path + '.hdf5'

    custom = {'NCEContext': NCEContext, 'NCE': NCE, 'NCE_seq': NCE_seq}
    fModel = open(model_path)
    model = model_from_json(fModel.read(), custom_objects=custom)
    model.load_weights(param_path)
    for layer in model.layers:
        weights = layer.get_weights()
        if 'embedding' in layer.name:
            return weights[0]

def load_w2v_weight(path):
    f = open('NCE/bestModels/' + path, 'rb')
    return cPickle.load(f) 