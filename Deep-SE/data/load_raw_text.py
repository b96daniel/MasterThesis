import pandas
import re
import numpy

def normalize(seqs):
    for i, s in enumerate(seqs):
        words = s.split()                #split a sentence into a list of words
        if len(words) < 1:               #filling out empty cells with Null
            seqs[i] = 'null'

    return seqs

def load_pretrain(path):
    data = pandas.read_csv(path).values
    #return title and description normalized
    return normalize(data[:, 1].astype('str')), normalize(data[:, 2].astype('str'))

def load(path):

    #Capping the highest story-point values (the upper 10%)
    #cut-off value: the highest value of the lower 90%
    def cut_of90(labels):
        #creating dicionary of enumerated set of labels(types of SP values)
        val_y = list(set(labels))
        val_y.sort()
        l_dict = dict()
        for i, val in enumerate(val_y): l_dict[int(val)] = i

        #counting the instances of each type of label
        count_y = [0] * len(val_y)
        for label in labels:
            count_y[l_dict[int(label)]] += 1


        n_samples = len(labels)
        s, threshold = 0, 0
        #Cutting off the upper 10%
        for i, c in enumerate(count_y):
            s += c
            if s * 10 >= n_samples * 9:
                threshold = val_y[i]
                break
        for i, l in enumerate(labels):
            labels[i] = min(threshold, l)

        return labels.astype('float32')

    data = pandas.read_csv(path).values
    title = normalize(data[:, 1].astype('str'))               #Title is column B (0. numpy col is A, is the issuekey)
    description = normalize(data[:, 2].astype('str'))
    labels = data[:, 3].astype('float32')

    return title, description, cut_of90(labels)               #lists(vectors) of strings