EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
FILENAME = 'datasets/twitter_en.txt'
limit = {
        'maxq' : 20,
        'minq' : 0,
        'maxa' : 20,
        'mina' : 3
        }
UNK = 'unk'
VOCAB_SIZE = 6000
TRAINING_SIZE = 10000

import random
import sys
import nltk
import itertools
from collections import defaultdict
import numpy as np
import pickle
from keras.preprocessing.text import text_to_word_sequence

'''
 read lines from file
     return [TRAINING_SIZE list of lines]

'''
def read_lines(filename):
    return open(filename).read().split('\n')[:TRAINING_SIZE]

'''
 remove anything that isn't in the whitelist
    return pure english

'''
def filter_line(line, whitelist):
    return ''.join([ char for char in line if char in whitelist ])

'''
 filter too long and too short sequences
    return tuple( filtered_question, filtered_answer )

'''
def filter_data(sequences):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(text_to_word_sequence(sequences[i])), len(text_to_word_sequence(sequences[i+1]))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa'] - 2:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])
    
    # print the fraction of the original data, filtered
    # filt_data_len = len(filtered_q)
    # filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    # print(str(filtered) + '% filtered from original data')
    return filtered_q, filtered_a

'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( idx2w, w2idx, vocab->(word, count) )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    #print(vocab[:11])
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist

'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_q([indices]), array_a([indices]) )
 
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)
    idx_target_decoder = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])
        target_decoder_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])[1:]
        target_decoder_indices.append(0)
        
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        idx_target_decoder[i] = np.array(target_decoder_indices)

    return idx_q, idx_a, idx_target_decoder

def zero_pad_and_hot_encode(test_sentence, w2idx, vocab_len):
    # num of rows
    test_sentence_tokenized = [ wordlist.lower() for wordlist in test_sentence.split(' ') ]
    print("Tokenized sentence:")
    print(test_sentence_tokenized)
    
    data_len = len(test_sentence_tokenized)

    # numpy arrays to store indices
    # idx_test_sentence = np.zeros([data_len, limit['maxq']], dtype=np.int32)
 
    test_sentence_indices = pad_seq(test_sentence_tokenized, w2idx, limit['maxq'])
    idx_test_sentence = np.array(test_sentence_indices)
    print("Numpy array for the sentence:")
    print(idx_test_sentence)

    # one hot encode
    onehot_encoded = list()
    for value in idx_test_sentence:
        word = [0 for _ in range(vocab_len)]
        word[value] = 1
        onehot_encoded.append(word)
        
    return onehot_encoded

'''
 replace words with indices in a sequence
  replace with unknown if word not in vocabulary
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

'''
one hot encoding given a sequence
  return [one-hot encoded vector]
  
'''
def one_hot_encode(seq, vocab_len):
    final_list = list()
    for num in range(len(seq)):        
        # one hot encode
        onehot_encoded = list()
        for value in seq[num]:
            word = [0 for _ in range(vocab_len)]
            word[value] = 1
            onehot_encoded.append(word)
        final_list.append(onehot_encoded)
    return final_list








# PROCESS THE DATA
lines = read_lines(FILENAME)
lines = [ line.lower() for line in lines ]
lines = [ filter_line(line, EN_WHITELIST) for line in lines ]
qlines, alines = filter_data(lines)

# Add start and end tags to the answers
answers = []
for answer in alines:
    answer = '<str> ' + answer + ' <end>'
    answers.append(answer)

qtokenized = []
atokenized = []

for q in qlines:
    qlist = text_to_word_sequence(q)
    qtokenized.append(qlist)

for a in answers:
    alist = text_to_word_sequence(a)
    atokenized.append(qlist)

# Create the necessary metadata
idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, VOCAB_SIZE)

# Create the questions and responses
idx_q, idx_a, idx_target_decoder = zero_pad(qtokenized, atokenized, w2idx)

# Reverse questions token order
idx_q = [question[::-1] for question in idx_q]

# Save numpy arrays
np.save('idx_q.npy', idx_q)
np.save('idx_a.npy', idx_a)
np.save('idx_target_decoder.npy', idx_target_decoder)

# Save the metadata
metadata = {
        'w2idx' : w2idx,
        'idx2w' : idx2w,
        'limit' : limit,
        'freq_dist' : freq_dist
            }

# Write to disk
with open('metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# USED in the actual word_embedding_seq2seq.py
def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    idx_target_decoder = np.load(PATH + 'idx_target_decoder.npy')
    return metadata, idx_q, idx_a, idx_target_decoder