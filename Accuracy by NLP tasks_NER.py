#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:29:10 2019

@author: AZROUMAHLI Chaimae
"""

# =============================================================================
# Libraries
# =============================================================================
import io
from glove import Glove
import regex 
import re
import csv 
import os 
from os import listdir
from os.path import isfile, join
import numpy as np 
from numpy import zeros

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential

from keras.layers import Embedding
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import precision_score,recall_score,f1_score

# =============================================================================
# Reading, preprocessing & Layering the labeled Data
# =============================================================================
#AQMAR NER tag needs to be cleaned as well there is a lot of emty and non arabic entries
def remove_diacritics(text):
    arabic_diacritics = re.compile(" ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ", re.VERBOSE)
    text = re.sub(arabic_diacritics,'',(regex.sub('[^\p{Arabic}]','',text)))
    return text 

def normalizing(text):
    a='ا'
    b='ء'
    c='ه'
    d='ي'
    text=regex.sub('[آ]|[أ]|[إ]',a,text)
    text=regex.sub('[ؤ]|[ئ]',b,text)
    text=regex.sub('[ة]',c,text)
    text=regex.sub('[ي]|[ى]',d,text)
    return remove_diacritics(text)

def remove_empty_lines(filename):
    #Overwrite the file, removing empty lines and lines that contain only whitespace.
    with open(filename,encoding='utf-8-sig') as in_file, open(filename,'r+',encoding='utf-8-sig') as out_file:
        out_file.writelines(line for line in in_file if line.strip())
        out_file.truncate()

def normalizing_labels(text):
    text=regex.sub('--ORG','-ORG',text)
    text=regex.sub('MIS2','MIS',text)
    text=regex.sub('MIS0','MIS',text)
    text=regex.sub('MIS1','MIS',text)
    text=regex.sub('MIS-1','MIS',text)
    text=regex.sub('MIS3','MIS',text)
    text=regex.sub('MISS1','MIS',text)
    text=regex.sub('MIS`','MIS',text)
    text=regex.sub('IO','O',text)
    return text

#get the NER annotated data 
def get_data(files_directory):
    NER_test=[]
    NER_files=[f for f in listdir(files_directory) if isfile(join(files_directory,f))]
    os.chdir(files_directory)
    for file in NER_files:
        remove_empty_lines(file)
        f=open(file,'r',encoding='utf-8-sig')
        reader=csv.reader(f)
        rows=list(reader)
        for row in rows:
            try:
                NER_test.append([normalizing(row[0].split(' ')[0]),normalizing_labels(row[0].split(' ')[1])])
            except:
                pass
        f.close()
    NER_test_annotations=[row for row in NER_test if row[0]!='']
    return NER_test_annotations


def get_data_labels(labeled_data):
    labels=[row[1] for row in labeled_data]
    return list(set(labels))

def get_data_words(labeled_data):
    words=[row[0] for row in labeled_data]
    return words

def normalizing_labels_to_float(text):
    text=re.sub('B-MIS','2',text)
    text=re.sub('B-Per','3',text)
    text=re.sub('B-PER','3',text)
    text=re.sub('B-ORG','4',text)
    text=re.sub('B-LOC','5',text)
    text=re.sub('I-MIS','6',text)
    text=re.sub('I-Per','7',text)
    text=re.sub('I-PER','7',text)
    text=re.sub('I-ORG','8',text)
    text=re.sub('I-LOC','9',text)
    text=re.sub('O','1',text)
    return text

def get_label_to_word(labeled_data):
    labels=[normalizing_labels_to_float(row[1]) for row in labeled_data]
    return np.asarray(list(map(int,labels)),dtype=np.int32)

# =============================================================================
# Reading & Layering the dictionnary
# =============================================================================
#For Glove: reading it from a model
def get_WE_Glove(model_file):
    model=Glove.load(model_file)
    Glove_dict={}
    for key,val in model.dictionary.items():
        Glove_dict.update({key:model.word_vectors[val]})
    return Glove_dict
#For Word2ve: reading it directly from a npy dicrtionnary
def get_WE_Word2Vec(dictionnary_file):
    return np.load(dictionnary_file).item()

def load_vectors(fname,fdirectory):
    os.path()
    fin = io.open(fname, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        try:
            data[tokens[0]] = list(map(float,tokens[1:]))
        except:
            pass
    return data

# =============================================================================
# Clustering & Accuracy
# =============================================================================
#this fucnction should train and return the accuracy, precsion and Recall in a list
def get_model_WE_NER(dictionnary_file):
    #preparing the tokens
    t=Tokenizer()
    t.fit_on_texts(words)
    # integer encode the documents
    encoded_docs=t.texts_to_sequences(words)
    # pad documents to a max length of 1 word since our "docs" contain only one word each
    max_length = 1
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    
    #Split the data into a training sert ant testing set
    Validation_Split=0.2
    indices=np.arange(padded_docs.shape[0])
    np.random.shuffle(indices)
    padded_docs=padded_docs[indices]
    NER_tags=label_to_word[indices]
    num_validation_simples=int(Validation_Split*padded_docs.shape[0])
    X_train=padded_docs[:-num_validation_simples]
    Y_train=NER_tags[:-num_validation_simples]
    X_test=padded_docs[-num_validation_simples:]
    Y_test=NER_tags[-num_validation_simples:]
    
    # load the whole embedding into memory
    #The function here depends on the model (Word2vec or Glove)
    embeddings_index=load_vectors(dictionnary_file)
    vector_dim=len(list(embeddings_index.values())[0])
    print('Loaded %s word vectors.' % len(embeddings_index))
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, vector_dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    #define model
    model = Sequential()
    e = Embedding(vocab_size,vector_dim,weights=[embedding_matrix], input_length=1,trainable=False)
    model.add(e)
    model.add(Dropout(0.2))
    model.add(LSTM(vector_dim))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # summarize the model
    print(model.summary())
    model.fit(X_train, Y_train, batch_size=128, epochs=4, validation_data=(X_test,Y_test), verbose=2)
    return get_NER_accuracy(model,X_test,Y_test)

#evaluate the model
#this functions should return the accuracy, precsion and Recall in a list
def get_NER_accuracy(model,X_test,Y_test):
    # predict probabilities for test set
    y_probs = model.predict(X_test, verbose=2)
    # predict crisp classes for test set
    y_classes = model.predict_classes(X_test, verbose=2)
    # reduce to 1d array
    y_probs = y_probs[:, 0]
    y_classes = y_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
    print('Accuracy: %f ' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, y_classes,average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, y_classes,average='weighted')
    print('Recall: %f' % recall)
    #    f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, y_classes,average='weighted')
    print('f1: %f' % recall)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
    return accuracy,precision,recall,f1

#test
#data=get_data('/home/khaosdev-6/AZROUMAHLI Chaimae/Embeddings analysis/Accuracy/My Arabic word-embeddings benchmarks/txt_NER tags')
#label_to_word=get_label_to_word(data)
#words=get_data_words(data)
#vocab_size=len(words)
#accuracy,precision,recall,f1= get_model_WE_NER('Glove.npy',label_to_word,words,vocab_size)

# =============================================================================
# Clustering for all the models
# =============================================================================

#This function should calculate ans save the diffrent accuracies and precsions in one file
def NER_accuracies_in_a_file(corpus_directory,results_file):
    #getting the models to test
    os.chdir(corpus_directory)
    os.chdir(os.path.join('./dictionnaries'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    print('The dictionnary files are read and waiting to be tested')
    os.chdir(corpus_directory)
    #Calculating the accuracy for each file
    with open(results_file,'a',encoding='utf-8-sig') as results:
        writer=csv.writer(results)
        writer.writerow(["dictionnary's name", "Accuracy", "Precesion", "Recall", "F1_measure"])
    print('The file: %s is created'%(results_file))
    for file in dict_files:
        os.chdir(os.path.join('./dictionnaries'))
        accuracy,precision,recall,f1= get_model_WE_NER(file)
        print("%s 's accuracies are:\n accuracy=%f \n precision=%f \n recall=%f \n f1=%f "%(file,accuracy,precision,recall,f1))
        os.chdir(corpus_directory)
        #the accuracies are list that contains the statistics of each relation with the accurucy obtained
        with open(results_file,'a',encoding='utf-8-sig') as results:
            writer=csv.writer(results)
            writer.writerow([file,accuracy,precision,recall,f1])
        print("%s 's accuracy is computed and noted in the accuracies file"%(file))
    return

# =============================================================================
# Applications
# =============================================================================
#Calculating the NER accuracies for the different models
#The needed parametres 
labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/txt_NER tags'
corpus_directory='/home/ubuntu/embeddings_analysis/FastText'
results_file='Fasttext_NER_accuracies.csv'

#Calling the labeled data for training
data=get_data(labeled_data_directory)
label_to_word=get_label_to_word(data)
words=get_data_words(data)
vocab_size=len(words)
#the accuracies in a file
NER_accuracies_in_a_file(corpus_directory,results_file)