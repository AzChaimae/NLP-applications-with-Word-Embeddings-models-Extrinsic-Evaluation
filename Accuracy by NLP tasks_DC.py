#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:02:42 2019

@author: AZROUMAHLI Chaimae
"""

# =============================================================================
# Libraries
# =============================================================================
from glove import Glove

import regex
import re
import os
from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from numpy import zeros
from numpy import array

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential

from keras.layers import Embedding
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import precision_score,recall_score,f1_score

# =============================================================================
# getiing and layering the labeled data
# =============================================================================

#The data need to be cleaned beacuse it being html format
#remove diacritics and non arabic words
def remove_diacritics(text):
    arabic_diacritics = re.compile(" ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ", re.VERBOSE)
    text = re.sub(arabic_diacritics,'',(regex.sub('[^\p{Arabic}]',' ',text)))
    return text 

#normalizing the words in the classified documents
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

def get_data(files_directory):
    taged_documents=[]
    os.chdir(files_directory)
    os.chdir(os.path.join('./culture'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    for file in dict_files:
        with open(file,'r',encoding='utf-8-sig') as f:
            reader=csv.reader(f)
            taged_documents.append([normalizing(list(reader)[0][0]),'culture'])
    os.chdir(files_directory)
    os.chdir(os.path.join('./economy'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    for file in dict_files:
        with open(file,'r',encoding='utf-8-sig') as f:
            reader=csv.reader(f)
            taged_documents.append([normalizing(list(reader)[0][0]),'economy'])
    os.chdir(files_directory)
    os.chdir(os.path.join('./international'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    for file in dict_files:
        with open(file,'r',encoding='utf-8-sig') as f:
            reader=csv.reader(f)
            taged_documents.append([normalizing(list(reader)[0][0]),'international'])
    os.chdir(files_directory)
    os.chdir(os.path.join('./local'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    for file in dict_files:
        with open(file,'r',encoding='utf-8-sig') as f:
            reader=csv.reader(f)
            taged_documents.append([normalizing(list(reader)[0][0]),'local'])
    os.chdir(files_directory)
    os.chdir(os.path.join('./religion'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    for file in dict_files:
        with open(file,'r',encoding='utf-8-sig') as f:
            reader=csv.reader(f)
            taged_documents.append([normalizing(list(reader)[0][0]),'religion'])
    os.chdir(files_directory)
    os.chdir(os.path.join('./sports'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    for file in dict_files:
        with open(file,'r',encoding='utf-8-sig') as f:
            reader=csv.reader(f)
            taged_documents.append([normalizing(list(reader)[0][0]),'sports'])
    return taged_documents

def get_data_labels(labeled_data):
    labels=[row[1] for row in labeled_data]
    return list(set(labels))

def get_data_docs(labeled_data):
    words=[row[0] for row in labeled_data]
    return words



def get_label_to_docs(labeled_data):
    labels=get_data_labels(labeled_data)
    labels=[[label,labels.index(label)] for label in labels]
    def normalizing_labels_to_onehotencoding(text):
        E=to_categorical(array([label[1] for label in labels]))
        for label in labels:
            if label[0]==text:
                A=E[label[1]]
        return A
    labels=[normalizing_labels_to_onehotencoding(row[1]) for row in labeled_data]
    return array(labels)

# ============================================================================
# getting the dictionnaries
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

# =============================================================================
# getting the trainable labeled data
# =============================================================================

#these trainable parametres are called one tim for all the dictionnaries 'they are the same)
def get_trainable_parametres(labeled_data_directory):
    D=get_data(labeled_data_directory)
    docs=get_data_docs(D)
    label_to_docs=get_label_to_docs(D)
    #preparing the tokens
    t=Tokenizer()
    t.fit_on_texts(docs)
    vocab_size=len(t.index_word)+1
    # integer encode the documents
    encoded_docs=t.texts_to_sequences(docs)
    max_length = 100
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    #Split the data into a training sert ant testing set
    Validation_Split=0.2
    indices=np.arange(padded_docs.shape[0])
    np.random.shuffle(indices)
    padded_docs=padded_docs[indices]
    DC_tags=label_to_docs[indices]
    num_validation_simples=int(Validation_Split*padded_docs.shape[0])
    X_train=padded_docs[:-num_validation_simples]
    Y_train=DC_tags[:-num_validation_simples]
    X_test=padded_docs[-num_validation_simples:]
    Y_test=DC_tags[-num_validation_simples:]
    return t,vocab_size,max_length,X_train,Y_train,X_test,Y_test

#vector_dim,embedding_matrix=get_embedding_matrix('Glove.npy')
#This function classifies and returns a model accordinf to the classifiers chosen
def get_model_WE_DC(t,X_train, Y_train,X_test,Y_test,max_length,vocab_size,dictionnary_file):
   #to avoid the problem of Memory error, we need to creae a separated fiunction that saves the embedding matrix
    def get_embedding_matrix(t,vocab_size,dictionnary_file):
        # function for loading the whole embedding into memory
        #The function here depends on the model (Word2vec or Glove)
        embeddings_index=get_WE_Glove(dictionnary_file)
        vector_dim=len(list(embeddings_index.values())[0])
        print('Loaded %s word vectors.' % len(embeddings_index))
        # create a weight matrix for words in training docs
        embedding_matrix = zeros((vocab_size, vector_dim))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        del embedding_vector,embeddings_index
        return vector_dim,embedding_matrix
    vector_dim,embedding_matrix=get_embedding_matrix(t,vocab_size,dictionnary_file)
    #define model
    model = Sequential()
    e = Embedding(vocab_size,vector_dim,weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Dropout(0.2))
    model.add(LSTM(vector_dim))
    model.add(Dense(6, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize the model
    print(model.summary())
    model.fit(X_train, Y_train, batch_size=225 , epochs=3,validation_data=(X_test,Y_test), verbose=1)
    del embedding_matrix
    return model

#evaluate the model
#this functions should return the accuracy, precsion and Recall in a list
def get_DC_accuracy(model,X_test,Y_test):
    # predict probabilities for test set
    y_probs = model.predict(X_test, verbose=1)
    # predict crisp classes for test set
    y_classes = model.predict_classes(X_test, verbose=1)
    # reduce to 1d array
    y_probs = y_probs[:, 0]
    Y_test1=array([np.where(y==1)[0][0] for y in Y_test])
    # accuracy: (tp + tn) / (p + n)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test1, y_classes,average='weighted')
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test1, y_classes,average='weighted')
    #f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test1, y_classes,average='weighted')
    return accuracy,precision,recall,f1

# =============================================================================
# the accuracies in one file
# =============================================================================
#This function should calculate ans save the diffrent accuracies and precsions in one file
def POS_accuracies_in_a_file(corpus_directory,results_file):
    #getting the dictionnaries to test
    os.chdir(corpus_directory)
    os.chdir(os.path.join('./models'))
    dict_files=[f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(),f))]
    print('The names of the dictionnary files are read and waiting to be tested')
    os.chdir(corpus_directory)
    #Calculating the accuracy for each file
    with open(results_file,'a',encoding='utf-8-sig') as results:
        writer=csv.writer(results)
        writer.writerow(["dictionnary's name", "Accuracy", "Precision", "Recall", "F1_measure"])
    print('The file: %s is created'%(results_file))
    #the training data for all the models 
    for file in dict_files:
        t,vocab_size,max_length,X_train,Y_train,X_test,Y_test=get_trainable_parametres(labeled_data_directory)
        print('the training labeled data are loaded')
        os.chdir(corpus_directory)
        os.chdir(os.path.join('./models'))
        print('\n********* The training of %s is starting *********\n'%(file))
        model=get_model_WE_DC(t,X_train,Y_train,X_test,Y_test,max_length,vocab_size,file)
        accuracy,precision,recall,f1=get_DC_accuracy(model,X_test,Y_test)
        print("%s 's accuracies are:\n accuracy=%f \n precision=%f \n recall=%f \n f1=%f "%(file,accuracy,precision,recall,f1))
        os.chdir(corpus_directory)
        #the accuracies are list that contains the statistics of each relation with the accurucy obtained
        with open(results_file,'a',encoding='utf-8-sig') as results:
            writer=csv.writer(results)
            writer.writerow([file,accuracy,precision,recall,f1])
        print("%s's accuracy is computed and noted in the accuracies file"%(file))
        del model, accuracy, precision, recall, f1
    return

# =============================================================================
# Application
# =============================================================================

labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/Document classification'
corpus_directory='/home/ubuntu/embeddings_analysis/Glove/wiki'
results_file='wiki_Glove_DC_accuracies.csv'

POS_accuracies_in_a_file(corpus_directory,results_file)