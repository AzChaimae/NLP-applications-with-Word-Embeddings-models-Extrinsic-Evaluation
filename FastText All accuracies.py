#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:26:28 2019

@author: AZROUMAHLI Chaimae
"""

# =============================================================================
# Libraries
# =============================================================================
import io 
import regex
import re 
import csv
import numpy as np
from numpy import array
from numpy import zeros
import os
from os import listdir
from os.path import isfile, join
import pandas as pd 
import nltk 
from nltk.cluster import KMeansClusterer

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Embedding
from keras.layers import Dense, LSTM, Dropout

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score

# =============================================================================
# Reading and Layering the Fast Text CBOW and Skip Gram
# =============================================================================

def load_vectors(fname):
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
# Useful function for cleaning the Benchmarks
# =============================================================================

#normalizing function from the pre-proceesing steps
def remove_diacritics(text):
    arabic_diacritics = re.compile(" ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ", re.VERBOSE)
    text = re.sub(arabic_diacritics,'',text)
    return text 

#Normalizing: input and output is a text
def normalizing(string):
    a='ا'
    b='ء'
    c='ه'
    d='ي'
    string=regex.sub('[آ]|[أ]|[إ]',a,string)
    string=regex.sub('[ؤ]|[ئ]',b,string)
    string=regex.sub('[ة]',c,string)
    string=regex.sub('[ي]|[ى]',d,string)
    return remove_diacritics(string)

#removing empty lines from a file 
def remove_empty_lines(filename):
    #Overwrite the file, removing empty lines and lines that contain only whitespace.
    with open(filename,encoding='utf-8-sig') as in_file, open(filename,'r+',encoding='utf-8-sig') as out_file:
        out_file.writelines(line for line in in_file if line.strip())
        out_file.truncate()

# =============================================================================
# Function for calclating Tha Analogy accuracy
# =============================================================================
#reading the WAT benchmarks
def get_WAT_relation_questions(questions_directory):
    questions=[]
    #detecting the files
    testing_files=[f for f in listdir(questions_directory) if isfile(join(questions_directory,f))]
    #Openin,g the directory
    os.chdir(questions_directory)
    #reading and listing the files' content
    for file in testing_files:
        with open(file,'r', encoding='utf-8-sig') as f:
            reader=csv.reader(row.replace('\0','') for row in f)
            paires=list(reader)
            for row in paires:
                questions.append([normalizing(r) for r in row])
            print('%s has been appended'%(file))
    return questions

#getting WAT accuracy: the ouput is according to the files either a semantic or syntactic
#I have to call both questions and for each set of question I'll calculate the accuracy
def get_WAT_accuracy(testing_questions,dictionnary_to_test):
    words=[keys for keys in dictionnary_to_test.keys()]
    vectors=dictionnary_to_test
    vocab_size=len(words)
    vocab={w:idx for idx,w in enumerate(words)}
    ivocab={idx:w for idx,w in enumerate(words)}
    vector_dim=len(list(vectors[ivocab[0]]))
    W=np.zeros((vocab_size,vector_dim))
    for word,v in vectors.items():
        W[vocab[word],:]=v
    #normalization des vecteur à des (unit variances or zero means)
    W_norm=np.zeros(W.shape)
    d=(np.sum(W ** 2,1)**(0.5))
    W_norm=(W.T/d).T
    split_size=100 #Memory overflow
    correct_que=0 #number of correct questions
    count_que_an=0 # questions answered
    count_que=0 #count all the questions
    count_que+=len(testing_questions)
    data=[x for x in testing_questions if all(word in vocab for word in x)]
    indices=np.array([[vocab[word] for word in row] for row in data])
    ind1, ind2, ind3, ind4=indices.T
    predictions=np.zeros((len(indices),))
    num_iter=int(np.ceil(len(indices)/float(split_size))) 
    for j in range(num_iter):
        subset=np.arange(j*split_size, min((j + 1)*split_size, len(ind1)))
        pred_vec=(W_norm[ind2[subset],:] - W_norm[ind1[subset],:] + W_norm[ind3[subset],:])
        #cosine similarity if input W has been normalized
        dist=np.dot(W_norm,pred_vec.T)
        for k in range(len(subset)):
            dist[ind1[subset[k]], k] = -np.Inf
            dist[ind2[subset[k]], k] = -np.Inf
            dist[ind3[subset[k]], k] = -np.Inf
        #predected word index
        predictions[subset]=np.argmax(dist,0).flatten()
    val=(ind4==predictions) 
    count_que_an=len(ind1)
    correct_que=sum(val)
    print('Total accuracy: %.2f%%  (%i/%i)' % (100*correct_que/float(count_que_an), correct_que, count_que))
    return 100*correct_que/float(count_que_an)

# =============================================================================
# Function for Calculating The categorization accuracy
# =============================================================================

#get the categories from the file
def get_CAT_categories(categories_directory,categories_file):
    os.chdir(categories_directory)
    with open(categories_file,'r',encoding='utf-8-sig') as file:
        reader=csv.reader(row.replace('/0','') for row in file)
        row_categories=list(reader)
    categories=[]
    for row1 in row_categories:
        test=[categories[i][0] for i in range(len(categories))]
        if (row1[0] not in test):
            list_of_content=[]
            list_of_content.append(normalizing(row1[1]))
            for row2 in row_categories:
                if row_categories.index(row1)!=row_categories.index(row2) and row1[0]==row2[0]:
                    list_of_content.append(normalizing(row2[1]))
            categories.append([row1[0],list_of_content])
    return categories

def get_CAT_true_clusters(word_representations,word_categories):
    true_word_clusters=[]
    i=0
    for row in word_categories:
        for word in word_representations.keys():
            if word in row[1] and word not in [true_word_clusters[i][1] for i in range(len(true_word_clusters))]: 
                true_word_clusters.append([row[0],word,i])
        i+=1
    return [true_word_clusters[i][2] for i in range(len(true_word_clusters))]

def get_kmeans_CAT_predicted_clusters(word_representions,Num_clusters):
    #from dictionnary type to transposed dataframe
    Y=pd.DataFrame(data=word_representions).T
    X=Y.values
    #Clustering the data using sklearn library
    kclusterer = KMeansClusterer(Num_clusters, distance=nltk.cluster.util.euclidean_distance, repeats=25, avoid_empty_clusters=False)
    predicted_clusters= kclusterer.cluster(X, assign_clusters=True, )
    return predicted_clusters

def CAT_purity_score(y_true_clusters,y_predicted_clusters):
    y_voted_labels=np.zeros(y_true_clusters.shape)
    labels=np.unique(y_true_clusters)
    ordered_labels=np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true_clusters[y_true_clusters==labels[k]]=ordered_labels[k]
    labels=np.unique(y_true_clusters)
    bins=np.concatenate((labels,[np.max(labels)+1]),axis=0)
    for cluster_ in np.unique(y_predicted_clusters):
        hist, _ =np.histogram(y_true_clusters[y_predicted_clusters==cluster_],bins=bins)
        winner=np.argmax(hist)
        y_voted_labels[y_predicted_clusters==cluster_]=winner
    return accuracy_score(y_true_clusters,y_voted_labels)

def get_CAT_dictionnary_benchmarks(dictionnary,categories_list):
    vector_representations=np.load(dictionnary).item()
    vector_dictionnary={}
    for row in categories_list:
        for i in range(len(row[1])):
            for key,val in vector_representations.items():
                if key==row[1][i]:
                    vector_dictionnary.update({key:val})
    print('categories representation is generated')
    return vector_dictionnary

# =============================================================================
# Function for reading and extracting the benchmarks for the classification algorithms
# =============================================================================
def get_data_labels(labeled_data):
    labels=[row[1] for row in labeled_data]
    return list(set(labels))

def get_data_words(labeled_data):
    words=[row[0] for row in labeled_data]
    return words

def get_label_to_word(labeled_data):
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

# =============================================================================
# Function for extracting the embeddings matrix a and trainable parametres 
#for classification algorithm
# =============================================================================

def get_Keras_accuracy(model,X_test,Y_test):
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
# Function for calculating The NER tag Accuracy
# =============================================================================
def normalizing_NER_labels(text):
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

def get_NER_data(files_directory):
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
                NER_test.append([normalizing(row[0].split(' ')[0]),normalizing_NER_labels(row[0].split(' ')[1])])
            except:
                pass
        f.close()
    NER_test_annotations=[row for row in NER_test if row[0]!='']
    return NER_test_annotations

def get_model_WE_NER(dictionnary_file,words,vocab_size):
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
    NER_tags=get_label_to_word[indices]
    num_validation_simples=int(Validation_Split*padded_docs.shape[0])
    X_train=padded_docs[:-num_validation_simples]
    Y_train=NER_tags[:-num_validation_simples]
    X_test=padded_docs[-num_validation_simples:]
    Y_test=NER_tags[-num_validation_simples:]
    
    # load the whole embedding into memory
    #The function here depends on the model (Word2vec or Glove)
    embeddings_index=dictionnary_file
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
    model.add(Dense(9, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize the model
    print(model.summary())
    model.fit(X_train, Y_train, batch_size=128, epochs=4, validation_data=(X_test,Y_test), verbose=2)
    return get_Keras_accuracy(model,X_test,Y_test)

# =============================================================================
# Function for calculating the POS tag Accuracy
# =============================================================================
def get_POS_data(files_directory):
    POS_test=[]
    POS_files=[f for f in listdir(files_directory) if isfile(join(files_directory,f))]
    os.chdir(files_directory)
    for file in POS_files:
        with open(file,'r',encoding='utf-8-sig') as f:
            reader=csv.reader(f,delimiter=',')
            for row in list(reader):
                try:
                    POS_test.append([normalizing(row[0]),row[1]])
                except:
                    pass
    POS_test=[row for row in POS_test if row[0]!='']
    return POS_test

def get_POS_trainable_parametres(labeled_data_directory):
    D=get_POS_data(labeled_data_directory)
    docs=get_data_words(D)
    label_to_docs=get_label_to_word(D)
    #preparing the tokens
    t=Tokenizer()
    t.fit_on_texts(docs)
    vocab_size=len(t.index_word)+1
    # integer encode the documents
    encoded_docs=t.texts_to_sequences(docs)
    max_length=1
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
    return t,vocab_size,X_train,Y_train,X_test,Y_test

def get_model_WE_POS(t,X_train, Y_train,X_test,Y_test,vocab_size,dictionnary):
    #to avoid the problem of Memory error, we need to creae a separated fiunction that saves the embedding matrix
    def get_embedding_matrix(t,vocab_size,dictionnary_file):
        # function for loading the whole embedding into memory
        #The function here depends on the model (Word2vec or Glove)
        embeddings_index=dictionnary
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
    vector_dim,embedding_matrix=get_embedding_matrix(t,vocab_size,dictionnary)
    #define model
    model = Sequential()
    e = Embedding(vocab_size,vector_dim,weights=[embedding_matrix], input_length=1,trainable=False)
    model.add(e)
    model.add(Dropout(0.2))
    model.add(LSTM(vector_dim))
    model.add(Dense(30, activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # summarize the model
    print(model.summary())
    model.fit(X_train, Y_train, batch_size=225, epochs=3, validation_data=(X_test,Y_test), verbose=2)
    del embedding_matrix
    return model

# =============================================================================
# Function Fior calculating The Document classification accuracy
# =============================================================================
def get_DC_data(files_directory):
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

def get_DC_trainable_parametres(labeled_data_directory):
    D=get_DC_data(labeled_data_directory)
    docs=get_data_words(D)
    label_to_docs=get_label_to_word(D)
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

def get_model_WE_DC(t,X_train, Y_train,X_test,Y_test,max_length,vocab_size,dictionnary):
   #to avoid the problem of Memory error, we need to creae a separated fiunction that saves the embedding matrix
    def get_embedding_matrix(t,vocab_size,dictionnary):
        # function for loading the whole embedding into memory
        #The function here depends on the model (Word2vec or Glove)
        embeddings_index=dictionnary
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
    vector_dim,embedding_matrix=get_embedding_matrix(t,vocab_size,dictionnary)
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

# =============================================================================
# Function for saving the resulted accuracy and it's task
# =============================================================================
#acc_file the name of the file, acc_tast is the name of the task acc is the accuracy it self
def save_acc_in_a_file(acc_file,acc_task,acc):
    os.chdir(Fast_text_data_directory)
    with open(acc_file,'a',encoding='utf-8-sig') as file:
        writer=csv.writer(file)
        writer.writerow([acc_task,acc])
    return 

# =============================================================================
# Application
# =============================================================================
Fast_text_data_directory='/home/ubuntu/embeddings_analysis/FastText'

SG_file_name='wiki.ar.vec'
SG_results_file_name='SG_Fasttext.csv'

print('********Loading the dictionnaries to the memory ********')
os.chdir(Fast_text_data_directory)
SG_dict=load_vectors(SG_file_name)
print('Skip-Gram vectors are loaded succefully')

print('*************Starting the Analgoy test*******************')
Syntactic_path="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/csv: Morphosyntacitc analogies"
Semantic_path="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/csv: Semantic analogies"

try:
    Semantic_questions=get_WAT_relation_questions(Semantic_path)
    Syntactic_questions=get_WAT_relation_questions(Syntactic_path)
    print('Calculating SG semantic analogy accuracy')
    SG_Semantic_Accuracy=get_WAT_accuracy(Semantic_questions,SG_dict)
    print('Calculating SG Morphosyntactic analogy accuracy')
    SG_Syntactic_Accuracy=get_WAT_accuracy(Syntactic_questions,SG_dict)
    save_acc_in_a_file(SG_results_file_name,'Semantic Accuracy',SG_Semantic_Accuracy)
    save_acc_in_a_file(SG_results_file_name,'Syntactic Accuracy',SG_Syntactic_Accuracy)
    print('The Word Analogy test is done for the Skip Gram')
    del Syntactic_path,Syntactic_questions,Semantic_path,Semantic_questions,SG_Semantic_Accuracy,SG_Syntactic_Accuracy
except:
    pass

print('*************Starting the Categorization test*******************')
categories_directory="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks"
categories_file="word categories Arabe.csv"

try:
    Num_clusters=22
    categories=get_CAT_categories(categories_directory,categories_file)
    print('Calculating SG categorization accuracy')
    categories_representation=get_CAT_dictionnary_benchmarks(SG_dict,categories)
    true_clusters=get_CAT_true_clusters(categories_representation,categories)
    print('true clusters generated')
    predicted_clusters=get_kmeans_CAT_predicted_clusters(categories_representation,Num_clusters)
    print('predicted clusters generated')
    CAT_SG_accuracy=CAT_purity_score(np.asarray(true_clusters),np.asarray(predicted_clusters))
    save_acc_in_a_file(SG_results_file_name,'Categorization Accuracy',CAT_SG_accuracy)
    del categories_representation,true_clusters,predicted_clusters,CAT_SG_accuracy
    del categories,categories_directory,categories_file,Num_clusters
except:
    pass

print('*************Starting the NER tag test*******************')
labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/txt_NER tags'
try:
    data=get_NER_data(labeled_data_directory)
    label_to_word=get_label_to_word(data)
    words=get_data_words(data)
    vocab_size=len(words)
    print('Calculating SG NER tag accuracy')
    accuracy,precision,recall,f1= get_model_WE_NER(SG_dict,words,vocab_size)
    save_acc_in_a_file(SG_results_file_name,'NER Accuracies',[accuracy,precision,recall,f1])
    del accuracy,precision,recall,f1
    del labeled_data_directory,data,label_to_word,words,vocab_size
except:
    pass

print('*************Starting the POS tag test*******************')
labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/txt_POS tags'
try:
    t,vocab_size,X_train,Y_train,X_test,Y_test=get_POS_trainable_parametres(labeled_data_directory)
    print('Calculating SG NER tag accuracy')
    model=get_model_WE_POS(t,X_train,Y_train,X_test,Y_test,vocab_size,SG_dict)
    accuracy,precision,recall,f1=get_Keras_accuracy(model,X_test,Y_test)
    save_acc_in_a_file(SG_results_file_name,'POS Accuracies',[accuracy,precision,recall,f1])
    del model,accuracy,precision,recall,f1
    del labeled_data_directory,t,vocab_size,X_train,Y_train,X_test,Y_test
except:
    pass

print('*************Starting the DC tag test*******************')
labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/Document classification'
try:
    t,vocab_size,max_length,X_train,Y_train,X_test,Y_test=get_DC_trainable_parametres(labeled_data_directory)

    print('Calculating SG NER tag accuracy')
    model=get_model_WE_DC(t,X_train,Y_train,X_test,Y_test,max_length,vocab_size,SG_dict)
    accuracy,precision,recall,f1=get_Keras_accuracy(model,X_test,Y_test)
    save_acc_in_a_file(SG_results_file_name,'DC Accuracies',[accuracy,precision,recall,f1])
    del model,accuracy,precision,recall,f1
    print('Calculating SG NER tag accuracy')
    del labeled_data_directory,t,vocab_size,X_train,Y_train,X_test,Y_test
except:
    pass

del SG_dict

print('*****************Finished check the file****************************')


CBOW_file_name='cc.ar.300.vec'
CBOW_results_file_name='CBOW_Fasttext.csv'

print('********Loading the dictionnaries to the memory ********')
os.chdir(Fast_text_data_directory)
CBOW_dict=load_vectors(CBOW_file_name)
print('CBOW vectors are loaded succefully')


print('*************Starting the Analgoy test*******************')
Syntactic_path="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/csv: Morphosyntacitc analogies"
Semantic_path="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/csv: Semantic analogies"

try:
    Semantic_questions=get_WAT_relation_questions(Semantic_path)
    Syntactic_questions=get_WAT_relation_questions(Syntactic_path)
    print('Calculating CBOW semantic analogy accuracy')
    CBOW_Semantic_Accuracy=get_WAT_accuracy(Semantic_questions,CBOW_dict)
    print('Calculating CBOW Morphosyntactic analogy accuracy')
    CBOW_Syntactic_Accuracy=get_WAT_accuracy(Syntactic_questions,CBOW_dict)
    save_acc_in_a_file(CBOW_results_file_name,'Semantic Accuracy',CBOW_Semantic_Accuracy)
    save_acc_in_a_file(CBOW_results_file_name,'Syntactic Accuracy',CBOW_Syntactic_Accuracy)
    print('The Word Analogy test is done for the CBOW')
    del Syntactic_path,Syntactic_questions,Semantic_path,Semantic_questions,CBOW_Semantic_Accuracy,CBOW_Syntactic_Accuracy
except:
    pass


print('*************Starting the Categorization test*******************')
categories_directory="/home/ubuntu/embeddings_analysis/Accuracy/benchmarks"
categories_file="word categories Arabe.csv"
try:
    Num_clusters=22
    categories=get_CAT_categories(categories_directory,categories_file)
    
    print('Calculating CBOW categorization accuracy')
    categories_representation=get_CAT_dictionnary_benchmarks(CBOW_dict,categories)
    true_clusters=get_CAT_true_clusters(categories_representation,categories)
    print('true clusters generated')
    predicted_clusters=get_kmeans_CAT_predicted_clusters(categories_representation,Num_clusters)
    print('predicted clusters generated')
    CAT_CBOW_accuracy=CAT_purity_score(np.asarray(true_clusters),np.asarray(predicted_clusters))
    save_acc_in_a_file(CBOW_results_file_name,'Categorization Accuracy',CAT_CBOW_accuracy)
    del categories_representation,true_clusters,predicted_clusters,CAT_CBOW_accuracy
    del categories,categories_directory,categories_file,Num_clusters
except:
    pass


print('*************Starting the NER tag test*******************')
labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/txt_NER tags'
try:
    data=get_NER_data(labeled_data_directory)
    label_to_word=get_label_to_word(data)
    words=get_data_words(data)
    vocab_size=len(words)
    
    print('Calculating CBOW NER tag accuracy')
    accuracy,precision,recall,f1= get_model_WE_NER(CBOW_dict,words,vocab_size)
    save_acc_in_a_file(CBOW_results_file_name,'NER Accuracies',[accuracy,precision,recall,f1])
    del accuracy,precision,recall,f1
    del labeled_data_directory,data,label_to_word,words,vocab_size
except:
    pass


print('*************Starting the POS tag test*******************')
labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/txt_POS tags'
try:
    t,vocab_size,X_train,Y_train,X_test,Y_test=get_POS_trainable_parametres(labeled_data_directory)
    print('Calculating CBOW NER tag accuracy')
    model=get_model_WE_POS(t,X_train,Y_train,X_test,Y_test,vocab_size,CBOW_dict)
    accuracy,precision,recall,f1=get_Keras_accuracy(model,X_test,Y_test)
    save_acc_in_a_file(CBOW_results_file_name,'POS Accuracies',[accuracy,precision,recall,f1])
    del labeled_data_directory,t,vocab_size,X_train,Y_train,X_test,Y_test,model,accuracy,precision,recall,f1
except:
    pass

print('*************Starting the DC tag test*******************')
labeled_data_directory='/home/ubuntu/embeddings_analysis/Accuracy/benchmarks/Document classification'
try:
    t,vocab_size,max_length,X_train,Y_train,X_test,Y_test=get_DC_trainable_parametres(labeled_data_directory)

    print('Calculating SG NER tag accuracy')
    model=get_model_WE_DC(t,X_train,Y_train,X_test,Y_test,max_length,vocab_size,CBOW_dict)
    accuracy,precision,recall,f1=get_Keras_accuracy(model,X_test,Y_test)
    save_acc_in_a_file(CBOW_results_file_name,'DC Accuracies',[accuracy,precision,recall,f1])
    del labeled_data_directory,t,vocab_size,X_train,Y_train,X_test,Y_test,model,accuracy,precision,recall,f1
except:
    pass
del CBOW_dict

print('*****************Finished check the file****************************')