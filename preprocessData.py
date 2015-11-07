# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities
import logging
from collections import defaultdict
import csv
import string


#set logging level
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#proportion of data for training
number_of_neighbors = 9
label1 = 1
label2 = 2
training_data = []
testing_data = []
training_data_label = []


dictionary = corpora.Dictionary("")

def main():
    getData()   
    global dictionary
    dictionary = corpora.Dictionary(line.lower().split() for line in training_data)
    trainModel()
   

def getData():
    exclude = set(string.punctuation)
    with open('D:/NLP/data/all_shuffle.csv', 'rb') as csvfile:
      spamreader = csv.reader(csvfile, delimiter=',')
      for row in spamreader:
           #eliminate all punctuation
           s = ''.join(ch for ch in row[0] if ch not in exclude)
           training_data.append(s)
           if row[1]=='1' or row[1]=='2' or row[1]=='3':
                training_data_label.append(label1)
           else:
                training_data_label.append(label2)
                
    with open('D:/NLP/data/all.txt') as f:
        lines = f.readlines()
        for line in lines:
            s = ''.join(ch for ch in line if ch not in exclude)
            testing_data.append(s)


def trainModel():
   # remove stop words and words that in stoplist appear only once
   stoplist = set('for a of the and to in'.split())
   stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
   once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
   dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
   dictionary.compactify() # remove gaps in id sequence after words that were removed

   #remove words appear in stoplist
   #use proportion of data to train the model
   texts = [[word for word in document.lower().split() if word not in stoplist]for document in testing_data]
        
   # remove words that appear only once
   frequency = defaultdict(int)
   for text in texts:
      for token in text:
         frequency[token] += 1
   texts = [[token for token in text if frequency[token] > 1]for text in texts]

   #get first model by using corpus
   corpus = [dictionary.doc2bow(text) for text in texts]
   tfidf = models.TfidfModel(corpus)

   #apply a transformation on the while corpus
   corpus_tfidf = tfidf[corpus]

   #all togethe there are 2 topics, Digital apple and Physical Apple
   lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

   #for each document, it will have some relevancy to the fist topics and relevancy to the second topic
   
   preProcessData(lsi, corpus, number_of_neighbors)
   
   
def preProcessData(lsi, corpus, k): 
   with open('D:/NLP/data/allRelevant.txt', 'a') as outf:
     for doc in testing_data:
      
        vec_bow = dictionary.doc2bow(doc.lower().split())
        vec_lsi = lsi[vec_bow] # convert the query to LSI space
        index = similarities.MatrixSimilarity(lsi[corpus])
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        label1_vote = 0
        label2_vote = 0
        count = 0
        for sim in sims:
            if sim[0] < len(training_data_label):
               count = count + 1
               if training_data_label[sim[0]] == label1:
                    label1_vote = label1_vote +1 
               else:
                   label2_vote = label2_vote +1
            if count == k: 
                break
             
        if label1_vote >= k-3:
            outf.write(doc)
        
main()
