# -*- coding:utf-8 -*-
# training the doc2vec model 
import gensim.models as g
import logging

vector_size = 300
window_size = 5
min_count = 5
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 1 #number of parallel processes

train_corpus = "ebd_train.txt"
saved_path = "model.bin"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


docs = g.doc2vec.TaggedLineDocument(train_corpus)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,  iter=train_epoch)


model.save(saved_path)
