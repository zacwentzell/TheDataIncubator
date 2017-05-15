
# coding: utf-8

# In[4]:

import gensim
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import spacy
import textacy

from IPython.core.debugger import Pdb
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from string import punctuation
from toolz import curry


# In[5]:
nlp = spacy.load('en_core_web_md')

pool = mp.Pool(processes=mp.cpu_count())

pdb = Pdb()

ps = punctuation.replace('\'','')


# In[3]:

# nlp = spacy.load('en_core_web_md')


# In[6]:

textacy_preprocess = curry(textacy.preprocess_text)(fix_unicode=True, lowercase=True, no_contractions=True, no_punct=True)

def preprocess(review):
    text = review
    
    for punc in ps:
        text = text.replace(punc, punc + ' ')
    return text


class ReviewSplitter(object):
    def __init__(self, reviews):
        self.reviews = reviews
        
    def __iter__(self):
        for review in self.reviews:
            # r = preprocess(review)
#             pdb.set_trace()
            yield review.split()


df = pd.read_csv('./Electronics_5.csv')
text = df.reviewText[df.reviewText.notnull()].values
del df

docs = nlp.pipe(text, n_threads=8, batch_size=100, entity=False)

sents = []
for doc in docs:
    sents.extend([preprocess(s.text) for s in doc.sents])

sents = pool.map(textacy_preprocess, sents)
pool.close()
pool.join()

with open('sents_preprocessed.pickle', 'wb') as outfile:
    outfile.write(sents, outfile)


class LabeledSentences(object):
    def __init__(self, sents):
        self.sents = sents
    def __iter__(self):
        for uid, sent in enumerate(sents):
            yield LabeledSentence(words=sent.split(), labels=['SENT_%s' % uid])


dv = Doc2Vec(LabeledSentences(sents), size=200, min_count=500, workers=16)


# In[ ]:

dv.save('./d2v_reviews.model')


# In[ ]:

pool.terminate()


# In[10]:

preprocess('this is a test')


# In[ ]:



