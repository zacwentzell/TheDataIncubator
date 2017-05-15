
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
from gensim.models.word2vec import Word2Vec
from string import punctuation
from toolz import curry


# In[5]:

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


# In[6]:

# import pickle
# with open('./cleaned_reviews.pickle', 'rb') as infile:
#     cleaned = pickle.load(infile)


# In[7]:

# import pickle
# c_splits = []
# with open('./cleaned_split_1.pickle', 'rb') as infile:
#     c_splits.extend(pickle.load(infile))

# with open('./cleaned_split_2.pickle', 'rb') as infile:
#     c_splits.extend(pickle.load(infile))


# In[8]:


# In[9]:

class ReviewSplitter(object):
    def __init__(self, reviews):
        self.reviews = reviews
        
    def __iter__(self):
        for review in self.reviews:
            # r = preprocess(review)
#             pdb.set_trace()
            yield review.split()


# In[ ]:
with open('./cleaned_reviews.pickle', 'rb') as infile:
    cleaned = pickle.load(infile)


# In[ ]:

wv = Word2Vec(ReviewSplitter(cleaned), size=200, min_count=500, workers=16)


# In[ ]:

wv.save('./w2v_reviews_v2.model')


# In[ ]:

pool.terminate()


# In[10]:

preprocess('this is a test')


# In[ ]:



