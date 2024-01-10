# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 21:04:37 2024

@author: renat
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 15:08:35 2024

@author: Renata Siimon
"""

import pandas as pd
import numpy as np
import nltk
#from nltk.tokenize import word_tokenize
from nltk import word_tokenize, sent_tokenize, pos_tag, pos_tag_sents
# from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer 
# from string import punctuation
import re
import inflect
import ast

import pattern.en
from pattern.en import comparative, superlative

from wordnet import *

# pip install inflect
# pip install pattern3


p = "data/"
fname = "ChatGPT_simple_sents2.csv"
fname_rev = "df_rev2_63.csv"

# ========================================================

# LOAD REVIEWED HYPONYMS/HYPERONYMS (WITH SENTENCES, LEMMAS, ETC)

# Did not save word form (plural/singular) and tokenized sent before reviewing...
# --> make the original df again to retrieve them:
sents = load_data(p, fname)  
results2 = add_relations2(sents, 0)

# Load reviewed file:
df_rev = pd.read_csv(p + fname_rev, encoding ="utf-8", sep = ",", index_col = 0)    # sep, , header = 'infer'
df_rev = df_rev.rename(columns = {'sent': 'sent2', 'lemma': 'lemma2'})

# Join:
df = results2.join(df_rev, how='inner')

# np.sum(df['lemma'] == df['lemma2']) == len(df) # True
# np.sum(df['sent'] == df['sent2']) == len(df) # True


# Reindex:
df.index = [x for x in range(len(df))]

# Add sentence index:
sents = df['sent'].tolist()
sent_idx = [0]
prev = sents[0]
idx = 0
for i in range(1, len(sents)):
    current = sents[i]
    if prev != current:
        idx+=1
        prev = current
    sent_idx.append(idx)

df['sent_id'] = sent_idx # 64 sents


# Rearrange / format columns:
df = df.drop(columns = ['sent2', 'lemma2'])
df = df.drop(columns = ['hypo1', 'hypo2', 'hypo3', 'hyper1', 'hyper2', 'hyper3'])
# df = df[['sent_id', 'sent', 'word', 'lemma', 'form', 'start', 'hyper', 'hypo', 'tokenized_sent']]
df = df[['sent_id', 'sent', 'lemma', 'hyper', 'hypo', 'word', 'form', 'start', 'tokenized_sent']]

df['hyper'] = df['hyper'].apply(lambda s: list(ast.literal_eval(s)))
df['hypo'] = df['hypo'].apply(lambda s: list(ast.literal_eval(s)))
df['tokenized_sent'] = df['tokenized_sent'].apply(lambda s: list(ast.literal_eval(s)))
# type(df['hyper'][0]) # list

# ============================================

# SAVE:

df.to_csv(p + "df_final.csv", encoding = "utf-8")

# -----

## LOAD :
## - also: convert cols containing lists as strings to lists
# df = pd.read_csv(p + "df_final.csv", encoding = "utf-8")
# df['hyper'] = df['hyper'].apply(lambda s: list(ast.literal_eval(s)))
# df['hypo'] = df['hypo'].apply(lambda s: list(ast.literal_eval(s)))
# df['tokenized_sent'] = df['tokenized_sent'].apply(lambda s: list(ast.literal_eval(s)))

# ============================================

# STATISTICS:

def print_stats(df):
    # how many sentences, target words?
    num_sents = np.max(df['sent_id']) + 1 # 64
    num_targets = len(df) # 123
    targets_per_sent = num_targets/num_sents # 1.9
    
    # how many hyponyms and hypernyms?
    hypos = df['hypo'].tolist()
    hypers = df['hyper'].tolist()
    num_hypos, num_hypers = 0, 0
    
    for i in range(len(df)):
        num_hypos+= len(hypos[i]) # 1064
        num_hypers+= len(hypers[i]) # 206
    hypos_per_target = num_hypos/num_targets # 8.65
    hypers_per_target = num_hypers/num_targets # 1.67
    
    print('sentences:', num_sents)
    print('target words:', num_targets, '(avg.', round(targets_per_sent, 2), 'per sentence)')
    print('hypernyms:', num_hypers, '(avg.', round(hypers_per_target, 2), 'per target word)')
    print('hyponyms:', num_hypos, '(avg.', round(hypos_per_target, 2), 'per target word)')


print_stats(df)

# sentences: 64
# target words: 123 (avg. 1.92 per sentence)
# hypernyms: 206 (avg. 1.67 per target word)
# hyponyms: 1064 (avg. 8.65 per target word)

# Note: Each target word has at least one hyponyme and one hypernyme.

# ========================================================

# GETTING NEW SENTENCE:

row = df.loc[7] # row 8
    
tokenized_sent = row['tokenized_sent']
print(tokenized_sent) # ['The', 'delicious', 'aroma', 'of', 'freshly', 'baked', 'cookies', 'filled', 'the', 'cozy', 'kitchen', '.']
    
# Getting new sentence, where original word is replaced with hyponym/hypernym:
new_sent = replace_word(tokenized_sent, row, 'ship galley')
# 'The delicious aroma of freshly baked cookies filled the cozy ship galley .'










    