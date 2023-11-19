# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 01:28:35 2023

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

import pattern.en
from pattern.en import comparative, superlative

from wordnet import *

# pip install inflect
# pip install pattern3


p = "data/"
fname = "ChatGPT_simple_sents.csv"


# ----------------------------------------

# LOAD SENTENCES:

# Load all sentences:
sents = load_data(p, fname)  

# sents[0] # 'The fluffy cat napped lazily in the warm sunlight.'

# ----------------------------------------

# TESTING WITH ALL DATA:    
    
# Tokenized sentences, with analysis of nouns and adjectives in each sentences:
results = [add_relations(sent) for sent in sents] # len 300

# Let's look at results for first sentence:
results1 = results[0]

# It contains:
    
# a) Tokenized sentence:
tokenized_sent = results1[0]
print(tokenized_sent) # ['The', 'fluffy', 'cat', 'napped', 'lazily', 'in', 'the', 'warm', 'sunlight', '.']


# b) Info for each noun and adjective, incl:
# -- hyponyms and hypernyms,
# -- position of the word in tokenized sentence,
# -- info needed for inserting the hyponym or hyperonym back into the sentence 
# (e.g. 'word_form' is needed for converting the lemma to correct word form 
# (pluaral for nouns, or comparative or superlative for adjectives), if the 
# original word was in that form).

words = results1[1]
for key, value in words[1].items():
    print(key, ':', str(value))
# word : cat 
# word_type : noun 
# start_pos : 2 
# lemma : cat 
# word_form : singular 
# hyper : ['feline', 'gossip', 'woman'] 
# hypo : ['saber-toothed tiger', 'sabertooth', 'wildcat', 'liger', 'snow leopard', 'ounce', 'panthera uncia', 'lion', 'king of beasts', 'panthera leo', 'cheetah', 'chetah', 'acinonyx jubatus', 'tiglon', 'tigon', 'tiger', 'panthera tigris', 'jaguar', 'panther', 'panthera onca', 'felis onca', 'domestic cat', 'house cat', 'felis domesticus', 'felis catus', 'sod', 'leopard', 'panthera pardus'] 

# -------------------------------

# GETTING NEW SENTENCE:
    
# Getting new sentence, where original word is replaced with hyponym/hypernym:
new_sent = replace_word(tokenized_sent, words[1], words[1]['hyper'][0])
# 'The fluffy feline napped lazily in the warm sunlight .'


# ============================================

# STATISTICS:
    
# how many hyponyms and hypernyms, nouns and adjectives?
noun_hypo, noun_hyper = 0, 0
adj_hypo, adj_hyper = 0, 0
tot_nouns, tot_adj = 0,0

for i, s in enumerate(results):
    for word in s[1]:
        if word["word_type"]=="noun":
            tot_nouns+=1
            noun_hypo+= len(word['hypo'])
            noun_hyper+= len(word['hyper'])
        else:
            tot_adj+=1
            adj_hypo+= len(word['hypo'])
            adj_hyper+= len(word['hyper'])   

noun_hypo_avg = str(round(noun_hypo/tot_nouns, 1))
noun_hyper_avg = str(round(noun_hyper/tot_nouns, 1))
adj_hypo_avg = str(round(adj_hypo/tot_adj, 1))
adj_hyper_avg = str(round(adj_hyper/tot_adj, 1))


print('Nouns (', tot_nouns, ') had', noun_hyper, 'hypernyms (on avg.',  noun_hyper_avg, 'per noun) and', noun_hypo, 'hyponyms (on avg.', noun_hypo_avg, 'per noun).')
# Nouns ( 1134 ) had 2827 hypernyms (on avg. 2.5 per noun) and 35970 hyponyms (on avg. 31.7 per noun).

print('Adjectives (', tot_adj, ') had', adj_hyper, 'hypernyms (on avg.',  adj_hyper_avg, 'per adjective) and', adj_hypo, 'hyponyms (on avg.', adj_hypo_avg, 'per adjective).')
# Adjectives ( 438 ) had 0 hypernyms (on avg. 0.0 per adjective) and 0 hyponyms (on avg. 0.0 per adjective).


# 35970*31.7 # 1.1 million estimated hyponyms at next level
    
