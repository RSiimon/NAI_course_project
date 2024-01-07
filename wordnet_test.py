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
fname = "ChatGPT_simple_sents2.csv"

# ----------------------------------------

# NOTE:
# This file was used to compile a table of sentences with wordnet outputs (hyponyms and hyperonyms). Different columns in the output file (df.csv) correspond to different levels of filtering in wordnet.

# This output file was then manually reviewed (currently 63 sentences out of 300 have been reviewed). The review consisted in choosing the final hyponyms and hyperonyms to be used. They were chosen so that replacing the target word with its  hyponym or hyperonym doesnt make the sentence weird. In many cases, the hyponyms/hyperonyms from wordnet had somewhat different meaning and did not fit in the sentence - those were left out. On the other hand, in some cases wordnet did not offer any suitable hyperonyms or hyponyms. In such cases, if possible, hyponym(s) / hyperonym(es) were added manually, or otherwise the target word was left out. (If all target words in a sentence were left out, then the whole sentence was of course also left out). 
# With hyponyms, the reviewing mainly consisted in leaving out hyponyms that were extremenly specific (latin names of species or rarely used colloquialisms), and hyponyms that did not fit in the sentence. Hyponyms from wordnet were rarely missing. 
# With hyperonyms, the output of wordnet contained fewer terms, and in some cases did not offer any suitable ones. Still, in the majority of cases, it did offer some suitabl ones. So the main task here was to add suitable hyponyms manually if none from wordnet fitted the sentence.  


# The input file (300 sentences generated with ChatGPT): "ChatGPT_simple_sents2.csv"
# The output file: "df_rev1.csv"
# The manually reviewed output file is: 'df_rev2_63.csv'
# 

# ----------------------------------------

# LOAD SENTENCES:

# Load all sentences:
sents = load_data(p, fname)  

# sents[0] # 'The fluffy cat napped lazily in the warm sunlight.'

# ----------------------------------------

# TESTING WITH ALL DATA:    
    
# Tokenized sentences, with analysis of nouns and adjectives in each sentences:
# results = [add_relations(sent) for sent in sents] # len 300
results2 = add_relations2(sents, 0) 
results3 = add_relations2(sents, 1) 
results4 = add_relations2(sents, 2) 

df = pd.DataFrame()
df['sent'] = results2['sent']
df['lemma'] = results2['lemma']
# df['form'] = results2['form']
df['hypo1A']= results2['hypo1']
df['hypo1B']= results3['hypo1']
df['hypo1C']= results4['hypo1']
df['hypo2A']= results2['hypo2']
df['hypo2B']= results3['hypo2']
df['hypo2C']= results4['hypo2']
df['hypo3A']= results2['hypo3']
df['hypo3B']= results3['hypo3']
df['hypo3C']= results4['hypo3']
   
df['hyper1A']= results2['hyper1']
df['hyper1B']= results3['hyper1']   
df['hyper1C']= results4['hyper1']   
df['hyper2A']= results2['hyper2']
df['hyper2B']= results3['hyper2']
df['hyper2C']= results4['hyper2']
df['hyper3A']= results2['hyper3']
df['hyper3B']= results3['hyper3']
df['hyper3C']= results4['hyper3']

   
df.to_csv(p + "df_rev1.csv", encoding = "utf-8")



# # ========================================================

# # LOOKING AT OUTPUT:

# # Let's look at results for first sentence:
# row = results2.loc[8]

# for key, value in row.items():
#     print(key, ':', value)

# # sent : The delicious aroma of freshly baked cookies filled the cozy kitchen.
# # word : kitchen
# # lemma : kitchen
# # form : singular
# # start : 10
# # hypo1 : ['kitchenette', 'caboose', 'galley', 'cookhouse', "ship's galley"]
# # hypo2 : ['cuddy', 'trireme']
# # hypo3 : []
# # hyper1 : ['room']
# # hyper2 : ['chance', 'opportunity', 'gathering', 'assemblage', 'position', 'area', 'spatial relation']
# # hyper3 : ['possibleness', 'peril', 'occupation', 'body part', 'role', 'job', 'social group', 'line', 'orientation', 'structure', 'risk', 'extent', 'bodily property', 'construction', 'business', 'issue', 'relation', 'assumption', 'mental attitude', 'attitude', 'subject', 'matter', 'region', 'danger', 'point', 'topic', 'possibility', 'line of work']
# # tokenized_sent : ['The', 'delicious', 'aroma', 'of', 'freshly', 'baked', 'cookies', 'filled', 'the', 'cozy', 'kitchen', '.']


# # -------------------------------

# # GETTING NEW SENTENCE:
    
# tokenized_sent = row['tokenized_sent']
# print(tokenized_sent) # ['The', 'delicious', 'aroma', 'of', 'freshly', 'baked', 'cookies', 'filled', 'the', 'cozy', 'kitchen', '.']
    
# # Getting new sentence, where original word is replaced with hyponym/hypernym:
# new_sent = replace_word(tokenized_sent, row, 'ship galley')
# # 'The delicious aroma of freshly baked cookies filled the cozy ship galley .'



