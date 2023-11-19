# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 01:57:47 2023

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

# pip install inflect
# pip install pattern3


## Wordnet notebooks:
# https://www.kaggle.com/code/roblexnana/nlp-with-nltk-tokenizing-text-and-wordnet-basics
# https://www.kaggle.com/code/sharanharsoor/wordnet-relationships-semantic-processing#Happy-Learning


inflect_eng = inflect.engine()

# ===============================================


# LOAD DATA:
    
def load_data(folder, fname):
    with open(folder + fname, encoding = "utf-8") as f:
        sents = f.readlines()
    sents = [sent.strip('\n') for sent in sents]
    sents = [sent.strip('\ufeff') for sent in sents] 
    sents = [sent.strip('"') for sent in sents]
    return sents

# ------------------------------------------

# TOKENIZE, LEMMATIZE AND IDENTIFY ADJECTIVES AND NOUNS:

def tagset_map(tag, use_default):
    tag = re.sub('^N[A-Z]{1,3}$', 'n', tag) # noun
    tag = re.sub('^J[A-Z]{1,2}$', 'a', tag) # adjective
    tag = re.sub('^R[A-Z]{1,2}$', 'r', tag) # adverb
    tag = re.sub('^V[A-Z]{1,2}$', 'v', tag) # verb
    if tag not in list('narv'):
        if use_default:
            tag = 'n'
        else:
            tag = 'x'
    return tag


def prep_sent(sent):
    
    # Tokenize, lemmatize and add postags:
    tokenized_sent = nltk.word_tokenize(sent) #  # ['The', 'fluffy', ... ]
    tokenized_sent2 = [word.lower() for word in tokenized_sent]
    tagged_sent = pos_tag(tokenized_sent2) # [[('The', 'DT'), ('fluffy', 'JJ'), ('cat', 'NN'), ('napped', 'VBD'), ... ], ... ]
    lemmatizer = WordNetLemmatizer()
    lemmatized_sent = [lemmatizer.lemmatize(word, tagset_map(tag, True)) for (word,tag) in tagged_sent] # ['The', 'fluffy', 'cat', 'nap', 'lazily', ... ]
    
    # Check:
    assert(len(tokenized_sent2) == len(tagged_sent))
    assert(len(lemmatized_sent) == len(tagged_sent))
    
    # Gather adjectives, nouns, their ids and lemmas:
    analysis = []
    
    for i, token in enumerate(tagged_sent):
        tag = tagset_map(token[1], False)
        if token[1] == "NNP": # personal name - should not be included
            tag = "x"
        if tag == "n" or tag == "a": 
            item = {
                    "word": token[0], 
                    "word_type": "noun",
                    #"tag": token[1], 
                    "start_pos": i, 
                    "lemma": lemmatized_sent[i], 
                    "word_form": ""}
            if tag == "a":
                item["word_type"] = "adj"
                if token[1] == "JJS":
                    item["word_form"] = "superlative" # eg "deepest"
                elif token[1] == "JJR":
                    item["word_form"] = "comparative" # eg "deeper"
                else: # "JJ"
                    item["word_form"] = "normal" # eg "deep"
            else:  # tag == "n"
                if token[1] == "NNS": 
                    item["word_form"] = "plural"
                else: # "NN" - singular
                    item["word_form"] = "singular"
                    
            analysis.append(item)

    return [tokenized_sent, analysis]
                
            
# ------------------------------------------

# REPLACING NOUN OR ADJECTIVE WITH ANOTHER ONE IN THE SENTENCE:
    
# - If noun was in plural, or an adjective was in superlative or comparative form (eg 'worse' or 'worst'), then also replacement lemma is converted into that form. 
# - If it's first word in sentence, then it is capitalized.
# - Tokenization is reversed.
# - New "word" can also consist of several words. 

def replace_word(tokenized_sent, word, new_word):
    
    # sample inputs:
        
    # tokenized_sent = ['The', 'excited', 'children', 'eagerly', 'awaited', 'the', 'colorful', 'fireworks', '.']
    # word = {'word': 'children', 'word_type': 'noun', 'start_pos': 2, 'lemma': 'child', 'word_form': 'plural'}
    # new_word = "school kid" 

    # ----------------------
    
    # If noun was in plural, convert new noun also to plural; 
    if word["word_form"] == "plural":
        new_word = inflect_eng.plural(new_word) # 'school kid' -> 'school kids'
    
    # If adj. was in comparative or superlative, convert new adj. also to this form:    
    elif word["word_form"] == "comparative":
        new_word = comparative(new_word) # "deeper"
    elif word["word_form"] == "superlative":
        new_word = superlative(new_word) # "deepest"

    # ----------------------
    
    # Capitalize if it's 1st word in sentence:
    word_idx = word["start_pos"]
    if word_idx == 0:
        new_word = new_word[0].upper() + new_word[1:]
    
    # ----------------------
    
    # New sentence:
    word_len = len(word["word"].split()) # number of words in the original "word" (eg could be "school kids" or "african elephant")
    
    tokenized_sent_new = tokenized_sent[:word_idx] + new_word.split() + tokenized_sent[word_idx + word_len:] 
    
    new_sent = " ".join(tokenized_sent_new)  # 'The excited school kids eagerly awaited the colorful fireworks .'
    
    return new_sent


# ------------------------------------------


# FIND RELATED TERMS:
    
# a) hyperonyms (= parent terms) --> more abstract
# b) hyponyms (=child terms) --> more specific

# - Since a word may have various meanings and thus belong to multiple
# synsets, it may also have multiple hyernyms (one for each meaning). 
    
def get_related_terms2(word, word_type):
    orig = {"lemma": word, "word_type": word_type} 
    return get_related_terms(orig) # word_type


def get_related_terms(orig_word): 
    
    word = orig_word["lemma"]
    word_type = orig_word["word_type"]
    
    # wordnet representation of compound words:
    word_ = word.replace(" ", "_") # "cookery book" --> 'cookery_book'
    
    # ---------------------------------
        
    # # Identify synsets of the word:
    if word_type == "noun": # only use synsets corresponding to nouns
        all_syns = wn.synsets(word_, pos = wn.NOUN) 
    elif word_type == "adj": # only use synsets of adjectives
        all_syns = wn.synsets(word_, pos = wn.ADJ) # also: wn.ADJ, wn.VERB
    # all_syns #  [Synset('tractor.n.01'), Synset('tractor.n.02')]
    
    # ---------------------------------
    
    # Identify related synsets and lemmas in them:
        
    # a) Hyponyms:
        
    hyponymes = [x for syn in all_syns for x in syn.hyponyms()]
    hyponymes = list(set(hyponymes))    
    hyponyme_lemmas = [x.replace("_", " ").lower() for h in hyponymes for x in h.lemma_names()] 
    
    
    # b) Hyperonyms:
        
    # -For hypernyms, in order to limit irrelevant hypernyms of synsets with
    # different meaning than the word itself, we gather hypernyms only for synsets
    # where the word itself is also the first lemma in the synset. (Because first
    # lemma is the most commonly used lemma with that meaning). And then also for
    # each hypernym synset we only use the first lemma in its synset:
        
    hypernyme_lemmas = []
    
    for i in range(len(all_syns)):
        syn = all_syns[i]
        if syn.lemma_names()[0] == word:
            hypernymes = [x for x in syn.hypernyms()]
            hyp_lemmas = [hypernymes[i].lemma_names()[0] for i in range(len(hypernymes))]
            hypernyme_lemmas+= [x.replace("_", " ").lower() for x in hyp_lemmas] 
            
    hypernyme_lemmas=list(set(hypernyme_lemmas))

    # ---------------------------------
    
    # Return as dict:
    result = {"hyper": hypernyme_lemmas, # parent terms
              "hypo": hyponyme_lemmas}  # child terms

    return result


# ========================================

def add_relations(sent):
    
    # sent = "The fluffy cat napped lazily in the warm sunlight."
    
    tokenized_sent, orig_words =  prep_sent(sent)    
    
    for i, orig_word in enumerate(orig_words):
        # if i ==1:
        #     break
    
        related_terms = get_related_terms(orig_word)  # dict
        # related_terms.keys() # ['hyper', 'hypo']
        
        orig_words[i]['hyper'] = related_terms['hyper']
        orig_words[i]['hypo'] = related_terms['hypo']        
    
    return [tokenized_sent, orig_words]

        
