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


# TODO: add to report: for nouns, only synsets of nouns, and for adjectives, only synsets of adjectives were considered.

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
    
    # Gather nouns, their ids and lemmas:
    analysis = []
    
    for i, token in enumerate(tagged_sent):
        tag = tagset_map(token[1], False)
        if token[1] == "NNP": # personal name - should not be included
            tag = "x"
        if tag == "n": # or tag == "a": 
            item = {
                    "word": token[0], 
                    "word_type": "noun",
                    #"tag": token[1], 
                    "start_pos": i, 
                    "lemma": lemmatized_sent[i], 
                    "word_form": ""}
            # if tag == "a":
            #     item["word_type"] = "adj"
            #     if token[1] == "JJS":
            #         item["word_form"] = "superlative" # eg "deepest"
            #     elif token[1] == "JJR":
            #         item["word_form"] = "comparative" # eg "deeper"
            #     else: # "JJ"
            #         item["word_form"] = "normal" # eg "deep"
            # else:  # tag == "n"
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

def replace_word(tokenized_sent, row, new_word):
    
    # sample inputs:
        
    # tokenized_sent = ['The', 'excited', 'children', 'eagerly', 'awaited', 'the', 'colorful', 'fireworks', '.']
    # word = {'word': 'children', 'word_type': 'noun', 'start_pos': 2, 'lemma': 'child', 'word_form': 'plural'}
    # new_word = "school kid" 

    # ----------------------
    
    # If noun was in plural, convert new noun also to plural; 
    if row["form"] == "plural":
        new_word = inflect_eng.plural(new_word) # 'school kid' -> 'school kids'
    
    # # If adj. was in comparative or superlative, convert new adj. also to this form:    
    # elif word["word_form"] == "comparative":
    #     new_word = comparative(new_word) # "deeper"
    # elif word["word_form"] == "superlative":
    #     new_word = superlative(new_word) # "deepest"

    # ----------------------
    
    # Capitalize if it's 1st word in sentence:
    word_idx = row["start"]
    if word_idx == 0:
        new_word = new_word[0].upper() + new_word[1:]
    
    # ----------------------
    
    # New sentence:
    word_len = len(row["word"].split()) # number of words in the original "word" (eg could be "school kids" or "african elephant")
    
    tokenized_sent_new = tokenized_sent[:word_idx] + new_word.split() + tokenized_sent[word_idx + word_len:] 
    
    new_sent = " ".join(tokenized_sent_new)  # 'The excited school kids eagerly awaited the colorful fireworks .'
    
    return new_sent


# ------------------------------------------


# FIND RELATED TERMS:
    
# a) hyperonyms (= parent terms) --> more abstract
# b) hyponyms (=child terms) --> more specific

# - Since a word may have various meanings and thus belong to multiple
# synsets, it may also have multiple hyernyms (one for each meaning). 
    
def get_related_terms(word, hypos, hypers, filtering): 
    
    # wordnet representation of compound words:
    word_ = word.replace(" ", "_") # "cookery book" --> 'cookery_book'
    # word_ = 'cat'
    # word_ = 'apple'
    
    # ---------------------------------
        
    # # Identify synsets of the word:
    all_syns = wn.synsets(word_, pos = wn.NOUN)  # also: wn.ADJ, wn.VERB
    # all_syns #  [Synset('tractor.n.01'), Synset('tractor.n.02')]
    
    # Remove synsets which name does not contain the word or vice versa (usually they have different meaning):
    # all_syns = [syn fo1r syn in all_syns if syn.lemma_names()[0] in word_ or word_ in syn.lemma_names()[0]]
    
    # In order to avoid synsets with different meaning than the word itself, we
    # only include synsets where the word itself is also the first lemma in the
    # synset. (Because first lemma is the most commonly used lemma with that
    # meaning). --> Remove synsets which first lemma is not _word:
    all_syns = [syn for syn in all_syns if syn.lemma_names()[0] == word_]
    
    # FILTERING 1: to only include synset with smallest number:
    # (Eg if all_syns = [Synset('cat.n.01'), Synset('cat.n.03')], then only include
    # Synset('cat.n.01')):
    if filtering in (1, 2):
        if len(all_syns) > 1:
            min_nr = np.argmin([int(syn.name()[-1]) for syn in all_syns])
            all_syns = [all_syns[min_nr]]
        
    
    # ---------------------------------
    
    # Identify related synsets and lemmas in them:
    
    if hypers == False:  
        hyponyme_lemmas = get_hyponymes(all_syns)  # child terms
        return hyponyme_lemmas
    elif hypos == False:
        hypernyme_lemmas = get_hypernymes(all_syns, filtering) # parent terms
        return hypernyme_lemmas

    else:
        hyponyme_lemmas = get_hyponymes(all_syns)        
        hypernyme_lemmas = get_hypernymes(all_syns, filtering)
       
        return [hyponyme_lemmas, hypernyme_lemmas]



def get_hyponymes(all_syns2):  
    hyponyme_lemmas = []
    
    for syn in all_syns2:
        hyponymes = [x for x in syn.hyponyms()]
        for h in hyponymes:
            for x in h.lemma_names():
                # print(x)
                if isinstance(x, str):
                    hyponyme_lemmas.append(x.replace("_", " ").lower())
                else:
                    for x2 in x:
                        hyponyme_lemmas.append(x2.replace("_", " ").lower())
    
    hyponyme_lemmas=list(set(hyponyme_lemmas))
    return hyponyme_lemmas


def get_hypernymes(all_syns2, filtering):    
    hypernyme_lemmas = []
    
    for syn in all_syns2:
        # if syn.lemma_names()[0] == word: 
        hypernymes = [x for x in syn.hypernyms()]
        for h in hypernymes:
            for x in h.lemma_names():
                hypernyme_lemmas.append(x.replace("_", " ").lower())
                # FILTERING 2: break to include only first lemma:
                if filtering == 2:
                    break  
                    
    hypernyme_lemmas=list(set(hypernyme_lemmas))  
    return hypernyme_lemmas


# ========================================


def add_relations2(sents, filtering):
    # filtering: 0- none, 1 - only include synset with smallest number,
    # 2- include only first lemma for hypernyms
    
    sents2 = [] # duplicate original sent if it contains multiple nouns
    tokenized_sents = []
    words, lemmas, forms = [], [], []
    starts = [] # index of noun in tokenized sentence
    
    # lemmas of hyponyms and hypernyms:
    hypos_1, hypos_2, hypos_3 = [], [], []
    hypers_1, hypers_2, hypers_3 = [], [], []
    
    for sent in sents:
        # sent = "The fluffy cat napped lazily in the warm sunlight."
        
        tokenized_sent, orig_words =  prep_sent(sent)    
                
        for i, orig_word in enumerate(orig_words):
            # if i ==1:
            #     break
        
            related_terms = get_related_terms(orig_word["lemma"], True, True, filtering)  
            
            # First level hyponyms and hypernyms:
            hypos, hypers = related_terms[0], related_terms[1]

            # Only add if there is at least one hyponym or hypernym:
            if len(hypos) + len(hypers) > 0:
                
                # 2nd level hyponyms and hypernyms:
                hypos_2_tmp, hypers_2_tmp = [], []
                for hypo in hypos:
                    hypos_2_tmp+=get_related_terms(hypo, True, False, filtering) 
                for hyper in hypers:
                    hypers_2_tmp+=get_related_terms(hyper, False, True, filtering)
                hypos_2_tmp = list(set(hypos_2_tmp))
                hypers_2_tmp = list(set(hypers_2_tmp))
        
                # 3rd level hyponyms and hypernyms:            
                hypos_3_tmp, hypers_3_tmp = [], []
                for hypo in hypos_2_tmp:
                    hypos_3_tmp+=get_related_terms(hypo, True, False, filtering) 
                for hyper in hypers_2_tmp:
                    hypers_3_tmp+=get_related_terms(hyper, False, True, filtering)
    
                # Add:
                sents2.append(sent)
                tokenized_sents.append(tokenized_sent)
                words.append(orig_word['word'])
                lemmas.append(orig_word['lemma'])
                forms.append(orig_word['word_form']) # plural or singular
                starts.append(orig_word['start_pos'])   
                hypos_1.append(hypos)
                hypers_1.append(hypers)             
                hypos_2.append(hypos_2_tmp)
                hypers_2.append(hypers_2_tmp)
                hypos_3.append(list(set(hypos_3_tmp)))
                hypers_3.append(list(set(hypers_3_tmp)))
            
    
    df = pd.DataFrame(data = [sents2, words, lemmas, forms, starts, 
                              hypos_1, hypos_2, hypos_3,
                              hypers_1, hypers_2, hypers_3,
                              tokenized_sents]).T 

    df.columns = ['sent', 'word', 'lemma', 'form', 'start', 'hypo1', 'hypo2', 'hypo3', 'hyper1', 'hyper2', 'hyper3', 'tokenized_sent']
    
    return df       
        
    # return [tokenized_sent, orig_words]



# ------------------------------------------------
# # Could not use BLEU score (it doesnt  take meanings into account):
# hypothesis = ['It', 'is', 'a', 'cat', 'at', 'room']
# reference = ['It', 'is', 'a', 'cat', 'inside', 'the', 'room']
# hypothesis = ['It', 'is', 'indeed', 'awesome']
# reference = ['It', 'is', 'indeed', 'great']
# reference = ['It', 'is', 'indeed', 'horrible']
# #there may be several references
# BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
# print(BLEUscore)


