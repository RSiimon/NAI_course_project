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
import re
from copy import deepcopy

import seaborn as sns
import torch

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import DistilBertTokenizer, DistilBertModel

from bertviz import head_view
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine

# pip install inflect
# pip install pattern3


p = "data/"
# fname = "ChatGPT_simple_sents2.csv"
# fname_rev = "df_rev2_63.csv"
fname = "df_final.csv"
p2 = "pca_plots/"


# =====================================================

# LOAD REVIEWED DF:

def load_reviewed(p, fname):
    # - also: convert cols containing lists as strings to lists
    df = pd.read_csv(p + fname, encoding = "utf-8")
    df['hyper'] = df['hyper'].apply(lambda s: list(ast.literal_eval(s)))
    df['hypo'] = df['hypo'].apply(lambda s: list(ast.literal_eval(s)))
    df['tokenized_sent'] = df['tokenized_sent'].apply(lambda s: list(ast.literal_eval(s)))
    return df

# =====================================================

# TOKENIZING:
    
# Get original tokenized sentence, plus separately its parts (tokenized start, end and target word), and location of target word:
# -- Consider that tokenizer may split single word into multiple tokens, and that some hyponyms/hypernyms consist of several words.
# -- Original target word is always just one word, not a phrase.

def tokenize_orig(row):
    word_orig = row.word # 'trees'
    sent_orig = row.sent # 'The ancient trees whispered in the peaceful forest.'
    tokenized_sent_orig = row['tokenized_sent']
    word_idx = row["start"]
    sent_start = " ".join(tokenized_sent_orig[:word_idx]) # 'The ancient'
    sent_end = " ".join(tokenized_sent_orig[1+word_idx:]) # 'whispered in the  peaceful forest .'
    
    # Tokenized sentence and its parts:
    tokens_start = tokenizer.tokenize("[CLS] " + sent_start) 
    tokens_end = tokenizer.tokenize(sent_end + " [SEP]") 
    tokens_target = tokenizer.tokenize(word_orig) # ['trees']
    tokenized_sent_orig = tokens_start + tokens_target + tokens_end
    # ['[CLS]', 'the', 'ancient', 'trees', 'whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]']    

    # Token ids of target word:
    len_start, len_target = len(tokens_start), len(tokens_target)
    target_range = [len_start, len_start + len_target] # [3, 4]
    
    return [tokenized_sent_orig, target_range, tokens_start, tokens_end, tokens_target]



# Get new tokenized sentence, and tokenized new word with its location:
def tokenize_new(row, is_hypo, new_id, tokens_start, tokens_end):
    # new_id = hypo_id
    # is_hypo = True
    
    # New word:
    new_word = row.hypo[new_id] if is_hypo else row.hyper[new_id]     
    new_word = inflect_eng.plural(new_word) if row["form"] == "plural" else new_word # convert to plural if target word was in plural '
    
    # Tokenized sentence:
    tokens_new = tokenizer.tokenize(new_word) # ['al', '##der', 'trees']    
    tokenized_sent_new = tokens_start + tokens_new + tokens_end
    # ['[CLS]', 'the', 'ancient', 'al', '##der', 'trees', '##s', 'whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]']
    
    # Token ids of new word:
    len_start, len_new = len(tokens_start), len(tokens_new)
    new_range = [len_start, len_start + len_new]
    
    return [tokenized_sent_new, new_range, tokens_new]


# =====================================================

# GET HIDDEN STATES:
    
# - If target or replacement word consisted of several tokens, the embeddings of those tokens are averaged.

def get_embeddings(tokens, tokens_word, tok_range, model, join_toks):

    # tokens = tokenized_sent_new 
    # tokens_word = tokens_new
    # tok_range = new_range
    # tokens = result[0] # result2[0]  
    # tokens_word = result[4] # result2[2]
    # tok_range = result[1] # result2[1]
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
       
    with torch.no_grad():
        outputs = model(tokens_tensor) 
    
    hidden_states = outputs.hidden_states
    hidden_states = torch.stack(hidden_states, dim=0) # list to tensor
    hidden_states = torch.squeeze(hidden_states, dim=1) # remove dim 1
    # hidden_states.size() #torch.Size([7, 13, 768])
    # len(tokens) # 13
    
    # Average the embeddings of target or replacement word if it had several tokens:
    if join_toks and len(tokens_word) != 1:
        
        avg = hidden_states[:,tok_range[0]:tok_range[1],:] # torch.Size([7, 3, 768])
        avg = torch.mean(avg, dim = 1, keepdim=True) # torch.Size([7, 1, 768])
        hidden_states = torch.cat([hidden_states[:,:tok_range[0],:], avg,
                       hidden_states[:,tok_range[1]:,:]], axis = 1)
        # hidden_states.shape # ([7, 11, 768]) #layer, tokens, neurons
    
    return hidden_states


# ========================================================

# TOKENIZE, GET HIDDEN STATES:

def get_results_orig(df, sent_id, model, join_toks): # for targt word
    
    row = df.loc[sent_id]
    # row['lemma'] # 'tree'
    
   # Tokenized sent, its parts and location of target word:
    result = tokenize_orig(row)
    # print(result)
    ## ['[CLS]', 'the', 'ancient', 'trees', 'whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]'],
    ## [3, 4],  <-- location of targeet word in tokenized sentence
    ## ['[CLS]', 'the', 'ancient'], 
    ## ['whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]'],
    ## ['trees']]    
    
    # Hidden states:
    hid_target = get_embeddings(result[0], result[4], result[1], model, join_toks) # torch.Size([7, 11, 768])
    
    return [result, hid_target]


def get_results_new(df, sent_id, is_hypo, new_id, result, model, join_toks): # for hyponyms/hypernyms  
    
    row = df.loc[sent_id]
    
    # New tokenized sentence, tokenized new word with its location:
    result2 = tokenize_new(row, is_hypo, new_id, result[2], result[3])
    # print(result2)
    ## [['[CLS]', 'the', 'ancient', 'al', '##der', 'trees', 'whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]'], 
    ## [3, 6],  <-- location of replacement word in tokenized sentence
    ## ['al', '##der', 'trees']]

    # Hidden states:
    hid_new = get_embeddings(result2[0], result2[2], result2[1], model, join_toks) # torch.Size([7, 11, 768])        
        
    return [result2, hid_new]


def get_results_all(df, sent_id, model, join_toks): # both target and replacement words
    row = df.loc[sent_id]
    
    # TARGET WORD: # 'visitor'
    result = tokenize_orig(row) # tokenized sent, its parts and loc. of target 
    hid_target = get_embeddings(result[0], result[4], result[1], model, join_toks) # torch.Size([7, 11, 768])
    
    # HYPERNYMS: 
    hyper_counts = len(df.hyper[sent_id]) # 2 # ['person', 'individual']
    hid_hypers, results2 = [], []
    for new_id in range(hyper_counts):
        result2 = tokenize_new(row, False, new_id, result[2], result[3])
        hid_new = get_embeddings(result2[0], result2[2], result2[1], model, join_toks) # torch.Size([7, 11, 768])  
        results2.append(result2)
        hid_hypers.append(hid_new)
    
    # HYPONYMS:
    hypo_counts = len(df.hypo[sent_id]) #['guest','invitee','guest of honour']
    hid_hypos, results3 = [], []
    for new_id in range(hypo_counts):
        result3 = tokenize_new(row, True, new_id, result[2], result[3])
        hid_new2 = get_embeddings(result3[0], result3[2], result3[1], model, join_toks) # torch.Size([7, 11, 768])  
        results3.append(result3)
        hid_hypos.append(hid_new2)
        
    return [[result, results2, results3], [hid_target, hid_hypers, hid_hypos]]

# ========================================================

# PCA (with 2 or 3 PCs):
    
def make_pca_df(hidden_statesX, resultX, layers,  use_scaling, join_toks, n_components=2):
    # layers = [0, 1, -1]
    # tok_range = new_range
    # tokenized_sent = tokenized_sent_new
    # hidden_statesX = hid_new
    # use_scaling = True
    # resultX = result # result2
    # hidden_statesX = hidden_states
    # join_toks = False
    # resultX = result2
    
    tokenized_sent = resultX[0]
    tok_range = resultX[1]
    
    # Merge tokens of target word if there are more than and join_toks = True:
    # (merging of hidden states was done for those cases in get_embeddings):
    if join_toks and tok_range[1] - tok_range[0] != 1:
        start, end = tok_range[0], tok_range[1]
        mid = ' '.join(tokenized_sent[start:end]) # 'al ##der trees'
        mid = [re.sub(' ##', '', mid)] # ['alder trees']
        tokenized_sent = tokenized_sent[:start] + mid + tokenized_sent[end:]
        # len(tokenized_sent) # 11
    
       
    # PCA jointly for all layers:
    # hidden_statesX.shape # [7, 8, 768]
    num_layers = hidden_statesX.shape[0] # 7
    num_tokens = hidden_statesX.shape[1] # 8
    emb_layer = hidden_statesX.reshape(num_layers*num_tokens, 768) # ([56, 768])
    
    # PCA:
    pca = PCA(n_components)
    pca_2 = pca.fit_transform(emb_layer)
    # pca_2.shape # (56, 2)    
    pca_2 = pca_2.reshape(num_layers, num_tokens, n_components) #(7,8,2)
    
    # Leave out [CLS] and [SEP] tokens and '.' (we are only interested in relationships between words:
    pca_2 = pca_2[:,1:-2,:] 
    tokenized_sent = tokenized_sent[1:-2]
    
    # PCA df:
    pca_df = pd.DataFrame()
    pca_df["tokens"] = tokenized_sent   
    
    # SCALE EACH LAYER SEPARATELY FOR PLOTTING (OPTIONAL) AND STORE IN DF:
    for layer in layers:
        if use_scaling:
            layer_scaled = MinMaxScaler().fit_transform(pca_2[layer,:,:])  # (8, 2)
        else:
            layer_scaled = pca_2[layer,:,:] # no scaling
        pca_df["pc1_layer" + str(layer)] = layer_scaled[:,0] # shape (8,)
        pca_df["pc2_layer" + str(layer)] = layer_scaled[:,1]
        if n_components == 3:
            pca_df["pc3_layer" + str(layer)] = layer_scaled[:,2]

    if join_toks == True:
        tok_id = tok_range[0] -1 # start is same for target and new
        pca_df['color'] = ["blue" if x!=tok_id else "red" for x  in range(len(pca_df))]
    else:
        pca_df['color'] = ["blue" if x not in range(tok_range[0]-1, tok_range[1]-1) else "red" for x  in range(len(pca_df))]

    return pca_df

    
def make_pca_df_all(prep, layers, use_scaling, join_toks, colors):
    n_components=2  # other values not tested
    # layers = [0, 1, -1]
    # tok_range = result[1] # new_range
    # tokenized_sent = result[0] # tokenized_sent_new
    # tokenized_sent = results3[1][0]
    # tok_range = results3[1][1]
    # hidden_statesX = hid_new
    # use_scaling = True
    # resultX = result # result2
    # hidden_statesX = hidden_states
    # join_toks = False
    # resultX = result2
    
    res_all, hids_all = prep[0], prep[1]
    # target, hypers, hypos:
    result, results2, results3 = res_all[0], res_all[1], res_all[2] 
    hid_target, hid_hypers, hid_hypos = hids_all[0], hids_all[1], hids_all[2]

    # Colors:
    col_target = colors[0]
    col_hypo = colors[1]
    col_hyper = colors[2]
    col_target_rest = colors[3]
    col_hyper_rest = colors[4]
    col_hypo_rest = colors[5]
        
    # ---------------------------------
    
    # TOKENIZED SENTS, COLORS AND INPUT LENGTHS:
        
    # input_lens = [] # sum 81 # 63
    types = []
    tokenized_sents = [] # len 81 # 63
    colors = []   # len 81 # 63     
    
    hyper_counts = len(results2)
    hypo_counts =len(results3)
    
    tok_ranges = [result[1]] + [results2[x][1] for x in range(hyper_counts)] + [results3[x][1] for x in range(hypo_counts)] 
    tok_sents = [result[0]] + [results2[x][0] for x in range(hyper_counts)] + [results3[x][0] for x in range(hypo_counts)] 
    
    for i in range(len(tok_sents)):
        tokenized_sent = tok_sents[i]
        tok_range = tok_ranges[i]
    
        # Merge tokens if join_toks = True (hid states were merged in get_embeddings):
        if join_toks and tok_range[1] - tok_range[0] != 1:
            tokenized_sent = merge_tokens(tokenized_sent, tok_range) # len 11
        # tokenized_sent = tokenized_sent[1:-2] # Leave out [CLS], [SEP] and '.'
        tokenized_sents+=tokenized_sent
        sent_len = len(tokenized_sent)
        # input_lens.append(sent_len)    
    
        # Colors and types:
        if i == 0:
            word_color, context_color = col_target, col_target_rest # targt
            word_type, context_type = "target", "target_context"
        elif i<=hyper_counts: # hyper
            word_color, context_color = col_hyper, col_hyper_rest
            word_type, context_type = "hyper", "hyper_context"
        else: # hypo
            word_color, context_color = col_hypo, col_hypo_rest
            word_type, context_type = "hypo", "hypo_context"
        
        if join_toks == True:
            colors+= [context_color if x!=tok_range[0] else word_color for x  in range(sent_len)] # tok_range[0]-1 if CLS,SEP, '.' left out above
            types+= [context_type if x!=tok_range[0] else word_type for x  in range(sent_len)]
        else:
            colors+= [context_color if x not in range(tok_range[0], tok_range[1]) else word_color for x  in range(sent_len)] # tok_range[0]-1 and tok_range[1]-1 if CLS,SEP, '.' left out above
            types+=[context_type if x not in range(tok_range[0], tok_range[1]) else word_type for x  in range(sent_len)]

        
    # ---------------------------------
    
    # PCA DF:
    pca_df = pd.DataFrame()
    pca_df["tokens"] = tokenized_sents  
    pca_df['color'] = colors 
    pca_df['type'] = types

    # ---------------------------------
    
    # STACK HIDDEN STATES OF ALL SENTENCES:
    hiddens_stacked = [hid_target] + [hid_hypers[x] for x in range(hyper_counts)] + [hid_hypos[x] for x in range(hypo_counts)] 
    hiddens_stacked = torch.cat(hiddens_stacked, dim=1) 
    # hiddens_stacked.shape # torch.Size([7, 81, 768])
    # 81-6*3 = 63 (contains also embeddings for SEP, CLS, '.')
    hidden_statesX = hiddens_stacked

  
    # PCA jointly for all layers:
    num_layers = hidden_statesX.shape[0] # 7
    num_tokens = hidden_statesX.shape[1] # 81
    emb_layer = hidden_statesX.reshape(num_layers*num_tokens, 768) 
    emb_layer.shape # [567, 768] # enne: ([56, 768])
    
    # PCA:
    pca = PCA(n_components)
    pca_2 = pca.fit_transform(emb_layer)
    # pca_2.shape # (567, 2) # enne: (56, 2)    
    pca_2 = pca_2.reshape(num_layers, num_tokens, n_components) # (7, 81, 2)
    
    # ADD TO DF:
    # layers = [1, 6]
    for layer in layers:
        layer2 = pca_2[layer,:,:] # (81, 2)
        pca_df["pc1_layer" + str(layer)] = layer2[:,0] # shape (8,)
        pca_df["pc2_layer" + str(layer)] = layer2[:,1]
        
    # Leave out [CLS] and [SEP] tokens and '.':
    pca_df = pca_df[pca_df.tokens != "[CLS]"] 
    pca_df = pca_df[pca_df.tokens != "[SEP]"]
    pca_df = pca_df[pca_df.tokens != "."]
    # len(pca_df) # 63
        
    
    # DO SCALING (OPTIONAL):
    if use_scaling:
        for col in pca_df.columns.tolist():
            if "layer" in col:
                # print(col)
                a = np.array(pca_df[col])
                a = a.reshape(-1, 1) 
                pca_df[col] =  MinMaxScaler().fit_transform(a) 

    pca_df.index = [x for x in range(len(pca_df))]

    return pca_df


def make_pca_df_all2(preps, layers, use_scaling, join_toks, colors):
    n_components=2  # other values not tested
    # layers = [0, 1, -1]
    # tok_range = result[1] # new_range
    # tokenized_sent = result[0] # tokenized_sent_new
    # tokenized_sent = results3[1][0]
    # tok_range = results3[1][1]
    # hidden_statesX = hid_new
    # use_scaling = True
    # resultX = result # result2
    # hidden_statesX = hidden_states
    # join_toks = False
    # resultX = result2
    
    # Colors:
    col_target = colors[0]
    col_hypo = colors[1]
    col_hyper = colors[2]
    col_target_rest = colors[3]
    col_hyper_rest = colors[4]
    col_hypo_rest = colors[5]    

    # input_lens = [] # sum 81 # 63
    types = []
    tokenized_sents = [] # len 81 # 63
    colors = []   # len 81 # 63   
    
    for prep in preps:
        res_all = prep[0]
        # target, hypers, hypos:
        result, results2, results3 = res_all[0], res_all[1], res_all[2] 
        
        # ---------------------------------
        
        # TOKENIZED SENTS, COLORS AND INPUT LENGTHS:
            
        hyper_counts = len(results2)
        hypo_counts =len(results3)
        
        tok_ranges = [result[1]] + [results2[x][1] for x in range(hyper_counts)] + [results3[x][1] for x in range(hypo_counts)] 
        tok_sents = [result[0]] + [results2[x][0] for x in range(hyper_counts)] + [results3[x][0] for x in range(hypo_counts)] 
        
        for i in range(len(tok_sents)):
            tokenized_sent = tok_sents[i]
            tok_range = tok_ranges[i]
        
            # Merge tokens if join_toks = True (hid states were merged in get_embeddings):
            if join_toks and tok_range[1] - tok_range[0] != 1:
                tokenized_sent = merge_tokens(tokenized_sent, tok_range) # len 11
            # tokenized_sent = tokenized_sent[1:-2] # Leave out [CLS], [SEP] and '.'
            tokenized_sents+=tokenized_sent
            sent_len = len(tokenized_sent)
            # input_lens.append(sent_len)    
        
            # Colors and types:
            if i == 0:
                word_color, context_color = col_target, col_target_rest # targt
                word_type, context_type = "target", "target_context"
            elif i<=hyper_counts: # hyper
                word_color, context_color = col_hyper, col_hyper_rest
                word_type, context_type = "hyper", "hyper_context"
            else: # hypo
                word_color, context_color = col_hypo, col_hypo_rest
                word_type, context_type = "hypo", "hypo_context"
            
            if join_toks == True:
                colors+= [context_color if x!=tok_range[0] else word_color for x  in range(sent_len)] # tok_range[0]-1 if CLS,SEP, '.' left out above
                types+= [context_type if x!=tok_range[0] else word_type for x  in range(sent_len)]
            else:
                colors+= [context_color if x not in range(tok_range[0], tok_range[1]) else word_color for x  in range(sent_len)] # tok_range[0]-1 and tok_range[1]-1 if CLS,SEP, '.' left out above
                types+=[context_type if x not in range(tok_range[0], tok_range[1]) else word_type for x  in range(sent_len)]

        
    # ---------------------------------
    
    # PCA DF:
    pca_df = pd.DataFrame()
    pca_df["tokens"] = tokenized_sents  
    pca_df['color'] = colors 
    pca_df['type'] = types # ln 3099

    # ---------------------------------
    
    # STACK HIDDEN STATES OF ALL SENTENCES:
    
    hiddens_stacked = []
    
    for prep in preps:
        hids_all = prep[1]
        hid_target, hid_hypers, hid_hypos = hids_all[0], hids_all[1], hids_all[2]
        hyper_counts = len(hid_hypers)
        hypo_counts =len(hid_hypos)
        hiddens_stacked+= [hid_target] + [hid_hypers[x] for x in range(hyper_counts)] + [hid_hypos[x] for x in range(hypo_counts)] 
        # len(hiddens_stacked)
        
        
    hiddens_stacked = torch.cat(hiddens_stacked, dim=1) 
    # hiddens_stacked.shape # torch.Size([7, 3099, 768])
    hidden_statesX = hiddens_stacked

  
    # PCA jointly for all layers:
    num_layers = hidden_statesX.shape[0] # 7
    num_tokens = hidden_statesX.shape[1] # 81
    emb_layer = hidden_statesX.reshape(num_layers*num_tokens, 768) 
    # emb_layer.shape # [21693, 768] # enne: [567, 768] # enne: ([56, 768])
    
    # PCA:
    pca = PCA(n_components)
    pca_2 = pca.fit_transform(emb_layer)
    # pca_2.shape # (567, 2) # enne: (56, 2)    
    pca_2 = pca_2.reshape(num_layers, num_tokens, n_components) # (7, 81, 2)
    
    # ADD TO DF:
    # layers = [1, 6]
    for layer in layers:
        layer2 = pca_2[layer,:,:] # (81, 2)
        pca_df["pc1_layer" + str(layer)] = layer2[:,0] # shape (8,)
        pca_df["pc2_layer" + str(layer)] = layer2[:,1]
        
    # Leave out [CLS] and [SEP] tokens and '.':
    pca_df = pca_df[pca_df.tokens != "[CLS]"] 
    pca_df = pca_df[pca_df.tokens != "[SEP]"]
    pca_df = pca_df[pca_df.tokens != "."]
    # len(pca_df) # 2346 # 63
        
    
    # DO SCALING (OPTIONAL):
    if use_scaling:
        for col in pca_df.columns.tolist():
            if "layer" in col:
                # print(col)
                a = np.array(pca_df[col])
                a = a.reshape(-1, 1) 
                pca_df[col] =  MinMaxScaler().fit_transform(a) 

    pca_df.index = [x for x in range(len(pca_df))]

    return pca_df


def merge_tokens(tokenized_sentX, tok_rangeX): 
    # input: tokenized sent, output: tokenized sent where target/replacement word tokens have been merged
    start, end = tok_rangeX[0], tok_rangeX[1]
    mid = ' '.join(tokenized_sentX[start:end]) # 'al ##der trees'
    mid = [re.sub(' ##', '', mid)] # ['alder trees']
    tokenized_sentX = tokenized_sentX[:start] + mid + tokenized_sentX[end:]
    return tokenized_sentX

# -----------------------------------------

# PLOT EMBEDDINGS 2D:
    
# PLOT ONE SENTENCE (WITH EITHER TARGET OR REPLACEMENT WORD):

def plot_embeddings(layers, pca_dfX, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels):
   # pca_dfX = pca_df
    
    single_row = type(sent_id) == int # True - plot one sentence, possibly with multiple hyponyms/hypernyms # False: plot multiple sentences (rows in df), with multiple hyponyms/hypernyms in each
    if single_row:
        lemma = df.loc[sent_id]["lemma"]
        sent = df.loc[sent_id]["sent"]
        plot_title = "Sentence: " + re.sub(lemma, lemma.upper(), sent)       
    else:
        plot_title = sent_id
        
    # Only plot context words for original sentence:
    has_type = "type" in pca_dfX.columns.tolist() # if we plot multiple sents
    if has_type:
        pca_dfX = pca_dfX[pca_dfX["type"] != "hyper_context"]
        pca_dfX = pca_dfX[pca_dfX["type"] != "hypo_context"]    
        pca_dfX.index = [x for x in range(len(pca_dfX))]
    
    # Order so that hyper, target will be plotted last:
    if has_type:
        d = {"target": 3, "hyper": 2, "hypo": 1, "target_context": 0, "hyper_context": -1, "hypo_context": -2}
        priorities = []
        
        for t in pca_dfX["type"].tolist():
            priorities.append(d[t])
            
        pca_dfX["priority"] = priorities
        pca_dfX= pca_dfX.sort_values(by = ["priority"])
        pca_dfX.index = [x for x in range(len(pca_dfX))]    
    
    # Rows and columns of subplots:
    nrows = str(2 if len(layers)>3 else 1)
    ncols = str(len(layers) if len(layers) < 4 else 3)
    # pca_df2['color'] = "blue"
    # pca_df2['color'][tok_id] = "red"
    
    # Plot:
    plt.rcParams['figure.figsize'] = figsize # [18, 6] # [12, 3]
    # plt.rcParams.update({'font.size': 26})
    plt.figure() 
    plt.suptitle(plot_title)

    for i, layer in enumerate(layers):        
        plt.subplot(int(nrows + ncols + str(i+1)))
        label1 ="pc1_layer" + str(layer)
        label2 = "pc2_layer" + str(layer)
        plt.grid(alpha=0.75)
        if single_row:
            plt.scatter(pca_dfX[label1], pca_dfX[label2], c = pca_dfX["color"]) # marker="o", markerfacecolor = "red", markeredgecolor="orange", markersize=20
        else:
            plt.scatter(pca_dfX[label1], pca_dfX[label2], c = pca_dfX["color"], marker = "o", s = 2) # s - marker size
            #marker = "o", markersize=5

        plt.title("Embeddings in layer " + str(layer))
        plt.xlabel("PC1")
        plt.ylabel("PC2")   
        
        # plt.show()

        # Add labels:
        if has_labels:
            for j, label in enumerate(pca_dfX['tokens']):
                if has_type:
                    tok_type = pca_dfX.loc[j]['type']
                    # alpha = 0.5 if "context" in tok_type else 1.0
                    if "context" not in tok_type:
                        label = label.upper()
                    
                x = pca_dfX[label1][j] + 0.03 
                y = pca_dfX[label2][j] + 0.03
                color_ = pca_dfX["color"][j]
                if use_scaling == True:  # axis in range (0.0, 1.0)
                    if x > 0.9:
                        x-=0.1
                    if y > 0.95:
                        y-=0.1
                plt.annotate(label, (x, y), color = color_, ) # alpha = alpha
            
    plt.subplots_adjust(hspace = 0.45, wspace = 0.3)
    
    if save_plot:
        if single_row: 
            fn = p2 + str(sent_id) + "_" + lemma + ".png"
        else:
            fn = p2 + sent_id + ".png"
        plt.savefig(fn)    
    else:
        plt.show()


# ========================================================

# COSINE DISTANCES:
    
# Distance of target emb. to other embeddings in the layer:
def dists_target2rest(target_embs, results_target, hid_target, layers):
    dists_target2rest = {f'layer_{i}': [] for i in layers}
    # tok_range= results_target[1]
    tok_start = results_target[1][0]
    num_tokens = hid_target.shape[1]
    
    for i, layer in enumerate(layers):
        target_emb = target_embs[i] # hid_target[layer, tok_start, :]
        for j in range(num_tokens):
            if j!=tok_start:
                dist = cosine(hid_target[layer, j, :], target_emb)
                dists_target2rest[f'layer_{layer}'].append(dist)
    # dists_target2rest['layer_6']
    
    mean_dists_target2rest = []
    for layer in layers:
        layer_dists = dists_target2rest[f'layer_{layer}']
        mean_dists_target2rest.append(np.mean(layer_dists))    
    
    return mean_dists_target2rest

# Distance of target embedding to hyponyme/hypernym embeddings in each layer:
def dists_target2hyp(target_embs, results_target, hid_target, results_hyps, hid_hyps, layers):
    dists_target2hyp = {f'layer_{i}': [] for i in layers}
    tok_start = results_target[1][0]
    
    for hid_hyp in hid_hyps: 
        for i, layer in enumerate(layers):
            target_emb = target_embs[i] # hid_target[layer, tok_start, :]
            hyp_emb = hid_hyp[layer, tok_start, :]
            dist = cosine(hyp_emb, target_emb)
            dists_target2hyp[f'layer_{layer}'].append(dist)

    mean_dists_target2hyp = []
    for layer in layers:
        layer_dists = dists_target2hyp[f'layer_{layer}']
        mean_dists_target2hyp.append(np.mean(layer_dists))   
    
    return mean_dists_target2hyp

    
# Distance of hyponyme/hypernym emb. to other embeddings in the layer (for all hyponymes/hypernyms)
def dists_hyp2rest(results_hyps, hid_hyps, layers):
    dists_hyp2rest = {f'layer_{i}': [] for i in layers}
    tok_start = results_hyps[0][1][0] # from results of first hypo 
    num_tokens = hid_hyps[0].shape[1] # from results of first hypo 

    for hid_hyp in hid_hyps: 
        for i, layer in enumerate(layers):
            hyp_emb = hid_hyp[layer, tok_start, :]
            for j in range(num_tokens):
                if j!=tok_start:
                    dist = cosine(hid_hyp[layer, j, :], hyp_emb)
                    dists_hyp2rest[f'layer_{layer}'].append(dist)        
    
    mean_dists_hyp2rest = []
    for layer in layers:
        layer_dists = dists_hyp2rest[f'layer_{layer}']
        mean_dists_hyp2rest.append(np.mean(layer_dists))
        
    return mean_dists_hyp2rest


# ========================================================

# LOAD REVIEWED DF, MODEL, TOKENIZER:

df = load_reviewed(p, fname)

model_name = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(model_name) 
model = DistilBertModel.from_pretrained(
    model_name, output_hidden_states=True, output_attentions=True)

model.eval() 

inflect_eng = inflect.engine() # for convering nouns to plural


# ========================================================

# PLOTTNG PCA OF EMBEDDINGS OF ONE SENTENCE:
    
# -- Plots were done in IDE and not saved. Analysed qualitatively.
    
figsize = [12, 6] # [18, 6]
layers = [1, 2, 3, 4, 5, 6] # [0, 1, -1] # [1, 5] # for pca plots
use_scaling = True # dont change (for pca)
# use_scaling = False
n_components = 2
join_toks = True # False # True
save_plot = True
has_labels = True

sent_id = 8
is_hypo = True # False # True # hyponyme or hypernyme
new_id = 0 # id of hyponyme or hypernyme in their list

# len(df.hypo[sent_id])
# len(df.hyper[sent_id])
# df.hypo[sent_id][12]
        
# ---------------

sent_id, new_id, is_hypo, join_toks = 8, 0, True, True
sent_id, new_id, is_hypo, join_toks = 8, 12, True, False # tree
sent_id, new_id, is_hypo, join_toks = 2, 0, True, False # door
sent_id, new_id, is_hypo, join_toks = 2, 0, False, False
sent_id, new_id, is_hypo, join_toks = 2, 0, False, True


# a) Original sentence (containing target word):
save_plot = True
sent_id = 31 #117 apple
lemma = df.loc[sent_id]["lemma"]
print(lemma)

result, hid_target = get_results_orig(df, sent_id, model, join_toks)
pca_df_target = make_pca_df(hid_target, result, layers, use_scaling, join_toks, n_components=2)
plot_embeddings(layers, pca_df_target, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels) 


# b) Sentence with hyponym/hypernym:
is_hypo =  True
counts = len(df.hypo[sent_id]) if is_hypo else len(df.hyper[sent_id]) 
print(counts)
new_id = 2
join_toks = True # False

result2, hid_new = get_results_new(df, sent_id, is_hypo, new_id, result, model, join_toks)
pca_df_new = make_pca_df(hid_new, result2, layers, use_scaling, join_toks, n_components=2)
plot_embeddings(layers, pca_df_new, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels) 


# ==========================================================

# ALL HYPONYMS/HYPERNYMS AND THEIR LEMMA ON SAME PLOT:
    
# -- other words plotted only for lemma (their locations shifted a bit when lemma 
# was replaced with a related term (hyponyms/hypernyms), but not too much).

col_hyper_rest = "#bababa" # gray
col_hypo_rest = "#e0e0e0" # light gray
col_hyper = "#2171b5"  # red --> blue
col_target = "#cb181d" # blue --> red
col_hypo = "#66bd63" # green
col_target_rest = "#4d4d4d" #  "#878787"

colors = [col_target, col_hypo, col_hyper, 
          col_target_rest, col_hypo_rest, col_hyper_rest]

figsize = [18, 9] # [12, 6] # [18, 6]
layers = [1, 2, 3, 4, 5, 6] # [0, 1, -1] # [1, 5] # for pca plots
use_scaling = True # dont change (for pca)
# use_scaling = False
save_plot = False
save_plot = True
with_labels = True

# TEST:
sent_id, join_toks = 28, False
sent_id, join_toks = 8, False
sent_id, join_toks = 7, True

# lemma = df.loc[sent_id]["lemma"]
# sent = df.loc[sent_id]["sent"]
sent_id = 122 # 122
prep = get_results_all(df, sent_id, model, join_toks)
pca_df = make_pca_df_all(prep, layers, use_scaling, join_toks, colors)
plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels) 


# SAVE ALL 124 PLOTS:

figsize = [14, 7] # [18, 6]
layers = [1, 2, 3, 4, 5, 6] # [0, 1, -1] # [1, 5] # for pca plots
use_scaling = True     
save_plot = True
join_toks = True

for sent_id in range(len(df)):
    lemma = df.loc[sent_id]["lemma"]
    print(sent_id, lemma, '::', df.loc[sent_id]["sent"])
    prep = get_results_all(df, sent_id, model, join_toks)
    pca_df = make_pca_df_all(prep, layers, use_scaling, join_toks, colors)
    plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels) 
    # print(sent_id)


# sent_id = 71 # 122
# # lemma = df.loc[sent_id]["lemma"]
# # sent = df.loc[sent_id]["sent"]
# prep = get_results_all(df, sent_id, model, join_toks)
# pca_df = make_pca_df_all(prep, layers, use_scaling, join_toks, colors)
# plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels)
# print(sent_id)

 
# ==========================================================

# PLOT WITHOUT LABELS:

use_scaling = True     
save_plot = True
join_toks = True
has_labels = False

for sent_id in range(len(df)):
    lemma = df.loc[sent_id]["lemma"]
    print(sent_id, lemma, '::', df.loc[sent_id]["sent"])
    prep = get_results_all(df, sent_id, model, join_toks)
    pca_df = make_pca_df_all(prep, layers, use_scaling, join_toks, colors)
    plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels) 
    # print(sent_id)
 
# ========================================================== 

# 20 SENTENCES ON THE SAME PLOT, WITHOUT LABELS

# -- labels only for target words
has_labels = False
save_plot = True
# len(df) # 123
snts = [100, 123] # (0, 20), (100, 123)
sent_ids = [x for x in range(snts[0], snts[1])] 
preps = []
for sent_id in sent_ids:
    preps.append(get_results_all(df, sent_id, model, join_toks))

pca_df = make_pca_df_all2(preps, layers, use_scaling, join_toks, colors)
# len(preps) # 20
# len(pca_df) # 2346

sent_id = "sentences " + str(snts[0]) + "-" + str(snts[1]-1) 
plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels)

# ========================================================== 

## ALL SENTENCES ON THE SAME PLOT, WITHOUT LABELS:
sent_ids = [x for x in range(len(df))] # (0, 20)
preps = []
for sent_id in sent_ids:
    preps.append(get_results_all(df, sent_id, model, join_toks))

pca_df = make_pca_df_all2(preps, layers, use_scaling, join_toks, colors)

# len(preps) # 20
# len(pca_df) # 2346
has_labels = False
save_plot = True
sent_id = "All sentences" # 0-20 # 20-39 # 
plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, df, sent_id, has_labels)

# ==========================================================

# COSINE DISTANCES

# MEASURE COS DISTANCE BETWEEN:
# - TARGET WORD AND HYPONYMS/HYPERNYMS 
# - TARGET WORD AND AND REST OF THE WORDS (in target word's sentence)
# - HYPONYMES/HYPERNYMES AND REST OF THE WORDS (in hyponym's/hypernym's sentence)

# -- if target word, hyponyme or hypernym consisted of several tokens, 
# their embeddings were averaged

## NB!! Cos distance is distance, not similarity --> dist between vec
##  and itself is 0.0:
# dist = cosine(hid_target[0, 0, :], hid_target[0, 0, :]) # 0 

# use_scaling = False
join_toks = True
num_layers = len(layers)

sent_ids = [x for x in range(100, 123)] # (0, 20)
preps = []
for sent_id in sent_ids:
    preps.append(get_results_all(df, sent_id, model, join_toks))
    
# all_distances = {f'layer_{i}': [] for i in layers}
# {'layer_1': [],
#  'layer_2': [],
#  'layer_3': [],
#  'layer_4': [],
#  'layer_5': [],
#  'layer_6': []}

# len(preps) # 123

all_dists_target2rest  = []
all_dists_hypo2rest = []
all_dists_hyper2rest = []
all_dists_target2hypo = []
all_dists_target2hyper = []    

for sent_id in range(len(preps)):
    # sent_id = 0    
    
    # Target embeddings, results:
    results_target = preps[sent_id][0][0] # sent id, results, target
    hid_target = preps[sent_id][1][0]  # sent id, hidden_states, target
    # hid_target.shape # [7, 14, 768]
    # tok_start = results_target[1][0]
    target_embs = [hid_target[x, tok_start, :] for x in layers]
    
    # Hyponyme/hypernym embeddings, results: 
    results_hypos = preps[sent_id][0][2] # sent id, results, hypos
    hid_hypos = preps[sent_id][1][2] # sent id, hids, hypos
    # hypo_counts =len(results_hypos) # 17
    # tok_ranges = [results_hypos[x][1] for x in range(hypo_counts)] # len 17
    results_hypers = preps[sent_id][0][1] # sent id, results, hypos
    hid_hypers = preps[sent_id][1][1] # sent id, hids, hypos
    
    
    # Distance of target/hyponyme/hypernym emb. to other embeddings in the layer (for all hyponymes/hypernyms):
    all_dists_target2rest.append(dists_target2rest(target_embs, results_target, hid_target, layers))
    all_dists_hypo2rest.append(dists_hyp2rest(results_hypos, hid_hypos, layers))
    all_dists_hyper2rest.append(dists_hyp2rest(results_hypers, hid_hypers, layers))
    
    # Distance of target embedding to hyponyme/hypernym embeddings in each layer:
    all_dists_target2hypo.append(dists_target2hyp(target_embs, results_target, hid_target, results_hypos, hid_hypos, layers))
    all_dists_target2hyper.append(dists_target2hyp(target_embs, results_target, hid_target, results_hypers, hid_hypers, layers))


# Dataframe of distances:
means_target2rest = np.mean(np.array(all_dists_target2rest), axis = 0)
means_hypo2rest = np.mean(np.array(all_dists_hypo2rest), axis = 0)
means_hyper2rest = np.mean(np.array(all_dists_hyper2rest), axis = 0)
means_target2hypo = np.mean(np.array(all_dists_target2hypo), axis = 0)
means_target2hyper = np.mean(np.array(all_dists_target2hyper), axis = 0)

df_means = pd.DataFrame(data = [layers, means_target2hyper, means_target2hypo, means_target2rest, means_hyper2rest, means_hypo2rest]).T

df_means.columns = ["layer", "target2hyper", "target2hypo", "target2rest", "hyper2rest", "hypo2rest"]


df_means
#    layer  target2hyper  target2hypo  target2rest  hyper2rest  hypo2rest
# 0    1.0      0.638420     0.576258     0.699524    0.730394   0.708319
# 1    2.0      0.528787     0.488041     0.636356    0.669827   0.647615
# 2    3.0      0.408241     0.373546     0.509377    0.539567   0.512896
# 3    4.0      0.291147     0.264582     0.401568    0.424768   0.402173
# 4    5.0      0.204409     0.189776     0.335130    0.352437   0.344317
# 5    6.0      0.271908     0.257701     0.438266    0.468496   0.455659


# stds_t2rest = np.std(np.array(all_dists_target2rest), axis = 0)

# ---------------------------------------------

# COS DISTANCE:
#    
# hid_target.shape # [7, 11, 768]
# dists = []
# for i in range(7):
#     for j in range(7):
#         if i<j:
#             for k in range(11):
#                 for m in range(11):
#                     if k!=m:
#                         d = cosine(hid_target[i, k, :], hid_target[j, m, :])
#                         dists.append(d)
#                         print(round(d, 2), end = ', ')

# max(dists) # 1.12057
# min(dists) # 0.20

# dist = cosine(hid_target[0, 0, :], hid_target[0, 1, :]) # 0.8

## NB!! Its distance, not similarity --> dist between vec and itself is 0.0:
# dist = cosine(hid_target[0, 0, :], hid_target[0, 0, :]) # 0 


# ==========================================================

# IDEA: REMOVE PREPOSITIONS AND ARTICLES:
    
# Prepositions (eg. 'the', 'a', 'in', 'on', 'with') are not very informative; also, they tend to be quite far away from other tokens on the plots. So removing them would make other tokens less clustered and labels a bit less overlapping. 

# toks_dropped = ['to', 'at', 'with', 'on', 'in', 'a', 'the', 'as', 'of', 'without']
# drop_toks = True

# IDEA: GATHER NOUN PHRASES OR ADJECTIVE + NOUN (delicious red apple)
    
# Gather nouns and preceding adjectives, or noun phrases consisting of several words (or at least nouns consisting of seveeral tokens). See how their embeddings come close to form a single meaning. 



