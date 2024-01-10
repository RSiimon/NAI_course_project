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

# pip install inflect
# pip install pattern3


p = "data/"
# fname = "ChatGPT_simple_sents2.csv"
# fname_rev = "df_rev2_63.csv"
fname = "df_final.csv"
p2 = "plots/"


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
    
       
    # A) PCA jointly for all layers:
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

def plot_embeddings(layers, pca_dfX, figsize, use_scaling, save_plot, p2, lemma, sent_id):
    # layers = [1, 2, 3, 4, 5, 6]
    # pca_dfX = pca_df_new
    # pca_dfX = pca_df_target
    # figsize = [12, 6] # [18, 6]
    # tok_id = result[1][0] - 1 # target
    # tok_id = result2[1][0] - 1 #new 
   
    # Only plot context words for original sentence:
    pca_dfX = pca_dfX[pca_dfX["type"] != "hyper_context"]
    pca_dfX = pca_dfX[pca_dfX["type"] != "hypo_context"]    
    pca_dfX.index = [x for x in range(len(pca_dfX))]
    
    # Order so that hyper, target will be plotted last:
    d = {"target": 3, "hyper": 2, "hypo": 1, "target_context": 0, "hyper_context": -1, "hypo_context": -2}
    priorities = []
    
    for t in pca_dfX["type"].tolist():
        priorities.append(d[t])
        
    pca_dfX["priority"] = priorities
    pca_dfX= pca_dfX.sort_values(by = ["priority"])
    pca_dfX.index = [x for x in range(len(pca_dfX))]    
    
    # Plot:
    plt.rcParams['figure.figsize'] = figsize # [18, 6] # [12, 3]
    # plt.rcParams.update({'font.size': 26})

    nrows = str(2 if len(layers)>3 else 1)
    ncols = str(len(layers) if len(layers) < 4 else 3)
    # pca_df2['color'] = "blue"
    # pca_df2['color'][tok_id] = "red"
    
    for i, layer in enumerate(layers):        
        plt.subplot(int(nrows + ncols + str(i+1)))
        label1 ="pc1_layer" + str(layer)
        label2 = "pc2_layer" + str(layer)
        plt.grid(alpha=0.75)
        plt.scatter(pca_dfX[label1], pca_dfX[label2], c = pca_dfX["color"]) # marker="o", markerfacecolor = "red", markeredgecolor="orange", markersize=20

        plt.title("Embeddings in layer " + str(layer))
        plt.xlabel("PC1")
        plt.ylabel("PC2")    

        for j, label in enumerate(pca_dfX['tokens']):
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
        # fn = p2 + lemma + ".jpg"
        fn = p2 + str(sent_id) + "_" + lemma + ".png"
        plt.savefig(fn)    
    else:
        plt.show()


# PCA + plotting:
def pca_plot(hidden_statesX, resultX, layers, figsize, use_scaling, join_toks, n_components=2):
    # hidden_statesX = hidden_states
    pca_dfX_ = make_pca_df(hidden_statesX, resultX, layers, use_scaling, join_toks, n_components=2)
    plot_embeddings(layers, pca_dfX_, figsize, use_scaling) 

    
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

sent_id = 8
is_hypo = True # False # True # hyponyme or hypernyme
# is_hypo = False
new_id = 0 # id of hyponyme or hypernyme in their list
join_toks = True # False # True

# for i, h in enumerate(df.hyper):
#     if "dog" in h:
#         print(i)
len(df.hypo[sent_id])
len(df.hyper[sent_id])
df.hypo[sent_id][12]
        

# a) Original sentence (containing target word):
result, hid_target = get_results_orig(df, sent_id, model, join_toks)
pca_plot(hid_target, result, layers, figsize, use_scaling, join_toks, n_components=2)
# pca_df_target = make_pca_df(hid_target, result, layers, use_scaling, join_toks, n_components=2)
# plot_embeddings(layers, pca_df_target, figsize, use_scaling) 

# b) Sentence with hyponym/hypernym:
result2, hid_new = get_results_new(df, sent_id, is_hypo, new_id, result, model, join_toks)
pca_plot(hid_new, result2, layers, figsize, use_scaling, join_toks, n_components=2)
# pca_df_new = make_pca_df(hid_new, result2, layers, use_scaling, join_toks, n_components=2)
# plot_embeddings(layers, pca_df_new, figsize, use_scaling) 

# ----------------------

# sent_id, new_id, is_hypo, join_toks = 8, 0, True, True
sent_id, new_id, is_hypo, join_toks = 8, 12, True, False # tree
sent_id, new_id, is_hypo, join_toks = 2, 0, True, False # door
sent_id, new_id, is_hypo, join_toks = 2, 0, False, False
sent_id, new_id, is_hypo, join_toks = 2, 0, False, True

result, hid_target = get_results_orig(df, sent_id, model, join_toks)
pca_plot(hid_target, result, layers, figsize, use_scaling, join_toks, n_components=2)

is_hypo = False
is_hypo = True

counts = len(df.hypo[sent_id]) if is_hypo else len(df.hyper[sent_id]) 
for i in range(counts):
    result2, hid_new = get_results_new(df, sent_id, is_hypo, i, result, model, join_toks)
    pca_plot(hid_new, result2, layers, figsize, use_scaling, join_toks, n_components=2)

# ==========================================================

# PLOTTING HYPONYMS/HYPERNYMS AND LEMMA ON SAME PLOT:
    
# -- other words plotted only for lemma (their locations shifted a bit when lemma 
# was replaced with a related term (hyponyms/hypernyms), but not too much).

col_hyper_rest = "#bababa" # gray
col_hypo_rest = "#e0e0e0" # light gray
col_hyper = "#cb181d" # "red"
col_target = "#2171b5" # "black"
col_hypo = "#66bd63" # "green"
col_target_rest = "#4d4d4d" #  "#878787"

colors = [col_target, col_hypo, col_hyper, 
          col_target_rest, col_hypo_rest, col_hyper_rest]

figsize = [18, 9] # [12, 6] # [18, 6]
layers = [1, 2, 3, 4, 5, 6] # [0, 1, -1] # [1, 5] # for pca plots
use_scaling = True # dont change (for pca)
# use_scaling = False
save_plot = True

# TEST:
sent_id, join_toks = 28, False
sent_id, join_toks = 8, False
sent_id, join_toks = 7, True

lemma = df.loc[sent_id]["lemma"]
prep = get_results_all(df, sent_id, model, join_toks)
pca_df = make_pca_df_all(prep, layers, use_scaling, join_toks, colors)
plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, lemma, sent_id) 


# SAVE ALL 64 PLOTS:

figsize = [14, 7] # [18, 6]
layers = [1, 2, 3, 4, 5, 6] # [0, 1, -1] # [1, 5] # for pca plots
use_scaling = True     
save_plot = True
join_toks = True

# for sent_id in range(len(df)):
#     lemma = df.loc[sent_id]["lemma"]
#     prep = get_results_all(df, sent_id, model, join_toks)
#     pca_df = make_pca_df_all(prep, layers, use_scaling, join_toks, colors)
#     plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, lemma, sent_id) 
#     print(sent_id)


sent_id = 64
lemma = df.loc[sent_id]["lemma"]
prep = get_results_all(df, sent_id, model, join_toks)
pca_df = make_pca_df_all(prep, layers, use_scaling, join_toks, colors)
plot_embeddings(layers, pca_df, figsize, use_scaling, save_plot, p2, lemma, sent_id)
print(sent_id)

 

# 
  



# ------------------------------------------





# TODO:
# Plot new and target on same plot (teised sõnad eri tooni; või lisaks nool nende vahel - et kust kuhu abstraktsemaks muutumisel liikus; lisaks: erista hypot/hyperit: värv, toon ja või suurtähed abstraktsema puhul)

# TODO:
# Plot only target word and all its hyponyms or hypernyms (leaving out other words for clarity)

# TODO:
# cos sims
    


# ========================================================

# TODO:
# PLOT TWO SENTENCES (WITH TARGET SND REPLACEMENT WORD):

def plot_embeddings(layers, pca_df, figsize, use_scaling):
    # layers = [1, 2, 3, 4, 5, 6]
    # pca_df = pca_df_new
    # pca_df = pca_df_target
    # figsize = [12, 6] # [18, 6]
    # tok_id = result[1][0] - 1 # target
    # tok_id = result2[1][0] - 1 #new 
   
    plt.rcParams['figure.figsize'] = figsize # [18, 6] # [12, 3]
    # plt.rcParams.update({'font.size': 26})

    nrows = str(2 if len(layers)>3 else 1)
    ncols = str(len(layers) if len(layers) < 4 else 3)
    # pca_df2['color'] = "blue"
    # pca_df2['color'][tok_id] = "red"
    
    for i, layer in enumerate(layers):        
        plt.subplot(int(nrows + ncols + str(i+1)))
        label1 ="pc1_layer" + str(layer)
        label2 = "pc2_layer" + str(layer)
        plt.grid(alpha=0.75)
        plt.scatter(pca_df[label1], pca_df[label2], c = pca_df["color"]) # marker="o", markerfacecolor = "red", markeredgecolor="orange", markersize=20

        plt.title("Embeddings in layer " + str(layer))
        plt.xlabel("PC1")
        plt.ylabel("PC2")    

        for j, label in enumerate(pca_df['tokens']):
            x = pca_df[label1][j] + 0.03 
            y = pca_df[label2][j] + 0.03
            color_ = pca_df["color"][j]
            if use_scaling == True:  # axis in range (0.0, 1.0)
                if x > 0.9:
                    x-=0.1
                if y > 0.95:
                    y-=0.1
            plt.annotate(label, (x, y), color = color_)
            
    plt.subplots_adjust(hspace = 0.45, wspace = 0.3)
    plt.show()


# ============================================

# SAVE:

# df.to_csv(p + "df_final2.csv", encoding = "utf-8")



# ========================================================

# SCRAP:

# ======================================================== 

# OLD:
# model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name, output_hidden_states=True, output_attentions=True)    

# ========================================================

# MOST COMMENTS REMOVED FROM HERE IN FINAL VERSION:

# TOKENIZE AND IDENTIFY IDS OF TARGET WORD AND HYPONYM/HYPERNYM IN TOKENIZED SENTENCE

# -- Needed for token tracking
# -- Consider that tokenizeer may split single word into multiple tokens, and that some hyponyms/hypernyms consist of several words.
# -- Original target word is always just one word, not a phrase.

sent_id = 8
is_hypo = True # hyponyme or hypernymee
new_id = 0 # id of hyponyme or hypernyme in their list

row = df.loc[sent_id]
# word_orig = row.word # 'trees'

# Tokenized original sentence, its parts and location of target word:
result = tokenize_orig(row)
# tokenized_sent_orig = result[0] # ['[CLS]', 'the', 'ancient', 'trees', 'whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]']
# target_range = result[1] #  [3, 4]
# tokens_start = result[2] # ['[CLS]', 'the', 'ancient']
# tokens_end = result[3] # ['whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]'])
# tokens_target = result[4] # ['trees']

print(result)
## ['[CLS]', 'the', 'ancient', 'trees', 'whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]'],
## [3, 4],  <-- location of targeet word in tokenized sentence
## ['[CLS]', 'the', 'ancient'], 
## ['whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]'],
## ['trees']]

# New tokenized sentence, tokenized new word with its location:
result2 = tokenize_new(row, is_hypo, new_id, result[2], result[3])
# tokenized_sent_new = result2[0] # ['[CLS]', 'the', 'ancient', 'al', '##der', 'trees', 'whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]']
# new_range = result2[1] # [3, 6]
# tokens_new = result2[2] # ['al', '##der', 'trees']

print(result2)
## [['[CLS]', 'the', 'ancient', 'al', '##der', 'trees', 'whispered', 'in', 'the', 'peaceful', 'forest', '.', '[SEP]'], 
## [3, 6],  <-- location of replacement word in tokenized sentence
## ['al', '##der', 'trees']]

# # Tokenized target word and replacement word:
# tokenized_sent_orig[target_range[0]: target_range[1]] # ['trees']
# tokenized_sent_new[new_range[0]: new_range[1]] # ['al', '##der', 'trees']

# ========================================================

# TEST OUTPUTS:
    
text = "im sad because you are always accusing me"
tokens = tokenizer.tokenize("[CLS] " + text + " [SEP]") # '##ing',
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
tokens_tensor = torch.tensor([indexed_tokens])
   
with torch.no_grad():
    outputs = model(tokens_tensor) # , segments_tensors

# tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
# print(tokens) # ['[CLS]', 'im', 'sad', 'because', 'you', 'are', 'accusing', 'me', 'of', 'this', 'paradigm', '[SEP]']

# # ALT:
# inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=768)     
# with torch.no_grad():
#     outputs = model(**inputs)

# ---------------

# Prediction:
# probs = torch.nn.functional.softmax(outputs.logits, dim=-1) # [0.9979, 0.0021]
# predicted_class = torch.argmax(probs, dim=-1).numpy()[0] # 0

# Hidden states:
hidden_states = outputs.hidden_states # len 7 
# type(hidden_states)  #tuple
# len(hidden_states) # 7: 6 layers + input embedding 
# len(hidden_states[0]) # 1 - batch size 
# len(hidden_states[0][0]) # 10 - No. of tokens
# len(hidden_states[0][0][0])  # 768 - No. of hidden units

embeddings = torch.stack(hidden_states, dim=0)
# embeddings.size() #torch.Size([7, 1, 10, 768])
embeddings = torch.squeeze(embeddings, dim=1) # remove dim 1
# embeddings.size() #torch.Size([7, 10, 768])
embeddings = embeddings.permute(1,0,2) # swap dims
# embeddings.size() #torch.Size([10, 7, 768])    

#  Layers: 0 - input embeds, 1 - 1st hidden state, 6 - last hidden state
layer_embeddings = embeddings[:,5,:]
# layer_embeddings.shape # torch.Size([10, 768])
# layer_embeddings.numpy().shape #  (10, 768) np.array

# layer_embeddings = hidden_states[5].squeeze(0).numpy()  # ALT
# layer_embeddings.shape # (10, 768) np.array


# ------------------------------------------------------

# HIDDEN STATES:
# layer5_embeddings = hid_new[5] # torch.Size([11, 768])
# target_emb = layer5_embeddings[new_range[0],:]
# target_emb.shape # torch.Size([768])

# ------------------------------------------------------

# REMOVED FROM make_pca_df():    
    # # B) PCA sparately for each layer:  NO! (vt scrap)
    # for layer in layers: # 0 - input embedding layer, 1 - first hidden state, -1 - last hidden state
    # # emb_layer = hidden_states[layer][0]
    #     emb_layer = hidden_states[layer]
    #     # emb_layer.size() # torch.Size([11, 768])
    
    #     # PCA:
    #     pca = PCA(n_components=2)
    #     pca_2 = pca.fit_transform(emb_layer)
    #     # pca_2.shape # (11, 2)
    
    #     # SCALE FOR PLOTTING:
    #     if use_scaling:
    #         pca_scaled = MinMaxScaler().fit_transform(pca_2)  # (11, 2)
    #     else:
    #         pca_scaled = pca_2
    
    #     # STORE IN DF:
    #     pca_df["pc1_layer" + str(layer)] = pca_scaled[:,0] # shape (11,)
    #     pca_df["pc2_layer" + str(layer)] = pca_scaled[:,1]
    
# ------------------------------------------------------

# PCA OF EMBEDDINGS OF ONE SENTENCE:

# a) Original sentence (containing target word):
pca_df_target = make_pca_df(hid_target, result[0], result[1], layers, use_scaling, n_components=2)

plot_embeddings(layers, pca_df_target, figsize, use_scaling) 

# b) Sentence where target word has beed replaced with hyponym/hypernym:
pca_df_new = make_pca_df(hid_new, result2[0], result2[1], layers, use_scaling, n_components=2)

plot_embeddings(layers, pca_df_new, figsize, use_scaling) 

# ------------------------------------------------------

# FINAL:

figsize = [12, 6] # [18, 6]
layers = [1, 2, 3, 4, 5, 6] # [0, 1, -1] # [1, 5] # for pca plots
use_scaling = True # dont change (for pca)

# for i, h in enumerate(df.hyper):
#     if "dog" in h:
#         print(i)
        
sent_id = 8
is_hypo = True # False # True # hyponyme or hypernymee
new_id = 0 # id of hyponyme or hypernyme in their list
join_toks = True # False # True

# a) Original sentence (containing target word):
result, hid_target = get_results_orig(df, sent_id, model, join_toks)
# pca_df_target = make_pca_df(hid_target, result, layers, use_scaling, join_toks, n_components=2)
# plot_embeddings(layers, pca_df_target, figsize, use_scaling) 
pca_plot(hid_target, result, layers, figsize, use_scaling, join_toks, n_components=2)


# b) Sentence with hyponym/hypernym:
result2, hid_new = get_results_new(df, sent_id, is_hypo, new_id, result, model, join_toks)
# pca_df_new = make_pca_df(hid_new, result2, layers, use_scaling, join_toks, n_components=2)
# plot_embeddings(layers, pca_df_new, figsize, use_scaling) 
pca_plot(hid_new, result2, layers, figsize, use_scaling, join_toks, n_components=2)


pca_plot(hid_new, result2, layers, figsize, True, join_toks, n_components=2)
pca_plot(hid_new, result2, layers, figsize, False, join_toks, n_components=2)

# ------------------------------------------------------


# ------------------------------------------------------







    