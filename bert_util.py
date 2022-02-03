#functions for creating masks and segments
"""
This file contains helper functions for the BERT based models used in the training.
"""

BERT_LAYER_LINK = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"

from tqdm import tqdm
import numpy as np
import sys
import pandas as pd

from TripleFormation import topic_entity_filter
from data_preprocess import get_all_topics

def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))
 
def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def create_single_input(sentence,MAX_LEN, max_seq_len, tokenizer, triples=None, custom_masks=None):
  stokens = tokenizer.tokenize(sentence)
  stokens = stokens[:MAX_LEN]
  stokens = ["[CLS]"] + stokens + ["[SEP]"]
  if(triples is not None):
    ttokens = tokenizer.tokenize(triples)
    stokens = stokens+ttokens
    stokens = stokens[:MAX_LEN]
    stokens = stokens+["[SEP]"]
    
  #print (stokens)
 
  ids = get_ids(stokens, tokenizer, max_seq_len)
  masks = get_masks(stokens, max_seq_len)
  segments = get_segments(stokens, max_seq_len)

  

  c_masks = []


  if(custom_masks!=None):
    train = pd.DataFrame()
    train['topics']=custom_masks[0]
    topics = get_all_topics(train)
    #topics=custom_masks[0].iloc[0]
    entities = custom_masks[1].iloc[0]
    #print (entities)
    triples = custom_masks[2]
    c_mask=1
    prev_token = ""
    cnt=0
    for c_tokens in stokens:
      print (c_tokens)
      if(c_tokens[0]!="#"):
        prev_token = c_tokens
        cnt=0
        if(c_tokens in topics):
          c_mask = 1
        elif(c_tokens in entities):
          c_mask = 1
        elif(c_tokens in triples):
          c_mask = 1
        else:
          c_mask = 0
        c_masks.append(c_mask)
        print ("Direct:"+c_tokens)
        if(c_mask==1):
          print (c_tokens+" is included")
        else:
          print (c_tokens+" is not included")
      else:
        t = prev_token+c_tokens[2:]
        prev_token=t
        cnt+=1
        if(t in topics):
          c_mask = 1
        elif(t in entities):
          c_mask = 1
        elif(t in triples):
          c_mask = 1
        else:
          c_mask = 0
        c_masks[-cnt:]=[c_mask]*cnt
        if(c_mask==1):
          print (t+" is included")
        else:
          print (t+" is not included")
    print (len(c_masks))
    print (c_masks)
    

  return ids,masks,segments

def create_input_array(sentences, max_seq_len, tokenizer,triples=None, custom_masks=None):
  input_ids, input_masks, input_segments = [], [], []
  counter=0
  if(custom_masks!=None):
    all_triples=custom_masks[2]
    all_entities=custom_masks[1]

  for sentence in tqdm(sentences,position=0, leave=True):
    if(custom_masks!=None):
      print ("Position: "+str(counter))
      custom_masks[2]=all_triples[counter]
      custom_masks[1]=all_entities[counter]
    if(triples is not None):
      ids,masks,segments=create_single_input(sentence,max_seq_len-2, max_seq_len, tokenizer, triples.iloc[counter], custom_masks)
    else:
      ids,masks,segments=create_single_input(sentence,max_seq_len-2, max_seq_len, tokenizer)
  
    counter+=1
    input_ids.append(ids)
    input_masks.append(masks)
    input_segments.append(segments)

  return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]