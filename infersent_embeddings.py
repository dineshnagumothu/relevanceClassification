# import stuff
#%load_ext autoreload
#%autoreload 2
#%matplotlib inline

from random import randint

import numpy as np
import torch

# Load model
from models import InferSent
model_version = 1
MODEL_PATH = "encoder/infersent%s.pkl" % model_version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
model = InferSent(params_model)
model.load_state_dict(torch.load(MODEL_PATH))

# Keep it on CPU or put it on GPU
use_cuda = True
model = model.cuda() if use_cuda else model

# If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
model.set_w2v_path(W2V_PATH)

# Load embeddings of K most frequent words
model.build_vocab_k_words(K=100000)

def generate_embeddings(news_data, col_name='text'):
  docs=[]
  for doc in news_data[col_name]:
    if col_name=='openie_triples':
      text=''
      for triple in doc:
        text+=triple[0]+" "+triple[1]+" "+triple[2]+". "
      text = text.strip()
    
    else:
      text=doc  
    sents=[]
    for sent in text.split('.'):
      sents.append(sent)
    docs.append(sents)
  print (docs[:1])
  print ("Number of docs:", len(docs))

  doc_embeddings=[]
  for doc in docs:
    embeddings = model.encode(doc, bsize=128, tokenize=False, verbose=True)
    doc_embeddings.append(embeddings)
    #print('nb sentences encoded : {0}'.format(len(embeddings)))
  print('nb Docs encoded : {0}'.format(len(doc_embeddings)))

  new_embeddings=[]
  for emb in doc_embeddings:
    adder=np.zeros(4096)
    for sent in emb:
      adder+=sent
    adder = adder/len(emb)
    new_embeddings.append(adder)

  return new_embeddings


