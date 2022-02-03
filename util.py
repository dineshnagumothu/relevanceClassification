import numpy as np

MAX_NB_WORDS = 50000
EMBEDDING_DIM = 300

TOPICS_LEN = 100
TEXT_LEN = 1000
ENTITIES_LEN = 1000
TRIPLES_LEN = 1000

BERT_SEQ_LEN = 512

SENTENCE_EMBEDDINGS = 'sentences'
GLOVE_EMBEDDINGS = 'glove'
BERT = 'bert'

triple_col = 'openie_triple_text'
#triple_col = 'triple_text'
sent_col = 'sent_embeddings'

LABELS = 2

def compute_embedding_matrix(tokenizer):
  embeddings_index = dict()
  f = open('glove.6B.'+str(EMBEDDING_DIM)+'d.txt', errors='ignore', encoding="utf-8")
  for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
  f.close()

  embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
  for word, index in tokenizer.word_index.items():
    if index > MAX_NB_WORDS - 1:
      break
    else:
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_matrix[index] = embedding_vector
  return embedding_matrix



def typeConv(data):
  X_train_array = []
  for x in data:
    x = np.asarray(x).astype(np.float32)
    X_train_array.append(x)

  X_train_array=np.asarray(X_train_array)
  return X_train_array

def triple_to_text(triple_series):
  triples_text_list = []
  for triples in triple_series:
    triples_text=""
    for triple in triples:
      for element in triple:
        triples_text+=element+" "
    triples_text_list.append(triples_text.strip())
  return triples_text_list

def calc_doc_lengths(texts):
  lengths=[]
  for text in texts:
    lengths.append(len(text.split()))
  #lengths.sort(reverse=True)
  #trunc_num = int(len(lengths)*0.3)
  #print (lengths[:trunc_num])
  #lengths=lengths[trunc_num:]
  return (lengths, sum(lengths)/len(lengths))

def exclude_long_docs(df):
  doc_lengths, avg_length = calc_doc_lengths(df['text'])
  print (avg_length)
  df['text_length']=doc_lengths
  desc_doc_lengths=doc_lengths
  desc_doc_lengths.sort(reverse=True)
  trunc_num = int(len(desc_doc_lengths)*0.1)
  max_length = desc_doc_lengths[:trunc_num][-1]
  max_length=600
  print ("Cut-off length: ", max_length)

  #desc_docs = df.sort_values(['text_length'], ascending=[False])
  #desc_doc_lengths.sort(reverse=True)
  
  df = df.loc[df['text_length'] >= max_length]
  _, avg_length = calc_doc_lengths(df['text'])
  print (avg_length)
  print ("Number of documents: ", len(df))
  return df

def convert_to_single_class(df):
  classes=[]
  for label in df['relevance']:
    classes.append(label[0])

  df['relevance']=classes
  return df

def convert_multi_class(train, val, test):
  train=convert_to_single_class(train)
  val=convert_to_single_class(val)
  test=convert_to_single_class(test)
  count=0
  label_dict={}
  train_labels=[]
  val_labels=[]
  test_labels=[]
  for label in train['relevance']:
    if(label not in label_dict.keys()):
      label_dict[label]=count
      count+=1
    train_labels.append(label_dict[label])
  for label in val['relevance']:
    if(label not in label_dict.keys()):
      label_dict[label]=count
      count+=1
    val_labels.append(label_dict[label])
  for label in test['relevance']:
    if(label not in label_dict.keys()):
      label_dict[label]=count
      count+=1
    test_labels.append(label_dict[label])
  train['relevance']=train_labels
  val['relevance']=val_labels
  test['relevance']=test_labels
  return train, val, test





    