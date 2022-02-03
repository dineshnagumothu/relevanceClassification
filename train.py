import pandas as pd
import sys
import numpy as np

from classifier_models import model_making, compute_metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import resample

import argparse

from util import MAX_NB_WORDS, EMBEDDING_DIM, TEXT_LEN, TOPICS_LEN, ENTITIES_LEN, TRIPLES_LEN, LABELS, BERT_SEQ_LEN
from util import SENTENCE_EMBEDDINGS, GLOVE_EMBEDDINGS, BERT
from util import triple_col, sent_col
from util import compute_embedding_matrix, typeConv
from util import triple_to_text

from util import exclude_long_docs, calc_doc_lengths

from util import convert_multi_class ##Class conversion for Reuters

from bert_util import create_input_array, BERT_LAYER_LINK

import bert
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub

train_sampled = pd.DataFrame()
val = pd.DataFrame()
test = pd.DataFrame()


def generate_model(epochs, batch_size, topics=False, entities=False, triples=False, text=False, embedding=GLOVE_EMBEDDINGS, name=""):
  """
  This function creates the input suitable to the model selected, creates a model with random weights and does the training for relevance classification.
  """
  count=0
  name+="_"
  if (text==True):
    count+=1
    name+='Text'
  if (topics==True):
    count+=1
    if(count>1):
      name+='_'
    name+='Topics'
  if (entities==True):
    count+=1
    if(count>1):
      name+='_'
    name+='Entities'
  if (triples==True):
    count+=1
    if(count>1):
      name+='_'
    name+='Triples'
  name=name+"_"+embedding
  
  #name+='_'+str(i)

  print (name)
  
  #logdir = os.path.join("EH_logs", name)
  #tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
  if (entities==True and embedding==SENTENCE_EMBEDDINGS):
    print ("Named Entities will use GloVe Embeddings")


  embedding_matrix= compute_embedding_matrix(tokenizer)
  train_inputs = []
  val_inputs = []
  test_inputs = []


  if(embedding==BERT):
    #downloading bert layer from TFhub
    bert_layer=hub.KerasLayer(BERT_LAYER_LINK,trainable=False, name="bert_layer")
    #initialise the tokenizer
    FullTokenizer=bert.bert_tokenization.FullTokenizer
    #load the vocab file
    vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
    #convert to lower case
    do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()

    #create tokenizer with vocab file and lower case
    bert_tokenizer=FullTokenizer(vocab_file,do_lower_case)

  if (text):
    if(embedding==GLOVE_EMBEDDINGS):
      train_text_inputs=train_sampled['text'].values
      train_text_inputs = tokenizer.texts_to_sequences(train_text_inputs)
      train_text_inputs = pad_sequences(train_text_inputs, maxlen=TEXT_LEN)
      train_text_inputs = typeConv(train_text_inputs)
      train_inputs.append(train_text_inputs)

      val_text_inputs=val['text'].values
      val_text_inputs = tokenizer.texts_to_sequences(val_text_inputs)
      val_text_inputs = pad_sequences(val_text_inputs, maxlen=TEXT_LEN)
      val_text_inputs = typeConv(val_text_inputs)
      val_inputs.append(val_text_inputs)

      test_text_inputs=test['text'].values
      test_text_inputs = tokenizer.texts_to_sequences(test_text_inputs)
      test_text_inputs = pad_sequences(test_text_inputs, maxlen=TEXT_LEN)
      test_text_inputs = typeConv(test_text_inputs)
      test_inputs.append(test_text_inputs)

    elif (embedding==SENTENCE_EMBEDDINGS):
      train_sents_inputs=train_sampled['sent_embeddings'].values
      train_sents_inputs = typeConv(train_sents_inputs)
      train_inputs.append(train_sents_inputs)
      val_sents_inputs=val['sent_embeddings'].values
      val_sents_inputs = typeConv(val_sents_inputs)
      val_inputs.append(val_sents_inputs)
      test_sents_inputs=test['sent_embeddings'].values
      test_sents_inputs = typeConv(test_sents_inputs)
      test_inputs.append(test_sents_inputs)
    
    elif (embedding==BERT):
      #sys.exit()
      train_inputs.append(create_input_array(train_sampled['text'], BERT_SEQ_LEN, bert_tokenizer))
      val_inputs.append(create_input_array(val['text'], BERT_SEQ_LEN, bert_tokenizer))
      test_inputs.append(create_input_array(test['text'], BERT_SEQ_LEN, bert_tokenizer))

  if (topics):
    train_topic_inputs=train_sampled['topic_probs'].values
    train_topic_inputs = typeConv(train_topic_inputs)
    train_inputs.append(train_topic_inputs)

    val_topic_inputs=val['topic_probs'].values
    val_topic_inputs = typeConv(val_topic_inputs)
    val_inputs.append(val_topic_inputs)

    test_topic_inputs=test['topic_probs'].values
    test_topic_inputs = typeConv(test_topic_inputs)
    test_inputs.append(test_topic_inputs)
  if (entities):
    train_entity_inputs=train_sampled['entities_text'].values
    train_entity_inputs = tokenizer.texts_to_sequences(train_entity_inputs)
    train_entity_inputs = pad_sequences(train_entity_inputs, maxlen=ENTITIES_LEN)
    train_entity_inputs = typeConv(train_entity_inputs)
    train_inputs.append(train_entity_inputs)

    val_entity_inputs=val['entities_text'].values
    val_entity_inputs = tokenizer.texts_to_sequences(val_entity_inputs)
    val_entity_inputs = pad_sequences(val_entity_inputs, maxlen=ENTITIES_LEN)
    val_entity_inputs = typeConv(val_entity_inputs)
    val_inputs.append(val_entity_inputs)

    test_entity_inputs=test['entities_text'].values
    test_entity_inputs = tokenizer.texts_to_sequences(test_entity_inputs)
    test_entity_inputs = pad_sequences(test_entity_inputs, maxlen=ENTITIES_LEN)
    test_entity_inputs = typeConv(test_entity_inputs)
    test_inputs.append(test_entity_inputs)

  if (triples):
    if (embedding==GLOVE_EMBEDDINGS):
      train_triple_inputs=train_sampled[triple_col].values
      train_triple_inputs = tokenizer.texts_to_sequences(train_triple_inputs)
      train_triple_inputs = pad_sequences(train_triple_inputs, maxlen=TRIPLES_LEN)
      train_triple_inputs = typeConv(train_triple_inputs)
      train_inputs.append(train_triple_inputs)

      val_triple_inputs=val[triple_col].values
      val_triple_inputs = tokenizer.texts_to_sequences(val_triple_inputs)
      val_triple_inputs = pad_sequences(val_triple_inputs, maxlen=TRIPLES_LEN)
      val_triple_inputs = typeConv(val_triple_inputs)
      val_inputs.append(val_triple_inputs)

      test_triple_inputs=test[triple_col].values
      test_triple_inputs = tokenizer.texts_to_sequences(test_triple_inputs)
      test_triple_inputs = pad_sequences(test_triple_inputs, maxlen=TRIPLES_LEN)
      test_triple_inputs = typeConv(test_triple_inputs)
      test_inputs.append(test_triple_inputs)
    elif(embedding==SENTENCE_EMBEDDINGS):
      train_sents_inputs=train_sampled['triple_sent_embeddings'].values
      train_sents_inputs = typeConv(train_sents_inputs)
      train_inputs.append(train_sents_inputs)
      val_sents_inputs=val['triple_sent_embeddings'].values
      val_sents_inputs = typeConv(val_sents_inputs)
      val_inputs.append(val_sents_inputs)
      test_sents_inputs=test['triple_sent_embeddings'].values
      test_sents_inputs = typeConv(test_sents_inputs)
      test_inputs.append(test_sents_inputs)
    elif (embedding==BERT):
      train_inputs.append(create_input_array(train_sampled[triple_col], BERT_SEQ_LEN, bert_tokenizer))
      val_inputs.append(create_input_array(val[triple_col], BERT_SEQ_LEN, bert_tokenizer))
      test_inputs.append(create_input_array(test[triple_col], BERT_SEQ_LEN, bert_tokenizer))
      #train_inputs.append(create_input_array(train_sampled['text'], BERT_SEQ_LEN, bert_tokenizer, triples=train_sampled[triple_col]))
      #val_inputs.append(create_input_array(val['text'], BERT_SEQ_LEN, bert_tokenizer, triples=train_sampled[triple_col]))
      #test_inputs.append(create_input_array(test['text'], BERT_SEQ_LEN, bert_tokenizer, triples=train_sampled[triple_col]))
      '''
      custom_tokens=[train_sampled['topics']]
      custom_tokens.append(train_sampled['entities'])
      x = []
      for triple_text in train_sampled[sent_col]:
        x.append(triple_text)
      custom_tokens.append(x)

      train_inputs.append(create_input_array(train_sampled['text'], BERT_SEQ_LEN, bert_tokenizer,custom_tokens))
      val_inputs.append(create_input_array(val['text'], BERT_SEQ_LEN, bert_tokenizer))
      test_inputs.append(create_input_array(test['text'], BERT_SEQ_LEN, bert_tokenizer))
      '''
        




      
  train_result = tf.keras.utils.to_categorical(train_sampled['relevance'], num_classes=LABELS)
  val_result = tf.keras.utils.to_categorical(val['relevance'], num_classes=LABELS)
  test_result = tf.keras.utils.to_categorical(test['relevance'], num_classes=LABELS)

  model=model_making(count, embedding_matrix,topics=topics, entities=entities, triples=triples, text=text, fine_tune=True, embedding=embedding, num_labels=LABELS)
  
  if(count==1):
    train_inputs=train_inputs[0]
    val_inputs=val_inputs[0]
    test_inputs=test_inputs[0]
  
  if(embedding==SENTENCE_EMBEDDINGS):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=10, restore_best_weights=True)
    model.fit(train_inputs,train_sampled['relevance'],epochs=epochs,batch_size=batch_size,validation_data=(val_inputs, val['relevance']), callbacks=[es])
  else:
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=3, restore_best_weights=True)
    model.fit(train_inputs,train_sampled['relevance'],epochs=epochs,batch_size=batch_size,validation_data=(val_inputs, val['relevance']), callbacks=[es])
  tf.keras.utils.plot_model(model, to_file='model_plots/'+name+'.png', show_shapes=True, show_layer_names=True)
  
  accr = model.evaluate(test_inputs, test['relevance'])
  metrics, matrix=compute_metrics(model, test_inputs, test_result, name=name)
  if(LABELS==2): 
    print('%f\t%f\t%f\t%f\t%f\t%f' %(metrics[0],metrics[1],metrics[2],metrics[3],metrics[4],metrics[5]))
  else:
    print('%f\t%f\t%f\t%f' %(metrics[0],metrics[1],metrics[2],metrics[3]))
  print (matrix)
  return model

if __name__ == "__main__":
  """
  This is the main function.
  To run this script, you have to provide three inputs. 
  1.Name of the dataset with --dataset argument, check readme file for the list of available datasets
  2.Model to be execute with --model argument, check readme file for the list of available models
  3.Emmbedding or Encoding that models use with --embedding argument, check readme file for the list of available embeddings
  """
  parser = argparse.ArgumentParser()   
  parser.add_argument('--dataset', required=True, help="Dataset required")
  parser.add_argument('--model', required=True, help="Choose a model name")
  parser.add_argument('--embedding', default=GLOVE_EMBEDDINGS,help="Choose an embedding")
  
  args = parser.parse_args()

  if args.dataset=='energyhub':
    filename = 'EH_infersents'
    dir_name = 'Energy Hub'
  elif args.dataset == 'reuters':
    filename = 'Reuters_infersents'
    #filename = 'reuters_full'
    dir_name = 'Reuters'
  elif args.dataset == '20ng':
    filename = '20ng'
    dir_name = '20ng'
  else:
    print ("Wrong dataset name")
    sys.exit()

  model_name = args.model

  print(f'{args.embedding} selected')
  print(f'{args.dataset} selected')
  print(f'{args.model} selected')
  name=args.dataset


  print ("Reading Data")

  train= pd.read_json(r"data/"+filename+"_train_probs.json")
  val= pd.read_json(r"data/"+filename+"_val_probs.json")
  test= pd.read_json(r"data/"+filename+"_test_probs.json")

  tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~', lower=True)

  '''
  train['triple_text'] = triple_to_text(train['triples'])
  val['triple_text'] = triple_to_text(val['triples'])
  test['triple_text'] = triple_to_text(test['triples'])
  '''

  token_text=train[triple_col].values.tolist()
  token_text.extend(val[triple_col].values)
  token_text.extend(test[triple_col].values.tolist())
  
  token_text.extend(train['text'].values)
  token_text.extend(val['text'].values)
  token_text.extend(test['text'].values)

  token_text.extend(train['topics_text'].values)
  token_text.extend(val['topics_text'].values)
  token_text.extend(test['topics_text'].values)

  token_text.extend(train['entities_text'].values)
  token_text.extend(val['entities_text'].values)
  token_text.extend(test['entities_text'].values)

  tokenizer.fit_on_texts(token_text)

  word_index = tokenizer.word_index

  if(filename=='reuters_full'):
    train,val,test = convert_multi_class(train, val, test)
    print(train['relevance'])
    all_docs = pd.concat([train, val, test], ignore_index=True)
    rel_count = all_docs.relevance.value_counts()
  else:
    rel_count = train.relevance.value_counts()
  LABELS = len(rel_count)
  print ("Number of labels", LABELS)

  if LABELS == 2:
    sample_size=0
    if(rel_count[0]>rel_count[1]):
      sample_size=rel_count[1]
    else:
      sample_size=rel_count[0]

    nd_majority = train[train.relevance==1]
    nd_minority = train[train.relevance==0]
    nd_majority_downsampled = resample(nd_majority, replace=False, n_samples=sample_size, random_state=2020) 
    train_sampled = pd.concat([nd_majority_downsampled, nd_minority])
  else:
    train_sampled=train

  EPOCHS=300
  BATCH_SIZE=32
  if(args.embedding==BERT):
    EPOCHS=3
    BATCH_SIZE = 4
    #train_sampled = exclude_long_docs(train_sampled)
    #val = exclude_long_docs(val)
    #test = exclude_long_docs(test)
    #doc_lengths,avg_length = calc_doc_lengths(train_sampled['text'])
    #doc_lengths.sort(reverse=True)
    #print (doc_lengths[:300])
    #sys.exit()

  if (model_name=='text'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,text=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='topics'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,topics=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='entities'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,entities=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='triples'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,triples=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='text_triples'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,text=True, triples=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='text_topics'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,text=True, topics=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='text_entities'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,text=True, entities=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='text_topics_entities'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,text=True, topics=True, entities=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='text_topics_triples'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,text=True, triples=True, topics=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='text_entities_triples'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,text=True, triples=True, entities=True, embedding=args.embedding, name=args.dataset)
  elif (model_name=='text_topics_entities_triples'):
    model = generate_model(epochs=EPOCHS, batch_size=BATCH_SIZE,text=True, entities=True, triples=True, topics=True, embedding=args.embedding, name=args.dataset)

  else:
    print ("Wrong model selected")

