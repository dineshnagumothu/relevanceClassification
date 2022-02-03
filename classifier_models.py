import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow import keras

from bert_util import BERT_LAYER_LINK

import tensorflow as tf
import tensorflow_hub as hub
print("tensorflow version : ", tf.__version__)
print("tensorflow_hub version : ", hub.__version__)

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.utils import resample

from util import MAX_NB_WORDS, EMBEDDING_DIM, TEXT_LEN, TOPICS_LEN, ENTITIES_LEN, TRIPLES_LEN, BERT_SEQ_LEN
from util import SENTENCE_EMBEDDINGS, GLOVE_EMBEDDINGS, BERT

def model_making(count, embedding_matrix, topics=False, entities=False, triples=False, text=False, fine_tune=False, embedding='glove', num_labels=2):
  """
  This function creates the models based on the input provided by the user. This function is usually called from the generate_model function from train.py
  """
  learning_rate = 2e-4
  if(embedding==BERT):
    learning_rate = 2e-5
  mod_out=[]
  mod_in=[]
  dropout_rate = 0.3
  dropout_rate_2 = 0.2
  if (text==True):
    if(embedding==GLOVE_EMBEDDINGS):
      input_text = tf.keras.layers.Input(shape=(TEXT_LEN,), name='input_text')
      m1_layers = tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=fine_tune, name='glove_text_embedding')(input_text)
      m1_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_text_3')(m1_layers)
      m1_layers = tf.keras.layers.Flatten(name='flatten_text')(m1_layers)
      m1_layers = tf.keras.layers.Dense(512, activation='relu', name='dropout_multi_text_5')(m1_layers)
      m1_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_text_2')(m1_layers)
      m1_layers = tf.keras.layers.Dense(100,activation='relu', name='dense_3_text')(m1_layers)
      if(count==1):
        m1_layers = tf.keras.layers.Dense(num_labels, activation='softmax', name='dense_output')(m1_layers)
      model_1 = tf.keras.models.Model(inputs=input_text, outputs=m1_layers, name='texts_model')      
      mod_out.append(model_1.output)
      mod_in.append(input_text)

    elif(embedding==SENTENCE_EMBEDDINGS):
      input_sents = tf.keras.layers.Input(shape=(4096,),name="input_sents")
      m1_layers = tf.keras.layers.Dense(1024, activation='relu', name='dense_1_sents')(input_sents)
      m1_layers = tf.keras.layers.Dropout(dropout_rate)(m1_layers)
      m1_layers = tf.keras.layers.Dense(512, activation="relu")(m1_layers)
      if(count==1):
        m1_layers = tf.keras.layers.Dense(num_labels, activation='softmax', name='dense_output')(m1_layers)
      model_1 = tf.keras.models.Model(inputs=input_sents, outputs=m1_layers, name='sents_model')      
      mod_out.append(model_1.output)
      mod_in.append(input_sents)
    
    elif(embedding==BERT):
      input_word_ids = tf.keras.layers.Input(shape=(BERT_SEQ_LEN,), dtype=tf.int32,name="input_word_ids")
      input_mask = tf.keras.layers.Input(shape=(BERT_SEQ_LEN,), dtype=tf.int32,name="input_mask")
      segment_ids = tf.keras.layers.Input(shape=(BERT_SEQ_LEN,), dtype=tf.int32,name="segment_ids")
      bert_text_inputs = [input_word_ids, input_mask, segment_ids]
      bert_layer=hub.KerasLayer(BERT_LAYER_LINK,trainable=fine_tune, name="bert_layer_text")
      pooled_output, sequence_output = bert_layer(bert_text_inputs)
      m1_layers = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
      m1_layers = tf.keras.layers.Dense(100, activation="relu")(m1_layers)
      if(count==1):
        m1_layers = tf.keras.layers.Dense(num_labels, activation='softmax', name='dense_output')(m1_layers)
      model_1 = tf.keras.models.Model(inputs=bert_text_inputs, outputs=m1_layers, name='bert_text_model')      
      mod_out.append(model_1.output)
      mod_in.append(bert_text_inputs)
  if topics==True:
    input_topics = tf.keras.layers.Input(shape=(TOPICS_LEN,), name='input_topics')
    if(embedding==BERT):
      m2_layers = tf.keras.layers.Dense(100,activation='relu', name='dense_2_topics')(input_topics)
    else:
      m2_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_multi_topic_4')(input_topics)
      m2_layers = tf.keras.layers.Dense(512,activation='relu', name='dense_1_topics')(m2_layers)
      m2_layers = tf.keras.layers.Dense(100,activation='relu', name='dense_2_topics')(m2_layers)
    if(count==1):
      m2_layers = tf.keras.layers.Dense(num_labels, activation='softmax', name='dense_output')(m2_layers)
    model_2 = tf.keras.models.Model(inputs=input_topics, outputs=m2_layers, name='topics_model')
    mod_out.append(model_2.output)
    mod_in.append(input_topics)
  if entities==True:
    input_entities = tf.keras.layers.Input(shape=(ENTITIES_LEN,), name='input_entities')
    m3_layers = tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=ENTITIES_LEN, weights=[embedding_matrix], trainable=fine_tune, name='glove_entity_embedding')(input_entities)        
    m3_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_1_entities')(m3_layers)
    m3_layers = tf.keras.layers.Flatten()(m3_layers)
    m3_layers = tf.keras.layers.Dropout(dropout_rate)(m3_layers)
    m3_layers = tf.keras.layers.Dense(512,activation='relu', name='dense_3_entities_2')(m3_layers)
    m3_layers = tf.keras.layers.Dropout(dropout_rate)(m3_layers)
    m3_layers = tf.keras.layers.Dense(100,activation='relu', name='dense_3_entities_4')(m3_layers)
    if(count==1):
      m3_layers = tf.keras.layers.Dense(num_labels, activation='softmax', name='dense_output')(m3_layers)
    model_3 = tf.keras.models.Model(inputs=input_entities, outputs=m3_layers)
    mod_out.append(model_3.output)
    mod_in.append(input_entities)
  if triples==True:
    
    if(embedding==SENTENCE_EMBEDDINGS):
      input_triples = tf.keras.layers.Input(shape=(4096,),name="input_triples")
      m4_layers = tf.keras.layers.Dense(100,activation='relu', name='dense_4_triples_1')(input_triples)
      #m4_layers = tf.keras.layers.Dropout(dropout_rate)(m4_layers)
      m4_layers = tf.keras.layers.Dense(10,activation='relu', name='dense_4_triples_3')(m4_layers)
      if(count==1):
        m4_layers = tf.keras.layers.Dense(num_labels, activation='softmax', name='dense_output')(m4_layers)
      model_4 = tf.keras.models.Model(inputs=input_triples, outputs=m4_layers)
      mod_out.append(model_4.output)
      mod_in.append(input_triples)
    elif(embedding==GLOVE_EMBEDDINGS):
      input_triples = tf.keras.layers.Input(shape=(TRIPLES_LEN,), name='input_triples')
      m4_layers = tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=fine_tune, name='glove_triple_embedding')(input_triples)
      #m4_layers= tf.keras.layers.Conv1D(32, 9, activation='relu', name='conv1d_1_triples')(m4_layers)
      #m4_layers = tf.keras.layers.MaxPooling1D(4, name='maxpool1d_1_triples')(m4_layers)
      #m4_layers = tf.keras.layers.GlobalAveragePooling1D()(m4_layers)
      m4_layers = tf.keras.layers.Dropout(dropout_rate, name='dropout_1_triples')(m4_layers)
      m4_layers = tf.keras.layers.Flatten()(m4_layers)
      m4_layers = tf.keras.layers.Dense(512,activation='relu', name='dense_4_triples_1')(m4_layers)
      m4_layers = tf.keras.layers.Dropout(dropout_rate)(m4_layers)
      m4_layers = tf.keras.layers.Dense(100,activation='relu', name='dense_4_triples_3')(m4_layers)
      if(count==1):
        m4_layers = tf.keras.layers.Dense(num_labels, activation='softmax', name='dense_output')(m4_layers)
      model_4 = tf.keras.models.Model(inputs=input_triples, outputs=m4_layers)
      mod_out.append(model_4.output)
      mod_in.append(input_triples)

    elif (embedding==BERT):
      input_word_ids = tf.keras.layers.Input(shape=(BERT_SEQ_LEN,), dtype=tf.int32,name="triple_input_word_ids")
      input_mask = tf.keras.layers.Input(shape=(BERT_SEQ_LEN,), dtype=tf.int32,name="triple_input_mask")
      segment_ids = tf.keras.layers.Input(shape=(BERT_SEQ_LEN,), dtype=tf.int32,name="triple_segment_ids")
      bert_triple_inputs = [input_word_ids, input_mask, segment_ids]
      bert_layer=hub.KerasLayer(BERT_LAYER_LINK,trainable=fine_tune, name="bert_layer_triples")
      pooled_output, sequence_output = bert_layer(bert_triple_inputs)
      m4_layers = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
      m4_layers = tf.keras.layers.Dense(100, activation="relu")(m4_layers)
      if(count==1):
        m4_layers = tf.keras.layers.Dense(num_labels, activation='softmax', name='dense_output')(m4_layers)
      model_4 = tf.keras.models.Model(inputs=bert_triple_inputs, outputs=m4_layers, name='bert_triple_model')      
      mod_out.append(model_4.output)
      mod_in.append(bert_triple_inputs)
  
  if (count>1):
    model_cat = tf.keras.layers.Concatenate()(mod_out)
    #model_cat = tf.keras.layers.Dense(512,activation='relu', name='dense_1_cat')(model_cat)
    #model_cat = tf.keras.layers.Dense(100,activation='relu', name='dense_2_cat')(model_cat)
    #model_cat = tf.keras.layers.Dropout(dropout_rate)(model_cat)
    #model_cat = tf.keras.layers.Dense(100, activation="relu", name='dense_out')(model_cat)
    #if(embedding==GLOVE_EMBEDDINGS):
    #  model_cat = tf.keras.layers.Flatten(name='flatten_layers')(model_cat)
    model_cat = tf.keras.layers.Dense(num_labels, activation='softmax', name='predictions')(model_cat)
    model = tf.keras.models.Model(mod_in, model_cat, name='Model_Multi')
  else:
    if text==True:
      model=model_1
    if topics==True:
      model=model_2 
    if entities==True:
      model=model_3
    if triples==True:
      model=model_4
  #optimiser = tf.keras.optimizers.SGD(learning_rate=learning_rate)
  optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
  #ce = tf.keras.losses.BinaryCrossentropy()
  ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


  model.compile(loss=ce, optimizer=optimiser, metrics=['accuracy'])

  model.summary()
  return model

def compute_metrics(model, X_test, Y_test, name='text'):
    """
    This function measures the model performance by calculating model accuracy, F1 score, precision, recall, AUC.
    In case of a multi class classification - we measure accuracy, F1-Macro, F1-Micro, Weighted F-score
    """
    yhat_probs = model.predict(X_test, verbose=0)
    #yhat_classes = model.predict_classes(X_test, verbose=0)

    Y_test_bin=[]
    text=""

    for y_bin in Y_test:
      index = np.where(y_bin==1)
      Y_test_bin.append(index[0][0])
    #Y_test_bin = Y_test
    
   
    yhat_classes = np.argmax(yhat_probs,axis=1)
    num_labels=len(yhat_probs[0])

    #print (Y_test_bin)
    #print (yhat_classes)

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_test_bin, yhat_classes)
    
    if(num_labels==2):
      #print('|%f|' % accuracy)
      # precision tp / (tp + fp)
      precision = precision_score(Y_test_bin, yhat_classes)
      #print('%f|' % precision)
      # recall: tp / (tp + fn)
      recall = recall_score(Y_test_bin, yhat_classes)
      #print('%f|' % recall)
      # f1: 2 tp / (2 tp + fp + fn)
      f1 = f1_score(Y_test_bin, yhat_classes)
      #print('%f|' % f1)

      # kappa
      kappa = cohen_kappa_score(Y_test_bin, yhat_classes)
      #print('%f|' % kappa)
      # ROC AUC
      auc = roc_auc_score(Y_test, yhat_probs)
      res_text = "Accuracy|Precision|Recall|F1 score|Kappa|ROC AUC|"
      #print (res_text)
      #print('%f\t%f\t%f\t%f\t%f\t%f' %(accuracy,precision,recall,f1,kappa,auc))
      res_text_2=str(accuracy)+"|"+str(precision)+"|"+str(recall)+"|"+str(f1)+"|"+str(kappa)+"|"+str(auc)
      return_metrics=[accuracy, precision, recall, f1, kappa, auc]
    else:
      f1_macro = f1_score(Y_test_bin, yhat_classes, average='macro')
      f1_micro = f1_score(Y_test_bin, yhat_classes, average='micro')
      f1_weighted = f1_score(Y_test_bin, yhat_classes, average='weighted')
      res_text = "Accuracy|F1-Macro|F1-Micro|F1-Weighted|"
      #print (res_text)
      res_text_2=str(accuracy)+"|"+str(f1_macro)+"|"+str(f1_micro)+"|"+str(f1_weighted)
      #print('%f\t%f\t%f\t%f' %(accuracy,f1_macro,f1_micro,f1_weighted))
      return_metrics=[accuracy, f1_macro,f1_micro,f1_weighted]

    # confusion matrix
    matrix = confusion_matrix(Y_test_bin, yhat_classes)
    print(matrix)
    #print (Y_test_bin)
    #print (yhat_classes)
    
    text+=res_text+"\n"
    text+=res_text_2+"\n\n"

    for res in Y_test_bin:  
      text +=str(res)+"," 
    text=text[:-1]
    text=text+"\n"
    for res in yhat_classes:  
      text +=str(res)+"," 
    text=text[:-1]
    text=text+"\n\n"

    if(num_labels==2):
      target_names=['Relevant', 'Irrelevant']
      text+=classification_report(Y_test_bin, yhat_classes, target_names=target_names)

    f = open("results_2/"+name+".txt", "w")
    f.write(text)
    f.close()
    return (return_metrics, matrix)


