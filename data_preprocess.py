import pandas as pd
import numpy as np
from TopicModelling_v3 import ldatopicmodel, fetch_topics
from TripleFormation import getTriples


import sys

def get_terms_from_keywords(article_topics):
  topics_doc_list=[]
  for keywords in article_topics['Keywords']:
      keywords=keywords.split(',')
      topics_doc_list.append(keywords)
  return topics_doc_list

def topics_to_text(news_data):
  topics_text=[]
  for topics in news_data['topics'].values:
    text=""
    for topic in topics:
      text+=topic+" "
    topics_text.append(text)
  return topics_text

def entities_to_text(news_data):
  entities_text=[]
  for entities in news_data['entities'].values:
    text=""
    for entity in entities:
      #entity=entity.replace(" ", "_")
      text+=entity+" "
    entities_text.append(text)
  return entities_text

def triples_to_text(news_data, col_name='triples'):
  triples_text_new=[]
  for triples in news_data[col_name].values:
    text=""
    for triple in triples:
      triple_text=""
      for element in triple:
        triple_text+=element+" "
      triple_text.strip()
      #triple_text=triple_text.replace(" ", "_")
      text+=triple_text+" "
    triples_text_new.append(text)
  return triples_text_new

def get_all_topics(news_data):
  top=[]
  for topics in news_data['topics']:
    for topic in topics:
      topic=topic.replace("_", " ")
      topic=topic.strip()
      if topic not in top:
        top.append(topic)
  return top

def get_stanza_client():
  print ("Starting CORENLP Server")
  # Import client module
  from stanza.server import CoreNLPClient
  # Construct a CoreNLPClient with some basic annotators, a memory allocation of 4GB, and port number 9001
  client = CoreNLPClient(
      annotators=['openie'], 
      memory='4G', 
      endpoint='http://localhost:9001',
      be_quiet=True)
  print(client)

  # Start the background server and wait for some time
  # Note that in practice this is totally optional, as by default the server will be started when the first annotation is performed
  client.start()
  import time; 
  time.sleep(10)
  return client;


if __name__ == "__main__":
  if sys.argv[1]=='energyhub':
    filename = 'EH_infersents'
  elif sys.argv[1] == 'reuters':
    #filename = 'Reuters_infersents'
    filename = 'reuters_full'
  elif sys.argv[1] == '20ng':
    filename = '20ng'
  else:
    print ("Wrong dataset name")
    sys.exit()

  print ("Reading Data")

  train= pd.read_json(r"data/"+filename+"_train.json")
  val= pd.read_json(r"data/"+filename+"_val.json")
  test= pd.read_json(r"data/"+filename+"_test.json")

  #train= pd.read_json(r"data/"+filename+"_train_probs.json")
  #val= pd.read_json(r"data/"+filename+"_val_probs.json")
  #test= pd.read_json(r"data/"+filename+"_test_probs.json")

  if (sys.argv[2]=='sents'):
    from infersent_embeddings import generate_embeddings

    print ("Sentence Embeddings")

    train['sent_embeddings'] = generate_embeddings(train)
    val['sent_embeddings'] = generate_embeddings(val)
    test['sent_embeddings'] = generate_embeddings(test)

  print ("Topic Modelling")
  ##Remember to dowload nltk stopwords
  train_topics, lda_model, bigram_mod, trigram_mod, id2word, doc_probs  = ldatopicmodel(train)
  val_topics, val_probs=fetch_topics(val, bigram_mod, trigram_mod, id2word, lda_model)
  test_topics, test_probs=fetch_topics(test, bigram_mod, trigram_mod, id2word, lda_model)

  train['topic_probs'] = doc_probs
  val['topic_probs'] = val_probs
  test['topic_probs'] = test_probs

  train['topics']= get_terms_from_keywords(train_topics)
  val['topics']= get_terms_from_keywords(val_topics)
  test['topics']= get_terms_from_keywords(test_topics)

  print ("Forming Triples")

  all_topics = get_all_topics(train)

  openie_triples=True

  if(openie_triples):
    client = get_stanza_client()
    print ("Building Open IE Triples")
    train['openie_triples'], train['entities'] = getTriples(train, all_topics, client)
    val['openie_triples'], val['entities'] = getTriples(val, all_topics, client)
    test['openie_triples'], test['entities'] = getTriples(test, all_topics, client)
    # Shut down the background CoreNLP server
    client.stop()
    import time; 
    time.sleep(10)
  else:
    train['triples'] = getTriples(train, all_topics)
    val['triples'] = getTriples(val, all_topics)
    test['triples'] = getTriples(test, all_topics)

  train['topics_text']=topics_to_text(train)
  val['topics_text']=topics_to_text(val)
  test['topics_text']=topics_to_text(test)

  train['entities_text']=entities_to_text(train)
  val['entities_text']=entities_to_text(val)
  test['entities_text']=entities_to_text(test)
  if(openie_triples):
    train['openie_triple_text']=triples_to_text(train, col_name='openie_triples')
    val['openie_triple_text']=triples_to_text(val, col_name='openie_triples')
    test['openie_triple_text']=triples_to_text(test, col_name='openie_triples')
  else:
    train['triple_text']=triples_to_text(train)
    val['triple_text']=triples_to_text(val)
    test['triple_text']=triples_to_text(test)
  if (sys.argv[2]=='sents'):
    train['triple_sent_embeddings'] = generate_embeddings(train, col_name='openie_triples')
    val['triple_sent_embeddings'] = generate_embeddings(val, col_name='openie_triples')
    test['triple_sent_embeddings'] = generate_embeddings(test, col_name='openie_triples')



  print ("Writing to Disk")

  train.to_json(r'data/'+filename+'_train_probs.json', default_handler=str)
  val.to_json(r'data/'+filename+'_val_probs.json', default_handler=str)
  test.to_json(r'data/'+filename+'_test_probs.json', default_handler=str)