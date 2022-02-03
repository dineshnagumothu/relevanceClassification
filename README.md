# quokka

<ol>
<li>

<b> Download data </b> 

Download files from these links and copy them to the data directory

<b>Energy Hub</b>

    Energy Hub Training set - 

    Energy Hub Validation set - 

    Energy Hub Test set - 

<b> Reuters </b>

    Reuters Training set - 

    Reuters Validation set - 

    Retuers Test set - 

</li>
<li>    
<b> Downloading Necessary Packages </b>

  <ul>
    <li> Download NLTK stopwords using

  ```
  import nltk
  
  nltk.download('stopwords')
  ```
  </li>
  <li> Download Mallet from <a href="http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip">here</a>. Unzip and copy it to the directory.

  If you use Google Colab:
  
  ```
  !wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
  !unzip mallet-2.0.8.zip
  ```
  </li>
  
  <li>Download GloVe embeddings from <a href="https://nlp.stanford.edu/data/wordvecs/glove.6B.zip">here</a>. Unzip and copy it to the directory.
    
   If you use Google Colab:
   
   ```
   !wget https://nlp.stanford.edu/data/wordvecs/glove.6B.zip
   !unzip glove*.zip
   ```
   </li>
   <li>
   If you choose to build triples with Stanford OpenIE
    
   Install Stanza: `pip install stanza`

   Run the following code from python command shell 
   
   To run the Python Shell, open the command prompt or power shell on Windows and terminal window on mac, write python and press enter (or) you can use Jupyter Notebook (you can follow the same with Google Colab).
  
  Change corenlp_dir to a physical path on your machine. This will be the corenlp installation directory.
  
   ```
   import stanza

   # Download the Stanford CoreNLP package with Stanza's installation command
   # This'll take several minutes, depending on the network speed
   
   corenlp_dir = 'path/to/install/corenlp'
   stanza.install_corenlp(dir=corenlp_dir)

   # Set the CORENLP_HOME environment variable to point to the installation location
   
   import os
   os.environ["CORENLP_HOME"] = corenlp_dir
   ```
   
   </li>
  </ul>

  <li>    
<b> Build Topic-Entity Triples </b>

  This step involves
  <ul>
    <li>Training a Topic Modeler over the corpus</li>
    <li>Extracting Named-Entities using spaCy</li>
    <li>Building Triples using Dependency parser and POS tagger</li>
    <li>Apply Topic Entity Filter over these triples</li>
  </ul>
  
  Run the following python file.

  `python data_preprocess.py <dataset>` 
  
  Change `<dataset>` to "energy hub", "reuters" or "20ng" to select the corpus.
  
  </li>
  
  <li>
  <b> Training Models </b>
  
 Run the following python file.

  `python train.py --dataset=<dataset> --model=<model> --embedding=<embedding>`
  
  Change `<dataset>` to "energy hub", "reuters" or "20ng" to select the corpus.
  
  Change `<model>` to the following options
  <ul>
    <li>text - for GloVe based text model</li>
    <li>topics - To use topic distributions</li>
    <li>entites - To use Glove-enriched named entities</li>
    <li>triples - To use Glove-enriched triples</li>
    <li>text_topics - To use text and topic distributions</li>
    <li>text_entities - To use text(GloVe) and Named Entities(GloVe)</li>
    <li>text_triples - To use text(GloVe) and triples(GloVe)</li>
    <li>text_topics_entities - To use text(GloVe), topic distributions and Named Entities(GloVe)</li>
    <li>text_topics_triples - To use text(GloVe), topic distributions and triples(GloVe)</li>
    <li>text_entities_triples - To use text(GloVe), Named Entities(GloVe) and triples(GloVe)</li>
    <li>text_topics_entities_triples - To use text(GloVe), topic distributions, Named Entities(GloVe) and triples(GloVe)</li>
  </ul>
  
  Change `<embedding>` to "glove", "sentences" or "bert" to select the vector representation.
  
  </li>
  </ol>
