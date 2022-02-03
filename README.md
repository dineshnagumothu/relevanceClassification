# Linked Data Triples Enhance Document Relevance Classification

This is a Tensorflow based implementation of the document relevance classification systems described in the paper <a href="https://doi.org/10.3390/app11146636">Linked Data Triples Enhance Document Relevance Classification </a>

If you find this work useful, please cite our paper as:
    
    @Article{app11146636,
    AUTHOR = {Nagumothu, Dinesh and Eklund, Peter W. and Ofoghi, Bahadorreza and Bouadjenek, Mohamed Reda},
    TITLE = {Linked Data Triples Enhance Document Relevance Classification},
    JOURNAL = {Applied Sciences},
    VOLUME = {11},
    YEAR = {2021},
    NUMBER = {14},
    ARTICLE-NUMBER = {6636},
    URL = {https://www.mdpi.com/2076-3417/11/14/6636},
    ISSN = {2076-3417},
    DOI = {10.3390/app11146636}
    }

# Instructions

<ol>
<li>

<b> Download data </b> 

Download files from these links and copy them to the data directory

<b>Energy Hub</b>

    Energy Hub Training set - https://drive.google.com/file/d/1-2Rrr4lruYSXNx0r0DUpNzTyRITkFocp/view?usp=sharing

    Energy Hub Validation set - https://drive.google.com/file/d/1-AC0WW2FAjdM09YJ6V58R8u7CMiQn7AL/view?usp=sharing

    Energy Hub Test set - https://drive.google.com/file/d/1-CvtKz8oxtBW5s6xlt-w1icnEkWyafy9/view?usp=sharing

<b> Reuters </b>

    Reuters Training set - https://drive.google.com/file/d/1-3c2Wqn3544AO2GMdHC6rOcAwakzznML/view?usp=sharing

    Reuters Validation set - https://drive.google.com/file/d/1FAruSND8Lh3IGuEpP2MI-OWzq1scRp9Q/view?usp=sharing

    Retuers Test set - https://drive.google.com/file/d/1kTks59QOpMu1e1AqcbpWnykFD37wl_hZ/view?usp=sharing

<b> 20 News Groups </b>

    20 News Groups Training set - https://drive.google.com/file/d/1--yVr6rj_F-brd0cPqOgBRVpnsOaQ8lj/view?usp=sharing

    20 News Groups Validation set - https://drive.google.com/file/d/1-6MrisNQ-aoXA2aHHPT4-OUqDddSG1Kx/view?usp=sharing

    20 News Groups Test set - https://drive.google.com/file/d/1-FCIq69HIfsPrXgdOcnRfzrI5wjjPNTR/view?usp=sharing

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
      <b>Optional Step</b> (If you choose to use sentence embeeddings using InferSent, execute the following commands)
      
        !mkdir GloVe
        !curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
        !unzip GloVe/glove.840B.300d.zip -d GloVe/
        !mkdir fastText
        !curl -Lo fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
        !unzip fastText/crawl-300d-2M.vec.zip -d fastText/
      
        !mkdir encoder
        !curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
        !curl -Lo encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl
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
  
  #Contact
      
  Dinesh Nagumothu (dinesh.nagumothu@deakin.edu.au)
