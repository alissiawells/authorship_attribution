### Authorship attribution for short messages in Russian (1 tweet = 270 chars)
Authorship attribution has a long history, beginning from literature studies and criminal law. Nowadays the method is used in forensic linguistics to identify the authors of certain messages in DLP systems.
One of the approaches to the closed-set authorship attribution is to decompose the task on *feature engineering* (using doc2vec embeddings, for instance) + *multiclass classification* (for example, CNN based on Keras). 

Features: semantic of words, punctuation, different types of smiles, URLs, hashtags.
Without spellchecker as typos can contain useful lexical features.
Optional lemmatization (which is more approriate for Russian than stemming).

[Upload a corpus](https://drive.google.com/file/d/1O-wVcsJ-d4IgjzdI7qqHdfECOyu2jFBH/view?usp=sharing) gathered from twitter (or prepare your own) & save to the directory *dataset*. 

Format: *authorID*.txt; 1 author = 1 doc, one tweet per line.

Prepare.ipynb - data preprocessing (clean data & split it on train (75%) and test (25%) datasets). 

Ensure that all the requirements are installed, then run:
```
$ python doc2vec.py
$ python cnn.py
```

