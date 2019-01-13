### Authorship attribution for short messages (1 tweet = 270 chars)
One of approaches to the closed-set authorship attribution task (feature enginnering with doc2vec embeddings + multiclass classification using CNN based on Keras).
Features: semantic of words, punctuation, different types of smiles, URLs, hashtags.
Without spellchecker as typos can contain useful lexical features.
Optional lemmatization (which is more approriate for Russian than stemming).

[Upload a corpus](https://drive.google.com/file/d/1O-wVcsJ-d4IgjzdI7qqHdfECOyu2jFBH/view?usp=sharing) gathered from twitter (or prepare your own) & save to directory *dataset*. Format: <authorID>.txt; 1 author = 1 doc, 1 column, one tweet per line

Prepare.ipynb - data preprocessing (clean data & split it on train (75%) and test (25%) datasets). Ensure that all the requirements are installed, then run:
```
$ python doc2vec.py
$ python cnn.py
```

