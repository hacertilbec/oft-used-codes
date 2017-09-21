# This is how you can create summary of any text with gensim library

from gensim.summarization import summarize

text_file = '...'
text = open(text_file,'r').read()

summary = summarize(text, word_count=20)

# finding pos_tags and removing stopwords with nltk library

import nltk
sentence = "Big Data Techniques Learn online in your own time. The know-how You need to succeed."
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)

# output:
# [('Big', 'NNP'), ('Data', 'NNP'), ('Techniques', 'NNP'), ('Learn', 'NNP'), ('online', 'NN'),
#  ('in', 'IN'), ('your', 'PRP$'), ('own', 'JJ'), ('time', 'NN'), ('.', '.'), ('The', 'DT'),
#  ('know-how', 'NN'), ('You', 'PRP'), ('need', 'VBP'), ('to', 'TO'), ('succeed', 'VB'), ('.', '.')]

from nltk.corpus import stopwords
import string

stop_words = set(stopwords.words('english'))

def remove_stopwords(x):
    sentence = str(x).translate(None, string.punctuation)
    s = sentence.split()
    r = [i for i in s if i not in stop_words]
    return ' '.join(r)
