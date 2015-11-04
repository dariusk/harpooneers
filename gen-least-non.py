import sys
import gensim
import re
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
alldocs = []  # will hold all docs in original order
with open('/home/dariusk/doc2vec/gutenberg.txt') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[0:]
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        split = ['train','test','extra','extra'][line_no//50000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//25000] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

import numpy as np
import statsmodels.api as sm
from random import sample

# for timing
from contextlib import contextmanager
from timeit import default_timer
import time 

import random

#5414 lines in P&P
model = Doc2Vec.load('/home/dariusk/doc2vec/gutenberg.doc2vec')
doc_id = int(sys.argv[1])
sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
#print(u'TARGET (%d): %s\n' % (doc_id, ' '.join(alldocs[doc_id].words)))
#print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)

for y in range(len(sims) - 1-10,len(sims) - 1):
    print('%s {{%s}}' % (' '.join(alldocs[sims[y][0]].words).encode('utf8'), sims[y][0]))
