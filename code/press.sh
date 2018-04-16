import os
import numpy as np
# create dictionary first, using google list
if not os.path.exists("embed.dat"):
    print("Catching word embeddings in memmapped format...")
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True)
    fp = np.memmap("embed.dat", dtype=np.double, mode="w+", shape=wv.syn0norm.shape)
    fp[:] = wv.syn0norm[:]
    with open("embed.vocab", "w") as f:
        for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
            print(w, file=f)
    del fp, wv

W = np.memmap("embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))

with open("embed.vocab") as f:
    vocab_list = map(str.strip, f.readlines())
vocab_dict = {w: k for k, w in enumerate(vocab_list)}


# now load the documents we want to examine [google]
indexDocs = range(1,13)
docs = []
for i in indexDocs:
    namefile ="news/g" + str(i) + ".txt"
    f = open(namefile)
    d = f.read()
    docs.append(d)

# now load the documents we want to examine [apple]
# NOTE: All docs stored in "docs"
indexDocs = range(1,7)
for i in indexDocs:
    namefile ="news/a" + str(i) + ".txt"
    f = open(namefile)
    d = f.read()
    docs.append(d)

# printing docs
i = 1
for doc in docs:
    print("*"*20, " DOC # ", i, "*"*20)
    print(doc)
    print("*"*20, " EN OF DOC # ", i, "*"*20)
    print("="*50,"\n")
    i = i + 1


########################################
# DOCS PREPROCESSING         ###########
########################################
# remove stopwords
stopWords = open("stopwordsEN.txt")
ss = stopWords.read()
stopList = ss.split()
#print(stopList)
tt =[[word for word in text.lower().split() if word not in stopList] for text in docs]

# documents preprocessing
from collections import defaultdict
frequency = defaultdict(int)
for doc in tt:
    for token in doc:
        frequency[token] += 1

i = 0
for doc in tt:
    i += 1
    tot = 0
    elim = 0
    print("*"*10, "Doc # ", i ," :: ")
    for token in doc:
        tot +=1
        if frequency[token] == 1:
            elim += 1
    print("Based on Words Freq Doc # ", i, " eliminated ", elim, " words out of ", tot)

docs = [[token for token in doc if frequency[token] > 1] for doc in tt]
# print(docs)

# eliminate words that are not in the vocabulary
# let's double check
i = 0
for doc in docs:
    i += 1
    tot = 0
    elim = 0
    for token in doc:
        tot += 1
        if token not in vocab_dict:
            elim += 1
            # print(token , " is NOT in vocabulary")
    print("Based on Vocab Doc # ", i, " eliminated ", elim, " words out of ", tot)

docs = [[token for token in doc if token in vocab_dict] for doc in docs]
input("Are all words in vocab?")

# rejoin documents
docs = [[" ".join(token for token in doc)] for doc in docs]
i = 0
for doc in docs:
    i += 1
    print("-"*20, "DOC # ", i, " :: ")
    print(doc)

print("END OF DOCS from FACTIVA ")
print("*"*50)
input("Factiva ok?")


# load dictionary (borrowed from google)
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

W = np.memmap("embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with open("embed.vocab") as f:
    vocab_list = map(str.strip, f.readlines())

vocab_dict = {w: k for k, w in enumerate(vocab_list)}

# now we have dk10 as principal document and docs are all the news
# strange access to docs, it is a list...
# vect = CountVectorizer(stop_words="english").fit([dk10, docs[0][0], docs[1][0],
# docs[2][0], docs[3][0], docs[4][0], docs[5][0], doc[6][0], doc[7][0], doc[8][0]])
# vect = CountVectorizer(stop_words="english").fit([docs[0][0], docs[1][0]])

# vect = CountVectorizer(stop_words="english").fit([docs[0][0], docs[1][0],
# docs[2][0], docs[3][0], docs[4][0], docs[5][0], docs[6][0], docs[7][0],
# docs[8][0], docs[9][0], docs[10][0], docs[11][0]])

allDocs = []
for doc in docs:
    allDocs.append(doc[0]) 

vect = CountVectorizer(stop_words="english").fit(allDocs)
print("Features = ", ",".join(vect.get_feature_names()))









