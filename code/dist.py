"""
/***************************************************************************
 *   copyright (C) 2018 by Marco Caserta                                   *
 *   marco dot caserta at ie dot edu                                       *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

 Algorithm for the Computation of Distances between documents.

 Author: Marco Caserta (marco dot caserta at ie dot edu)
 Started : 02.02.2017
 Ended   :

 Command line options (see parseCommandLine):
-i inputfile

"""

import sys, getopt
import cplex
from time import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_INFTY = sys.float_info.max

baseName = ""

def vocabularyBuilding():
    '''
    Build embeddings and vocabulary based on Google News set (1.5Gb and 3M
    words and phrases based on 100B words from Google News)
    '''

    # create dictionary first, using google list
    if not os.path.exists("embed.dat"):
        print("Catching word embeddings in memmapped format...")
        from gensim.models import KeyedVectors
        wv = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True)
        wv.init_sims()
        fp = np.memmap("embed.dat", dtype=np.double, mode="w+", shape=wv.syn0norm.shape)
        fp[:] = wv.syn0norm[:]
        with open("embed.vocab", "w") as f:
            for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
                print(w, file=f)
        del fp, wv

    print("Reading embeddings from disk...")

    global W
    W = np.memmap("embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))

    with open("embed.vocab") as f:
        vocab_list = map(str.strip, f.readlines())
    global vocab_dict
    vocab_dict = {w: k for k, w in enumerate(vocab_list)}
    print("Vocabulary built ... ")

# Parse command line
def parseCommandLine(argv):
    global inputfile
    global ftype
    
    try:
        opts, args = getopt.getopt(argv, "ht:i:", ["help","type=", "ifile="])
    except getopt.GetoptError:
        print ("Command Line Erorr. Usage : python cflp.py -t <type> -i\
        <inputfile> ")

        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ("Usage : python cflp.py -t <type> -i <inputfile> ")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--type"):
            ftype = arg



def printSummary(docs, nCompany, prefix):
    print("\n\nmarco caserta (c) 2018 ")
    print("====================================================")
    print("Nr. Docs              : {0:>25d}".format(len(docs)))
    print("Nr. Companies         : {0:>25d}".format(len(nCompany)))
    print("Avg. Docs per Company : {0:>25.2f}".format(np.mean(nCompany)))

    print("====================================================")


    fig, axes = plt.subplots(nrows=1, ncols=3)


    # get length distribution
    byteCount   = []
    wordCount   = []
    for doc in docs:
        byteCount.append(len(doc.encode("utf8")))
        wordCount.append(len(doc.split()))

    # statistics
    meanComp  = np.mean(nCompany)
    stdComp   = np.std(nCompany)
    meanBytes = np.mean(byteCount)
    meanWords = np.mean(wordCount)
    stdBytes = np.std(byteCount)
    stdWords = np.std(wordCount)


    df = pd.DataFrame({
        "bytes": byteCount, 
        "words": wordCount})

    sns.barplot(x=prefix, y=nCompany, ax=axes[0])
    axes[0].set(xlabel="Companies")
    axes[0].set_title(r"[$\mu = $" +
    str("{0:5.2f}".format(meanComp)) + " $\sigma = $" +
    str("{0:5.2f}".format(stdComp)) + "]")

    sns.distplot(df.bytes, hist_kws=dict(edgecolor="gray", linewidth=2),
    hist=True, kde=False, rug=False, bins=10, ax=axes[1])
    axes[1].set(xlabel="Bytes")
    axes[1].set_title(r"[$\mu = $" +
    str("{0:5.2f}".format(meanBytes)) + " $\sigma = $" +
    str("{0:5.2f}".format(stdBytes)) + "]")

    sns.distplot(df.words, hist_kws=dict(edgecolor="gray", linewidth=2),
    hist=True, kde=False, rug=False, bins=10, ax=axes[2])
    axes[2].set(xlabel="Words")
    axes[2].set_title(r"[$\mu = $" +
    str("{0:5.2f}".format(meanWords)) + " $\sigma = $" +
    str("{0:5.2f}".format(stdWords)) + "]")
    sns.plt.suptitle("Distribution Plots")
    #  sns.plt.show()
    sns.plt.savefig("distributionPlots.png")
    print("Distribution Plots saved on disk ('distributionPlots.png')")



def readDocuments(docs, nDocs, baseName, nCompany, prefix):
    # define docs organization
    # docs 1 to 12 --> google --> "g"lt
    # docs 13 to 18 --> apple --> "a"
    # nCompany = [12, 6] --> number of documents for each company
    # prefix = ["g", "a"]

    index = []
    index.append(nCompany[0])
    for i in range(1,len(nCompany)):
        index.append(index[i-1]+nCompany[i])

    for c in range(len(nCompany)):
        for i in range(1,nCompany[c]+1):
            namefile = baseName + prefix[c] + str(i) + ".txt"
            print("Reading file ", namefile, "\n")
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

def docsPreprocessing(docs):
    # 1. remove stopwords
    stopWords = open("stopwordsEN.txt")
    ss = stopWords.read()
    stopList = ss.split()
    tokenizedDocs =[[word for word in text.lower().split() if word not in stopList] for text in docs]

    # word count before pre-processing
    l0 = []
    for doc in tokenizedDocs:
        l0.append(len(doc))
    
    # 2. compute frequency and remove infrequent words
    from collections import defaultdict
    frequency = defaultdict(int)
    for doc in tokenizedDocs:
        for token in doc:
            frequency[token] += 1

    #  i = 0
    #  for doc in tt:
    #      i += 1
    #      tot = 0
    #      elim = 0
    #      print("*"*10, "Doc # ", i ," :: ")
    #      for token in doc:
    #          tot +=1
    #          if frequency[token] == 1:
    #              elim += 1
    #      print("Based on Words Freq Doc # ", i, " eliminated ", elim, " words out of ", tot)

    docs = [[token for token in doc if frequency[token] > 1] for doc in
    tokenizedDocs]

    # new word count
    l1 = []
    for doc in docs:
        l1.append(len(doc))

    df = pd.DataFrame({
        "before": l0,
        "after":l1
    })
    df["changeFreq"]= (df.before-df.after)/df.before

    print("Word Counting after frequency reduction::")
    print("========================================")
    print(df.sort_values(by=["changeFreq"],ascending=False))


    # 3. remove words that are not in the vocabulary (google vocab)
    docs = [[token for token in doc if token in vocab_dict] for doc in docs]
    l2 = [len(doc) for doc in docs]
    df["vocab"] = l2
    df["changeVoc"] = (df.after - df.vocab)/df.after
    print("Word Counting after vocabulary reduction::")
    print("========================================")
    print(df.sort_values(by=["changeVoc"],ascending=False))

    # 4. rejoin documents for processing
    docs = [[ " ".join(token for token in doc)] for doc in docs]
    
    i = 0
    for doc in docs:
        i += 1
        print("-"*20, "DOC # ", i, " :: ")
        print(doc)

    print("END OF DOCS from FACTIVA ")
    print("*"*50)
    input("Factiva ok?")

    return docs



def getFeatures(docs):

    from sklearn.feature_extraction.text import CountVectorizer

    i = 0
    for doc in docs:
        i += 1
        print("-"*20, "DOC # ", i, " :: ")
        print(doc)

    print("END OF DOCS from FACTIVA ")
    print("*"*50)
    input(" now Factiva ok?")

    allDocs = []
    for doc in docs:
        allDocs.append(doc[0])

    vect = CountVectorizer(stop_words="english").fit(allDocs)

    print("Features = ", ",".join(vect.get_feature_names()))

    return vect




def computeDistances(nDocs, docs, vect):


    from sklearn.metrics import euclidean_distances
    from pyemd import emd


    #  for w in vect.get_feature_names():
    #      print("w is ", w)
    #      if w in vocab_dict:
    #          print(vocab_dict[w], " and the W is ", W[vocab_dict[w]])
    #          input("next")
    W_ = W[[vocab_dict[w] for w in vect.get_feature_names() if w in vocab_dict]]
    D_ = euclidean_distances(W_)


# check orthogonality (the closer to 1, the further apart)
    from scipy.spatial.distance import cosine
    cosVals = []
    for i in range(nDocs):
        aux = []
        for j in range(nDocs):
            v_1, v_2 = vect.transform([docs[i][0], docs[j][0]])
            print(v_1, " This is v1")
            v_1 = v_1.toarray().ravel()
            print(v_1, " This is v1")
            input("aka")
            v_2 = v_2.toarray().ravel()
            # print(v_1, v_2)
            #  print("cos(d{0}, d{1}) = {2:.2f}".format(i,j, cosine(v_1,v_2)))
            # aux.append(cosine(v_1,v_2))
            cosVals.append(cosine(v_1,v_2))

    #  j = 0
    #  for i in range(nDocs):
    #      print([cosVals[i*nDocs+j] for j in range(nDocs)])
    f = open("cosine.txt", "w")
    for i in range(nDocs):
        for j in range(nDocs):
            f.write("{0:8.2f}".format(cosVals[i*nDocs+j]))
        f.write("\n")
    f.close()

    input("Cosine written on disk file 'cosine.txt' ")
            

    wmd = []
    for i in range(nDocs):
        for j in range(nDocs):
            v_1, v_2 = vect.transform([docs[i][0], docs[j][0]])
            v_1 = v_1.toarray().ravel()
            v_2 = v_2.toarray().ravel()
            
            v_1 = v_1.astype(np.double)
            v_2 = v_2.astype(np.double)
            v_1 /= v_1.sum()
            v_2 /= v_2.sum()
            D_  = D_.astype(np.double)
            D_ /= D_.max()
            #  print("d(d{0}, d{1}) = {2:.2f}".format(i,j, emd(v_1,v_2,D_)))
            wmd.append(emd(v_1,v_2,D_))

    #  j = 0
    #  for i in range(nDocs):
    #      print([cosVals[i*nDocs+j] for j in range(nDocs)])
    f = open("wmd.txt", "w")
    for i in range(nDocs):
        for j in range(nDocs):
            f.write("{0:8.2f}".format(wmd[i*nDocs+j]))
        f.write("\n")
    f.close()
    input("WMD written on disk file 'wmd.txt' ")



    

def main(argv):
    '''
    Entry point.
    '''

    # vocabulary building (google based)
    vocabularyBuilding()

    #  baseName = "data/news/"
    baseName = "/home/marco/gdrive/research/nlp/examples2/txt/"
    nCompany = [12, 6]
    docs     = []
    nDocs    = sum(nCompany)
    prefix   = ["g", "a"]
    readDocuments(docs, nDocs, baseName, nCompany, prefix)
    printSummary(docs, nCompany, prefix)

    docs = docsPreprocessing(docs)
    vect = getFeatures(docs)


    computeDistances(nDocs, docs, vect)


if __name__ == '__main__':
    main(sys.argv[1:])
    #unittest.main()
