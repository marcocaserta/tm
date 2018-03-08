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
 Started : 02.02.2018
 Updated : 27.02.2018
 Ended   :

 Command line options (see parseCommandLine):
-t type of distance computation approach used
    -t 1 : doc2vec
    -t 2 : WMD (word mover's distance)
    -t 3 : type of clustering algorithm used (k.means)

NOTE: This code is based on the tutorial from:

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

Branch ECCO:

The idea is to work on the documents of the ECCO dataset at a sentence level.
Given a "reference sentence," we want to find the list of sentences closer to
the target in a given pool of sentences. Note that the dataset must be
organized at a sentence level.

"""

from multiprocessing import Pool
import json
import sys, getopt
import cplex
import os
from os import path, listdir
import csv
import itertools
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from timeit import default_timer as timer
import math

from gensim.models import doc2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
from gensim.corpora.dictionary import Dictionary

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import manifold
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist

from collections import namedtuple

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words("english")

prefix             = path.expanduser("~/gdrive/research/nlp/data/")
ecco_folders       = ["ecco/ecco_portions/normed/96-00/"]
vocab_folder       = "google_vocab/"
ecco_models_folder = "ecco_models/"

querySol           = "topList.txt" #  storing the current best query result
modelnameW2V       = "word2vec.model.96-00.100" #  the Word2Vec model used in WMD
modelnameD2V       = "doc2vec.model.all" #  the Word2Vec model used in WMD
premiumList        = "premium/ecco-donut_freqs.txt"
premiumDocs        = "preproc/premiumDocs.csv"
premiumCorpus      = "preproc/premiumCorpus.csv"
premiumDocsXRow    = "preproc/premiumDocsXRow.csv"
premiumCorpusXRow  = "preproc/premiumCorpusXRow.csv"
#  ecco_folders    = ["/home/marco/gdrive/research/nlp/data/temp/"]

nCores       =  4
totFiles     = -1
nChunks      = 10


class Top:
    """
    Data structure used to store the top N sentences matching a given target
    sentence.
    We store both the full sentence (untokenized) and the tokenized and
    preprocessed sentence. We also store the previous and next sentences.
    """
    def __init__(self, nTop):
        self.score = np.array([math.inf]*nTop)
        self.tokenSent = [[]]*nTop
        self.sent     = [""]*nTop
        self.previous = [""]*nTop
        self.next     = [""]*nTop


# Parse command line
def parseCommandLine(argv):
    global distanceType
    global inputfile
    global targetFile 
    targetFile = ""
    distanceType = -1
    
    try:
        opts, args = getopt.getopt(argv, "ht:i:s:", ["help","type=", "ifile=",
        "sentence="])
    except getopt.GetoptError:
        print ("Usage : python doc2Vec.py -t <type> -s <sentence> ")

        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ("Usage : python doc2Vec.py -t <type> -i <inputfile> ")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--type"):
            distanceType = arg
        elif opt in ("-s", "--sentence"):
            targetFile = arg

    if distanceType == -1:
        print("Error : Distance type not defined. Select one using -t : ")
        print("\t 1 : Cosine similarity (document-wise)")
        print("\t 2 : Word Mover's Distance (word-wise)")
        sys.exit(2)
    if targetFile == "":
        print("Error: Target sentence file not defined. Select one using -s\
        <namefile>")
        exit(2)

def readTargetSentence(targetFile):
    """
    Read the target sentence from disk.
    Note that we also read how many sentences should be included in the query.
    """
    ff = open(targetFile, "r")
    reader = csv.reader(ff)
    target = next(reader)[0]
    nTop   = int(next(reader)[0])

    ff.close()
    return target, nTop

def listOfFiles():
    """
    This is used to create the list of tasks in reading files. These tasks are
    then passed to the parallel version of the reading. But, at this moment, it
    does not work.
    """
    
    i = -1
    fullList = []
    for folder in ecco_folders:
        i += 1
        fullpath = path.join(prefix, folder)
        totFiles = len(fnmatch.filter(os.listdir(fullpath), '*.txt'))
        countFiles = 0
        for f in listdir(path.join(prefix, folder)):
            #  if f != "1780100800.clean.txt":
                #  continue

            countFiles += 1
            fullname = fullpath + f
            fullList.append(fullname)

            if countFiles > 3:
                break

    return fullList, totFiles

def readEccoParallel(namefile):

    print("Reading ", namefile)
    docs = []
    sentences = []
    countFiles = 0
    ff = open(namefile)
    sents = sent_tokenize(ff.read())
    for sent in sents:
        words = [w.lower() for w in word_tokenize(sent)]
        words = [w for w in words if w.isalpha() and w not in stop_words]
        if len(words) > 2:
            docs.append(words)
            sentences.append([sent])
    print("Returning :", docs, " AND ", sentences)
    return zip(*docs, *sentences)

def readEcco():

    i = -1
    docs = []
    sentences = []
    totFiles = 0
    for folder in ecco_folders:
        i += 1
        fullpath = path.join(prefix, folder)
        totFiles += len(fnmatch.filter(os.listdir(fullpath), '*.txt'))
        countFiles = 0
        for f in listdir(path.join(prefix, folder)):
            #  if f != "1780100800.clean.txt":
                #  continue

            countFiles += 1
            fullname = fullpath + f
            ff = open(fullname)
            sents = sent_tokenize(ff.read())
            for sent in sents:
                words = [w.lower() for w in word_tokenize(sent)]
                words = [w for w in words if w.isalpha() and w not in
                stop_words and w in vocab_dict]
                if len(words) > 2:
                    docs.append(words)
                    sentences.append([sent])

            print("{0:5d}/{1:5d} :: Reading file {2:10s} ".format(countFiles,
            totFiles, f))
            if countFiles > 1:
                break

    return docs, sentences, totFiles


def vocabularyBuilding(prefix):
    '''
    Build embeddings and vocabulary based on Google News set (1.5Gb and 3M
    words and phrases based on 100B words from Google News)
    '''

    global vocab_dict

    # create dictionary first, using google list
    fullpath = path.join(prefix, vocab_folder)
    #  fullname = fullpath + "embed.dat"
    namevocab = fullpath + "embed500.vocab"
    with open(namevocab) as f:
        vocab_list = map(str.strip, f.readlines())
    vocab_dict = {w: k for k, w in enumerate(vocab_list)}
    print("Vocabulary read from disk ... ")


    #  print("Reading model from disk ")
    #  start = timer()
    #  #  gPath = fullpath + "GoogleNews-vectors-negative300.bin.gz"
    #  #  model = KeyedVectors.load_word2vec_format(gPath, binary=True,
    #  #  limit=500000)
    #  filename = fullpath + "embed500.model"
    #  #  model = KeyedVectors.load(filename, mmap="r")
    #  model = KeyedVectors.load(filename)
    #  model.init_sims(replace=True)
    #  print("Done in ", timer()-start, " time")

    #  return model




def printSummary(totFiles, docs):
    from scipy import stats

    lengths = np.array([len(sent) for sent in docs])
    qq = stats.mstats.mquantiles(lengths, prob=[0.0, 0.25, 0.50, 0.75, 1.10])

    print("====================================================")
    print("\n\nmarco caserta (c) 2018 ")
    print("====================================================")
    print("Nr. Files             : {0:>25d}".format(totFiles))
    print("Nr. Sentences         : {0:>25d}".format(len(docs)))
    print("Avg. Length           : {0:>25.2f}".format(lengths.mean()))
    print(" Min                  : {0:>25.2f}".format(qq[0]))
    print(" Q1                   : {0:>25.2f}".format(qq[1]))
    print(" Q2                   : {0:>25.2f}".format(qq[2]))
    print(" Q3                   : {0:>25.2f}".format(qq[3]))
    print(" Max                  : {0:>25.2f}".format(qq[4]))

    print("\n * Distance Type      : {0:>25s}".format(distanceType))
    print("====================================================")


    return

    fig, axes = plt.subplots(nrows=1, ncols=1)


    df = pd.DataFrame({
        "words": lengths})

    step_size = round((qq[4]-qq[0])/15)
    bins = np.arange(start=qq[0], stop=qq[4]+1, step= step_size)
    sns.distplot(df.words, hist_kws=dict(edgecolor="gray", linewidth=2),
    hist=True, kde=False, rug=False, bins=bins, ax=axes)
    axes.set(xticks=bins)
    axes.set(xlabel="Nr. Words")
    axes.set_title(r"[$\mu = $" +
    str("{0:5.2f}".format(lengths.mean())) + " $\sigma = $" +
    str("{0:5.2f}".format(lengths.std())) + "]")

    sns.plt.suptitle("Distribution of Words Length")
    #  sns.plt.show()
    sns.plt.savefig("distributionPlots.png")
    print("Distribution Plots saved on disk ('distributionPlots.png')")

    return


def nltkPreprocessing(docs, sentences, doc):
    """
    Document preprocessing using NLTK:
    - tokenize document
    - remove stopwords (currently using stopwords list from nltk.corpus)
    - remove numbers and punctuation (using a function from nltk)
    - remove infrequent words
    """

    #  doc = doc.lower()
    sents = sent_tokenize(doc)
    i = 0
    for ss in sents: # all the sentences in this document
        words = [w.lower() for w in word_tokenize(ss)]
        #  words = [w for w in words if w.isalpha() and w not in stop_words and w
        #  in vocab_dict]
        words = [w for w in words if w.isalpha() and w not in stop_words]
        if len(words) > 2:
            docs.append(words)
            sentences.append([ss])
            i += 1

        if i > 10:
            return

def targetPreprocessing(doc):

    doc = word_tokenize(doc)
    doc = [w.lower() for w in doc if w.lower() not in stop_words] # remove stopwords

    doc = [w for w in doc if w.isalpha()] # remove numbers and pkt
    #  doc = [w for w in doc if w.isalpha() and w in vocab_dict] # remove numbers and pkt

    return doc

def docPreprocessing(doc, modelWord2Vec):

    doc = word_tokenize(doc)
    doc = [w.lower() for w in doc if w.lower() not in stop_words] # remove stopwords
    passing = 1
    for w in doc:
        if w not in modelWord2Vec.wv.vocab:
            print("[ERROR] Word ", w, " of target sentence is not in vocabulary.")
            passing = 0
        if w.isalpha() == False:
            print("[ERROR] Word ", w, " is not in the alphabet.")
            passing = 0

        if passing == 0:
            return -1

    doc = [w for w in doc if w.isalpha() and w in modelWord2Vec.wv.vocab] # remove numbers and pkt
    #  doc = [w for w in doc if w.isalpha() and w in vocab_dict] # remove numbers and pkt

    return doc


def transform4Doc2Vec(docs):
    """
    Doc2Vec requires documents to be tokenized and, in addition, each doc
    should have a unique TAG.

    Note that, if the corpus has been preprocessed, each document is already
    tokenized. Therefore, some of the steps below can be skipped.
    """

    # transform documents to be used by doc2Vec
    documents = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, doc in enumerate(docs):
        # use first line if documents are not tokenized, otherwise next line
        #  words = text.lower().split()
        tags = [i]
        documents.append(analyzedDocument(doc, tags))

    return documents


def applyDoc2Vec(docs):
    """

    This returns a vector for each document of the corpus, based on shallow
    neural network (as in word2vec). The vocabulary used for training is built
    from the corpus itself (as opposed to using a vocabulary from Google News,
    or any other source.)

    Parameters of Doc2Vec are:
    - size : embedding vector size
    - window : looking before and after the current word
    - min_count : consider only words with that frequence or above
    - iter : number of iterations over the corpus
    - dm : with 0, deactivate DBoW (this seems to produce better results)

    Once the model has been obtained, to get the vector associated to a doc,
    just use:
    > print(model.docvecs[document_tag_here]
    """
    
    fullpath = path.join(prefix,ecco_models_folder)
    fullname = fullpath + modelnameD2V 
    print("fullname is ", fullname)
    if os.path.exists(fullname):
        print(">>> Doc2Vec model was read from disk ({0})".format(modelnameW2V))
        modelDoc2Vec = doc2vec.Doc2Vec.load(fullname)
        #  model = KeyedVectors.load_word2vec_format(fullname, encoding="utf-8")
        modelDoc2Vec.init_sims(replace=True)

        return modelDoc2Vec

    # instantiate model (note that min_count=2 eliminates infrequent words)
    model = doc2vec.Doc2Vec(size = 300, window = 300, min_count = 2, iter
    = 300, workers = 4, dm=0)

    # we can also build a vocabulary from the model
    model.build_vocab(docs)
    #  print("Vocabulary was built : ", model.wv.vocab.keys(), " ----> this is a voc")

    # train the model
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)

    # if we want to save the model on disk (to reuse it later on without
    # training)
    model.save("doc2vec.model")

    # this can be used if we are done training the model. It will reelase some
    # RAM. The model, from now on, will only be queried
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    return model

def compileDoc2VecSimilarityList(model, target, docs, N):
    """
    We want to create a list of the top N most similar sentences w.r.t. the
    target sentence.
    """

    print("target here = ", target)
    inferred_vector = model.infer_vector(target)
    sims = model.docvecs.most_similar([inferred_vector], topn=N)

    return sims


def computeDocsSimilarities(model, docs):
    """
    Compute similarity score among documents using the "most_similar()" function
    of doc2vec. This function finds the top n most similar documents with
    respect to the input vector. Similarity is based on cosine.

    This function can be called only after the creation of "model," i.e., the
    Doc2Vec model created in "applyDoc2Vec()".

    Similarities can be computed in two ways:
    - using the original document vector (stored in model.docvecs[doc_id]
    - using a "recomputed", i.e., inferred vector

    This distinction is interesting. If one uses the original score, the method 
    is deterministic, since the original vector obtained using Doc2Vec does not
    change. On the other hand, if we want to treat the corpus as training, we
    might want to recompute the vector associated to each document, as if
    each document now were unseen, new.

    The inferred function is also useful when a new document, i.e., not part of
    the corpus used for training, is now provided in input.

    To infer a vector for a document, use:
    > inferred_vector = model.infer_vector(new_doc)

    To reuse the vector obtained during the model training phase, use:
    > vector = model.docvecs[doc_id]
    """

    nDocs = len(docs)
    vals = [ [0.0 for i in range(nDocs)] for j in range(nDocs)]
    for doc_id in range(nDocs):
        inferred_vector = model.docvecs[doc_id]
        #  inferred_vector = model.infer_vector(docs[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn =
        len(model.docvecs))

        # store similarity values in a matrix
        # Note: We are storing DISTANCES, not similarities
        for i,j in sims:
            if vals[doc_id][i] == 0.0:
                vals[doc_id][i] = round(1.0-j,4) # round is needed to symmetry
                vals[i][doc_id] = round(1.0-j,4) # round is needed to symmetry


    # save similarity matrix on disk
    f = open("similarityDoc2Vec.txt", "w")
    for i in range(nDocs):
        for j in range(nDocs):
            f.write("{0:4.2f}\t".format(vals[i][j]))
        f.write("\n")
    f.close()

    print("... similarity written on disk file 'similarityDoc2Vec.txt' ")

    return vals


def trainingModel4wmd(corpus):
    """
    Training a model to be used in WMD.

    Settings:
    - hs = 0: Do not use hiercarchical softmax, but negative sampling
    - negative > 0 : number of negative words to be used in negative sampling
    - sorted_vocab : the most frequent words come first (we can trim it)

    """
    model = Word2Vec(workers = nCores, size = 100, window = 300,
    min_count = 2, hs=0, negative= 15, iter = 100, sorted_vocab=1)
    # build vocabulary based on the corpus
    model.build_vocab(corpus)

    model.train(corpus, total_examples=model.corpus_count,
    epochs=model.iter) 

    # use the following if we want to normalize the vectors
    # in this case, all vectors are converted to unit-length
    model.init_sims(replace=True)

    return model


def wmd4Docs(target, docs):
    """
    Use Word Mover's Distance for the entire corpus. We create a corpus based
    on (a subset of) the documents in docs. Next, we train the model using the
    gensim function. The model is stored into "model".

    NOTE: In this version, we do now use a pre-processing for this phase. We
    take the unprocessed corpus and create a Word2Vec model from there. Then,
    depending on the max_count threshold used, we migth have to eliminate some
    words from the corpus, if we use the Transportation problem.
    """

    if os.path.exists("11similarityWMD.csv"):
        print(" ... reading WMD matrix from disk ...")
        start = timer()
        vals = []

        reader = csv.reader(open("similarityWMD.csv", "r"), delimiter=",")
        x = list(reader)
        vals = np.array(x).astype("float")
        print(" ... Done in {0:5.2f} seconds.\n".format(timer()-start))
        return vals


    if os.path.exists("word2vec.model"):
        print(">>> Word2Vec model was read from disk ")
        modelWord2Vec = Word2Vec.load("word2vec.model")
    else:

        # extra step to eliminate some of the sentences from the corpus
        wmd_corpus = []
        for doc in docs:
            #  doc = nltkPreprocessing(doc) # already done outside
            wmd_corpus.append(doc)


        print("## Building model for WMD .... ")
        start = timer()
        modelWord2Vec = trainingModel4wmd(wmd_corpus)
        modelWord2Vec.save("word2vec.model")
        print("... Done in {0:5.2f} seconds.\n".format(timer()-start))
        print("         (Word2Vec model saved on disk - 'word2vec.model')")

    print ("## Computing distances using WMD (parallel version [p={0}])\
    ...".format(nCores))

    for i in range(len(wmd_corpus)):
        res = modelWord2Vec.wmdistance(target, wmd_corpus[i])
        print("Result WMD = ", res)

    #  getMostSimilarWMD(modelWord2Vec, [], [], 0)
    transportationProblem(modelWord2Vec, [], [], [])
    input("aka")


    nDocs = len(wmd_corpus)
    start = timer()
    # create list of tasks for parallel processing
    tasks = []
    for i in range(nDocs):
        for j in range(i+1,nDocs):
            tasks.append([docs[i],docs[j]])

    p = Pool(nCores)
    #  results = p.starmap(model.wmdistance, product(docs, repeat=2))
    results = p.starmap(modelWord2Vec.wmdistance, tasks)
    p.close()
    p.join()
    print("... done with distance computation in {0:5.2f} seconds.\n".format(timer()-start))


    print("Copying matrix ...")
    start = timer()
    # copy upper triangular vector into matrix
    vals = [ [0 for i in range(nDocs)] for j in range(nDocs)]
    progr = 0
    for i in range(nDocs):
        for j in range(i):
            vals[i][j] = vals[j][i]
        for j in range(i+1,nDocs):
            vals[i][j] = results[progr]
            progr += 1
            
    print("... Done in {0:5.2f} seconds.\n".format(timer()-start))

    # save matrix on disk
    csvfile = "similarityWMD.csv.temp"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(vals)

    print("Word Mover's Distances written on disk file 'similarityWMD.csv' ")
    input("aka")
    
    return vals

def getMostSimilarWMD(model, corpus, target, nTop):
    """
    Using the word2vec "model", find in the corpus the nTop most similar
    documents to target.

    NOTE: WmdSimilarity() provide the "negative" of wmdistance(), i.e.:
    sim(d1,d2) = 1/(1+wmdistance(d1,d2)).

    See: https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
    """

    instance = WmdSimilarity(corpus, model, num_best=nTop)
    instance.num_best = nTop
    sims = instance[target]
    #  print('Query:', target)
    #  print("="*80)
    #  for i in range(nTop):
    #      print( 'sim = %.4f' % sims[i][1])
    #      print(corpus[sims[i][0]])
    #      print("="*80)

    return sims

def transportationProblem(model, target, source):
    """
    Define the Transportation model used for the Word Movers Distance (WMD). We
    also setup a graph, to visualize the weight associated to each pair of
    words in a query.
    """

    from gensim.corpora import Dictionary

    print("== == == OFFICIAL WMD = ", model.wmdistance(source, target))

    dctTarget = Dictionary([target])
    dctd1     = Dictionary([source])
    capacity  = dctd1.doc2bow(source)
    demand    = dctTarget.doc2bow(target)
    nS = len(capacity)
    nD = len(demand)
    cap = np.array([capacity[i][1] for i in range(nS)])
    cap = cap/sum(cap)
    dem = np.array([demand[i][1] for i in range(nD)])
    dem = dem/sum(dem)
    #  print("S and D = ", capacity, " " , demand)
    #  for i in range(nS):
    #      print("v2 = ", dctd1[i], " = ", capacity[i][1])
    #  print("*"*50)
    #  for j in range(nD):
    #      print("v2 = ", dctTarget[j], " = ", demand[j][1])

    #  print("Cap vector ", cap)
    #  print("Dem vector ", dem)

    # distances
    S         = [model.wv[dctd1[i]] for i in range(nS)]
    D         = [model.wv[dctTarget[i]] for i in range(nD)]
    dd        = pairwise_distances(S, D, metric = "euclidean")

    [z, xSol] = solveTransport(dd, cap, dem)

    indexMax  = [np.argmax(xSol[i]) for i in range(nS)]
    thickness = [xSol[i][indexMax[i]] for i in range(nS)]
    indexMax  = [indexMax[i] + nS for i in range(nS)]

        
    X  = []
    tt = []
    for i in range(nS):
        X.append(model.wv[dctd1[i]])
        tt.append("doc")
    for j in range(nD):
        X.append(model.wv[dctTarget[j]])
        tt.append("target")

    seed = np.random.RandomState(seed=27)
    mds  = manifold.MDS(n_components = 2, metric = True, max_iter = 3000, eps =
    1e-9, random_state = seed, dissimilarity = "euclidean", n_jobs = nCores)
    sol  = mds.fit(X).embedding_


    colors      = ["blue", "red", "green", "magenta"]
    typesLabels = ["doc", "target"]
    nPoints     = len(X)
    data        = []
    words       = [dctd1[i] for i in range(nS)]
    ids         = [i for i in range(nPoints) if tt[i] == "doc"]
    trace = go.Scatter(
                       x            = sol[ids,0],
                       y            = sol[ids,1],
                       showlegend   = False,
                       hoverinfo    = "skip",
                       text         = words,
                       textposition = "bottom",
                       mode         = "text+markers",
                       marker       = dict(color = "red",size = cap*100))
    data.append(trace)

    words = [dctTarget[i] for i in range(nD)]
    ids   = range(nS, nS+nD)
    trace = go.Scatter(
                       x            = sol[ids,0],
                       y            = sol[ids,1],
                       showlegend   = False,
                       text         = words,
                       textposition = "bottom",
                       mode         = "text+markers",
                       marker       = dict(color      = "blue",size = dem*100))
    data.append(trace)

    annotations = []
    for i in range(nS):
        annot = dict(
                     x         = sol[i,0],
                     y         = sol[i,1],
                     xref      = "x",
                     yref      = "y",
                     text      = round(thickness[i],2),
                     ax        = -50,
                     ay        = -50,
                     #  ax     = (sol[indexMax[i],0]-sol[i,0])*500,
                     #  ay     = (sol[indexMax[i],1]-sol[i,1])*500,
                     showarrow = False)
        annotations.append(annot)

        trace = go.Scatter(
                           x               = [sol[i,0], sol[indexMax[i],0]],
                           y               = [sol[i,1], sol[indexMax[i],1]],
                           showlegend      = False,
                           opacity         = 0.25,
                           mode            = "lines",
                           text            = round(thickness[i],2),
                           #  textposition = "top center",
                           line            = dict(color = "red",width = thickness[i]*20))
        data.append(trace)


    layout = go.Layout(
                        title       = "MDS of Two Sentences",
                        annotations = annotations,
                        xaxis       = dict(zeroline  = False),
                        yaxis       = dict(zeroline  = False))

    fig         = go.Figure(data = data, layout = layout)

    plotly.offline.plot(fig)


def solveTransport(matrixC, cap, dem):
    """
    Solve transportation problem as an LP.
    This is my implementation of the WMD.
    """
    
    nS    = len(cap)
    nD    = len(dem)
    cpx   = cplex.Cplex()
    x_ilo = []
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    for i in range(nS):
        x_ilo.append([])
        for j in range(nD):
            x_ilo[i].append(cpx.variables.get_num())
            varName = "x." + str(i) + "." + str(j)
            cpx.variables.add(obj   = [float(matrixC[i][j])],
                              lb    = [0.0],
                              names = [varName])
    # capacity constraint
    for i in range(nS):
        index = [x_ilo[i][j] for j in range(nD)]
        value = [1.0]*nD
        capacity_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [capacity_constraint],
                                   senses   = ["L"],
                                   rhs      = [cap[i]])

    # demand constraints
    #  for j in dctTarget:
    for j in range(nD):
        index = [x_ilo[i][j] for i in range(nS)]
        value = [1.0]*nS
        demand_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [demand_constraint],
                                   senses   = ["G"],
                                   rhs      = [dem[j]])
    cpx.parameters.simplex.display.set(0)
    cpx.solve()

    z = cpx.solution.get_objective_value()

    print("z* = ", z)
    x_sol = []
    for i in range(nS):
        x_sol.append(cpx.solution.get_values(x_ilo[i]))
        print([round(x_sol[i][j],2) for j in range(nD)])

    return [z, x_sol]

def createWord2VecModel(docs):
    """
    Read or create a Word2Vec model (depending on whether the one defined in
    the header of this file actually exists.).
    """

    fullpath = path.join(prefix,ecco_models_folder)
    fullname = fullpath + modelnameW2V
    if os.path.exists(fullname):
        print(">>> Word2Vec model was read from disk ({0})".format(modelnameW2V))
        modelWord2Vec = Word2Vec.load(fullname)
        #  model = KeyedVectors.load_word2vec_format(fullname, encoding="utf-8")
        modelWord2Vec.init_sims(replace=True)
    else:
        print("## Building word2vec model for WMD ...")
        start = timer()
        modelWord2Vec = trainingModel4wmd(docs)
        modelWord2Vec.save("word2vec.model")
        print("... Done in {0:5.2f} seconds.\n".format(timer()-start))
        print("         (Word2Vec model saved on disk - 'word2vec.model')")

    return modelWord2Vec

def cleanCorpus(modelWord2Vec, docs, corpus, premium):
    """
    After the model has been uploaded, some of the preprocessed sentences might
    be removed. The initial preprocessing, which was used to generate the lists
    stored on disk, was based on the google dictionary. However, at this point,
    we have a model, and a corresponding dictionary. We need to ensure that all
    the words in the corpus at this point are in the dictionary.
    """

    if len(docs) != len(corpus):
        print(len(docs), " vs ", len(corpus))
        print("ERROR : Two different lengths in original and preprocessed docs")
        exit(123)

    count = 0
    newDocs = []
    newCorpus = []
    totP = 0
    flagged = 0
    for i in range(len(docs)):
        #  print("-"*80)
        #  print("doc[{0}] = {1}".format(i, docs[i]))
        #  print("-"*80)
        #  for w in docs[i]:
        #      if w not in premium:
        #          print(w, " is not in premium")
        #          totP += 1
        #      if w not in modelWord2Vec.wv.vocab:
        #          print(w, " is not in vocab")


        ss = [w for w in docs[i] if (w in modelWord2Vec.wv.vocab and w in
        premium)]
        if len(ss) > 2 and len(ss) > int(0.75*len(docs[i])):
            newDocs.append(ss)
            newCorpus.append(corpus[i])
        else:
            count += 1



    print("** ** From cleaning phase, eliminated ", count, " docs ")
    input("aka")

    #  with open('fullPremiumSentences.txt', 'a') as outfile:
    #      json.dump(newCorpus, outfile)
    #  with open('docsPremiumSentences.txt', 'a') as outfile:
    #      json.dump(newDocs, outfile)

    return newDocs, newCorpus



def compileWMDSimilarityList(modelWord2Vec, target, docs, p):
    """
    Returns a list of the top N most similar sentences w.r.t. the target
    sentence, using the Word Mover's Distance method.
    """


    nDocs = len(docs)
    # create list of tasks for parallel processing
    tasks = []
    for i in range(nDocs):
        tasks.append([target,docs[i]])

    results = p.starmap(modelWord2Vec.wmdistance, tasks)

    #  print ("## Computing distances using WMD (sequential version)\
    #  ...".format(nCores))
    #  start = timer()
    #  sims = getMostSimilarWMD(modelWord2Vec, docs, target, 5)

    # NOTE: The corpus now needs to be re-processed, since when we use the
    # transportation model below, we need to get the word2vec representation of
    # every word. If, in training, min_count > 1, some of the words in the
    # preprocessed corpus might not be in the vocabulary produced by word2vec,
    # even though they were in the vocabulary produced by google.


    # source : the origin, the document we want to transform into the target
    #  source = docs[sims[2][0]]
    #  transportationProblem(modelWord2Vec, target, source)
    #  print("... done with distance computation in {0:5.2f} seconds.\n".format(timer()-start))

    #  return sims
    #  return idx, distances
    return results

def updateList(tops, distances, docs, corpus):
    """
    Update list of top N best results.
    This can be improved.
    """

    if min(distances) > tops.score[nTop-1]:
        return tops

    nDocs = len(docs)
    for i in range(nTop):
        distances.append(tops.score[i])

    dist = np.array(distances)
    idx = dist.argsort()
    topAux = Top(nTop)
    for i in range(nTop):
        topAux.score[i] = dist[idx[i]]
        # special cases for first and last sentence of the chunk
        if idx[i] < nDocs:
            if idx[i] == 0:
                topAux.previous[i] = "*"
            else:
                topAux.previous[i] = corpus[idx[i]-1]
            if idx[i] == len(distances)-1:
                topAux.next[i] = "*"
            else:
                topAux.next[i] = corpus[idx[i]+1]
            topAux.tokenSent[i] = docs[idx[i]]
            topAux.sent[i]      = corpus[idx[i]]
        else:
            topAux.tokenSent[i] = tops.tokenSent[idx[i]-nDocs]
            topAux.sent[i]      = tops.sent[idx[i]-nDocs]
            topAux.previous[i]  = tops.previous[idx[i]-nDocs]
            topAux.next[i]      = tops.next[idx[i]-nDocs]

    # write current best query to disk
    printQueryResults(topAux, nTop, toDisk = 1)

    return topAux

def printQueryResults(tops, nTop, toDisk = 1):

    # dump it to file or print it to screen
    if toDisk == 1:
        sys.stdout = open(querySol, "w")

    print("** ** ** Target sentence : ", target)
    for i in range(nTop):
        print("="*80)
        ss = ", ".join( repr(e) for e in tops.previous[i] )
        print(ss)
        print("-"*80)
        ss = ", ".join( repr(e) for e in tops.sent[i])
        print("dist({0:2d}) = {1:5.2f} :: {2}".format(i+1, tops.score[i], ss))
        print("-"*80)
        ss = ", ".join( repr(e) for e in tops.next[i] )
        print(ss)
        
        print("="*80)
        print("\n\n")

    sys.stdout = sys.__stdout__

def readPremiumList():
    
    premium = []
    with open(path.join(prefix,premiumList), "r") as f:
        for line in f:
            w = line.split()[0]
            if len(w) > 2 and w not in stop_words:
                premium.append(w)
    print("len premium = ", len(premium))
    
    return premium


def main(argv):
    '''
    Entry point.
    '''

    global target
    global nTop

    docsAux    = [] #  the original documents, as read from disk
    docs       = [] #  the documents in format required by doc2vec
    corpus     = [] #  the documents after preprocessing (not fit for doc2vec)
    distMatrix = [] #  the distance matrix used by hierarchical clustering

    parseCommandLine(argv)
    #  vocabularyBuilding(prefix)

    docToVec = False
    if docToVec == True:
        print("## Init Document Transformation for Doc2Vec...")
        start = timer()
        # add tags to documents (required by doc2vec)
        docs = transform4Doc2Vec(docs) 
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

        print("## Init Doc2Vec Model Creation ...")
        start = timer()
        modelDoc2Vec = applyDoc2Vec(docs)
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))
        
        similarityList = compileDoc2VecSimilarityList(modelDoc2Vec,
        targetTokenized, docs, nTop)
        print("Target sentence : ", target)
        print("="*80)
        for i,j in similarityList:
            ss = ", ".join( repr(e) for e in corpus[i] )
            print("s[{0:5d}] = {1:5.2f} :: {2}".format(i, j, ss))
            print("="*80)

        exit(111)

    wmd = True
    if wmd:


        start = timer()

        print("## Setting up Word2Vec model ...")
        start = timer()
        modelWord2Vec = createWord2VecModel(docs)
        #  modelDoc2Vec  = applyDoc2Vec(docs)
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

        fDocs = open(premiumDocsXRow, "r")
        totSents = sum(1 for _ in fDocs)
        

        while True:
            targetTokenized = -1
            while targetTokenized == -1:
                target, nTop = readTargetSentence(targetFile)
                targetTokenized = docPreprocessing(target, modelWord2Vec)
                if targetTokenized == -1:
                    input("Fix the target file and press any key to continue\
                    ('target.txt')")
                else:
                    print("** ** QUERY : ", target,"\n")

            # setup csv readers (corpus and docs)
            fCorpus = open (premiumCorpusXRow, "r") 
            readerCorpus = csv.reader(fCorpus)
            fDocs = open(premiumDocsXRow, "r")
            readerDocs   = csv.reader(fDocs)

            chunkSize = math.ceil(totSents/nChunks)
            print("Tot Sentences = ", totSents, " and Chunks Size = ", chunkSize)
            p = Pool(nCores)


            tops = Top(nTop)

            print("\n\n")
            print("*"*80)
            print("* Query : ", target)
            print("*"*80)
            print("\n\n")
            print("## Reading CHUNKS of preprocessed sentences as lists ...")
            start    = timer()
            bestDist = math.inf
            progr    = 0
            for counter in range(nChunks):
                docs    = []
                corpus  = []
                init    = progr
                till    = min(progr + chunkSize, totSents)
                ending  = till-init
                progr  += chunkSize

                print("Chunk [{0:3d}/{1:3d}] =\
                {2:9d}--{3:9d}/{4:9d}".format(counter+1, nChunks, init+1, till, totSents))
                for row in itertools.islice(readerCorpus, 0, ending):
                    #  print(readerCorpus.line_num, "Row corpus = ", row)
                    corpus.append(row)

                for row in itertools.islice(readerDocs, 0, ending):
                    #  print(readerDocs.line_num, " == ", row)
                    docs.append(row)

                for doc in docs:
                    print(doc)


                distances = compileWMDSimilarityList(modelWord2Vec,
                targetTokenized, docs, p)
                tops = updateList(tops, distances, docs, corpus)
                if tops.score[0] < bestDist:
                    bestDist = tops.score[0]
                    print("Best Match = {0:5.2f} :: {1}".format(tops.score[0], tops.tokenSent[0]))
                #  if counter > 0:
                #      break
            if till != totSents:
                print("ERROR here : Some sentences have been skipped : ( ", progr,
                ",", totSents, " )")

            printQueryResults(tops, nTop, toDisk = 1)

            print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

            print("Do you want to produce a mapping? Choose sentence [1-",nTop,"]\
            (any other number to exit)")
            k = int(input())
            if k > 0 and k <= nTop:
                source = tops.tokenSent[k-1] 
                transportationProblem(modelWord2Vec, targetTokenized, source)

            asw = input("Type 'q' to quit. Otherwise, restart.")

            p.close()
            p.join()

            if asw == "q":
                break




if __name__ == '__main__':
    main(sys.argv[1:])
    #unittest.main()
