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
#  from itertools import product
import json
import sys, getopt
import cplex
import os
from os import path, listdir
import csv
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

#  from sklearn.cluster import AffinityPropagation
#  from sklearn.datasets import load_digits

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist

from collections import namedtuple

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize

stop_words = stopwords.words("english")

pressrelease_folders = ["NEWS_RELEASE_v2/"]
pressrelease_folders_txt = ["NEWS_RELEASE_TXT_v2/"]
#  pressrelease_folders_txt = ["NEWS_RELEASE_TXT_v2CLEANED/"]
prefix = path.expanduser("~/gdrive/research/nlp/data/")
ecco_folders = ["ecco/ecco_portions/normed/96-00/"]
vocab_folder = "google_vocab/"

nCores       = 4


class Clusters:
    """
    Basic class to implement clusters related function. This class is used to
    store a cluster algoritm solution. Starting from a set of labels, i.e.,
    the cluster each document is assigned to, we reconstruct a full solution:
    - the number of clusters defined
    - the set of documents in each cluster
    - the centers of each cluster (centers can be set from outside, if computed
      by an algorithm)
    """
    def __init__(self):
        self.nPoints = -1 #  number of points in the cluster
        self.n       = -1 #  number of clusters
        self.labels  = [] #  list of labels (for each point)
        self.sets    = [] #  for each cluster, the set of documents in it
        self.tots    = [] #  total number of docs in each cluster
        self.centers = [] #  3-d center of each cluster (based on PCA)

    def updateClusterInfo(self):
        """
        Basic computation of clusters info, based on the labels vector.
        """
        self.nPoints = len(self.labels)
        self.n       = len(np.unique(self.labels))
        self.centers = [ [0.0 for j in range(3)] for i in range(self.n)]

    def createSetsFromLabels(self):
        """
        Get the set of points belonging to each cluster.

        Input:
        labels = [ for each point, the cluster label]

        Return:
        clusterSet =[
            [list of points in first cluster],
            [list of points in second cluster], 
            ...
            ]
        """
    
        self.tots = [0]*self.n
        for i in range(self.n):
            self.sets.append([])
        for i in range(self.nPoints):
            self.sets[self.labels[i]].append(i)
            self.tots[self.labels[i]] += 1
            


    def printSets(self):
        for c in range(self.n):
            print("C({0:3d}) :: {1}".format(c, self.sets[c]))

    def computeCenters3d(self, data):
        """
        Compute cluster centers based on PCA 3-d data. We transform each data
        point in a 3-d vector using PCA. Then, using the labels associated to a
        cluster solution, we compute the center of the cluster.

        Currently, we simply compute the average of each dimension. Other
        schemes can be explored.
        """


        for i in range(self.nPoints):
            print("Label of point ", i, " is ", self.labels[i])
            for j in range(3):
                self.centers[self.labels[i]][j] += data[i][j]

        for c in range(self.n):
            for j in range(3):
                self.centers[c][j] /= self.tots[c]

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
    ff = open(targetFile)
    target = ff.read()
    ff.close()
    return target

def readEcco(prefix):
    i = -1
    docs = []
    for folder in ecco_folders:
        i += 1
        fullpath = path.join(prefix, folder)
        totFiles = len(fnmatch.filter(os.listdir(fullpath), '*.txt'))
        countFiles = 0
        for f in listdir(path.join(prefix, folder)):
            if f != "1780100800.clean.txt":
                continue

            countFiles += 1
            fullname = fullpath + f
            ff = open(fullname)
            docs.append(ff.read())

            print("{0:5d}/{1:5d} :: Reading file {2:10s} ".format(countFiles,
            totFiles, f))

    return docs

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


    print("Reading model from disk ")
    start = timer()
    #  gPath = fullpath + "GoogleNews-vectors-negative300.bin.gz"
    #  model = KeyedVectors.load_word2vec_format(gPath, binary=True,
    #  limit=500000)
    filename = fullpath + "embed500.model"
    #  model = KeyedVectors.load(filename, mmap="r")
    model = KeyedVectors.load(filename)
    model.init_sims()
    print("Done in ", timer()-start, " time")

    return model




def printSummary(docs, sentences):
    from scipy import stats

    lengths = np.array([len(sent) for sent in sentences])
    qq = stats.mstats.mquantiles(lengths, prob=[0.0, 0.25, 0.50, 0.75, 1.10])

    print("\n\nmarco caserta (c) 2018 ")
    print("====================================================")
    print("Nr. Docs              : {0:>25d}".format(len(docs)))
    print("Nr. Sentences         : {0:>25d}".format(len(sentences)))
    print("Avg. Length           : {0:>25.2f}".format(lengths.mean()))
    print(" Min                  : {0:>25.2f}".format(qq[0]))
    print(" Q1                   : {0:>25.2f}".format(qq[1]))
    print(" Q2                   : {0:>25.2f}".format(qq[2]))
    print(" Q3                   : {0:>25.2f}".format(qq[3]))
    print(" Max                  : {0:>25.2f}".format(qq[4]))

    print("\n * Distance Type      : {0:>25s}".format(distanceType))
    print("====================================================")


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
    tokenized_sentence = []
    i = 0
    for ss in sents: # all the sentences in this document
        words = [w.lower() for w in word_tokenize(ss)]
        words = [w for w in words if w.isalpha() and w not in stop_words and w
        in vocab_dict]
        if len(words) > 1:
            docs.append(words)
            sentences.append([ss])
            i += 1

        #  if i > 100:
        #      return

def docPreprocessing(doc):

    doc = word_tokenize(doc)
    doc = [w.lower() for w in doc if w not in stop_words] # remove stopwords
    doc = [w for w in doc if w.isalpha() and w in vocab_dict] # remove numbers and pkt

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
    
    if os.path.exists("11doc2vec.model"):
        print(">>> Doc2Vec model was read from disk ")
        model = doc2vec.Doc2Vec.load("doc2vec.model")
        return model

    # instantiate model (note that min_count=2 eliminates infrequent words)
    model = doc2vec.Doc2Vec(size = 300, window = 300, min_count = 2, iter
    = 300, workers = 4, dm=0)

    # we can also build a vocabulary from the model
    model.build_vocab(docs)
    #  print("Vocabulary was built : ", model.wv.vocab.keys(), " ----> this is a voc")

    # train the model
    nrIters = 250
    #  model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
    model.train(docs, total_examples=model.corpus_count, epochs=nrIters)

    # if we want to save the model on disk (to reuse it later on without
    # training)
    model.save("doc2vec.model")

    # this can be used if we are done training the model. It will reelase some
    # RAM. The model, from now on, will only be queried
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    return model

def compileSimilarityList(model, target, docs, N):
    """
    We want to create a list of the top N most similar sentences w.r.t. the
    target sentence.
    """

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
    """
    model = Word2Vec(corpus, workers = nCores, size = 100, window = 300, min_count = 2, iter = 50)

    # use the following if we want to normalize the vectors
    # in this case, all vectors are converted to unit-length
    model.init_sims(replace=True)

    return model


def wmd4Docs(target, docs):
    """
    Use Word Mover's Distance for the entire corpus. We create a corpus based
    on (a subset of) the documents in docs. Next, we train the model using the
    gensim function. The model is stored into "model".
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

    See: https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
    """

    target = "this is the target document sentence"
    print(target, " ********")
    target = nltkPreprocessing(target)
    print(target, " ********")
    nTop = 6
    wmd_corpus = []
    wmd_corpus.append("this is the first sentence")
    wmd_corpus.append("this is the second sentence")
    wmd_corpus.append("this is the third sentence")
    wmd_corpus.append("this is the fourth sentence")
    wmd_corpus.append("this is the fifth sentence")
    wmd_corpus.append("this is the sixth sentence")

    pre = []
    for doc in wmd_corpus:
        pre.append(nltkPreprocessing(doc))
    print(pre)

    instance = WmdSimilarity(pre, model, num_best=nTop)
    sims = instance[target]
    print('Query:', target)
    for i in range(nTop):
        print( 'sim = %.4f' % sims[i][1])
        print(wmd_corpus[sims[i][0]])


def bench_k_means(estimator, name, data):
    """
    silhouette_score: 
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more
    similar.
    """
    t0 = timer()
    estimator.fit(data)

    print("{0:15s}{1:10.2f}{2:10.2f}{3:10.2f}\t{4}".format(name, timer()-t0,
    estimator.inertia_, metrics.silhouette_score(data,
    estimator.labels_,metric="euclidean",sample_size=len(data)),
    estimator.labels_))

def clustering(model, docs, chartType=0):

    nDocs = len(docs)

    print(82 * '_')
    print("{0:15s}{1:10s}{2:10s}{3:10s}\t{4}".format("init", "time",
    "inertia", "silhoutte", "clusters"))

    bench_k_means(KMeans(init="k-means++", n_clusters = 3, n_init = 2), name =
    "k-means++ 3", data = model.docvecs)
    for k in range(2,8):
        nn = "random-"+str(k)
        bench_k_means(KMeans(init="random", n_clusters = k, n_init = 2), name =
        nn, data = model.docvecs)
    pca = PCA(n_components=3).fit(model.docvecs)
    bench_k_means(KMeans(init=pca.components_, n_clusters = 3, n_init = 1), name =
    "PCA", data = model.docvecs)
    print(82 * '_')

    if chartType != "None":
        nClusters = 3
        pcaData = PCA(n_components = 3).fit_transform(model.docvecs)
        kmeans = KMeans(n_clusters=nClusters, n_init=3)
        kmeans.fit(pcaData)
        centers = kmeans.cluster_centers_
        print(kmeans.labels_)
        df = pd.DataFrame({
            "id": range(nDocs),
            "x1": pcaData[:,0],
            "x2": pcaData[:,1],
            "x3": pcaData[:,2],
            "cluster": kmeans.labels_})
        
        sets = []
        labels = kmeans.labels_
        for i in range(nClusters):
            sets.append([])
        for i in range(len(labels)):
            sets[labels[i]].append(i)

        for c in range(nClusters):
            print("Cluster ", c, " :: ", sets[c])

    else:
        return

    if chartType == "2d":
        sns.lmplot("x1", "x2",data=df , hue="cluster", fit_reg=False)
        plt.title("Clusters Visualization based on PCA")
        sns.plt.savefig("clusters.png")
        print("Clusters visualization saved on disk ('clusters.png')")

    elif chartType == "3d":
        create3dChart2(centers, pcaData, kmeans.labels_)
    

def affinityPropagation(model, docs):
    """
    This is used to get an idea of the number of clusters that should be
    created. It might be a preliminary step to cluster analysis.
    Note that it is possible to establish the relative importance of each
    point in the dataset.
    """

    nDocs = len(docs)
    pcaData = PCA(n_components = 2).fit_transform(model.docvecs)
    df = pd.DataFrame({
        "id": range(nDocs),
        "x1": pcaData[:,0],
        "x2": pcaData[:,1]})

    af = AffinityPropagation(affinity="euclidean",
    damping=0.9,convergence_iter=1000,verbose=True).fit(pcaData)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Labels = ", labels)

    colors = ['blue','green','red','cyan','magenta']
    data = []
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        ids = [i for i in range(nDocs) if class_members[i] == True]
        cluster_center = pcaData[cluster_centers_indices[k], :]
        trace1 = go.Scatter(x=pcaData[class_members, 0], 
                            y=pcaData[class_members, 1],
                            showlegend=False,
                            text = ids,
                            textposition = "bottom",
                            mode='text+markers', marker=dict(color=col,
                                                       size=10))
        
        trace2 = go.Scatter(x=[cluster_center[0]], 
                            y=[cluster_center[1]], 
                            showlegend=False,
                            mode='markers', marker=dict(color=col,
                                                        size=14))
        data.append(trace1)
        data.append(trace2)
        for x in pcaData[class_members]:
            trace3 = go.Scatter(x = [cluster_center[0], x[0]], 
                                y=[cluster_center[1], x[1]],
                                showlegend=False,
                                opacity=0.25,
                                mode='lines', line=dict(color=col,
                                                        width=2))
            data.append(trace3)

    layout = go.Layout(title='Estimated number of clusters: %d' % n_clusters_,
                       xaxis=dict(zeroline=False),
                       yaxis=dict(zeroline=False))
    fig = go.Figure(data=data, layout=layout)

    plotly.offline.plot(fig)

def getClustersFromLabels(labels):

    numClusters = len(labels)
    sets = []
    for i in range(numClusters):
        sets.append([])
    for i in range(len(labels)):
        sets[labels[i]].append(i)

    
def nltkClustering(model, nrClusters):
    """
    Use the clustering function of nltk to define my own distance function.
    """

    import nltk
    from nltk.cluster import KMeansClusterer

    num_clusters = nrClusters
    kclusterer = KMeansClusterer(num_clusters, 
        distance = nltk.cluster.util.cosine_distance,
        #  distance = nltk.cluster.util.euclidean_distance,
        repeats = 500)
    labels = kclusterer.cluster(model.docvecs, assign_clusters=True)

    return labels

def hierarchicalClustering(distanceMatrix, withDendrogram=False):
    """ 
    Use the linkage function of scipy.cluster.hierarchy.

    Parameters:
    - withDendrogram : True if the dendrogram should be produced and saved on disk

    We try out different methods for hierarchical clustering, based on
    different ways of computing distances between clusters. We select the one 
    that maximizes the Cophenetic Correlation Coefficient.

    Returns:
    - labels: The clusters labels
    """

    # convert symmetric distance matrix into upper triangular array
    distArray = ssd.squareform(np.asmatrix(distanceMatrix), checks=False)
    # find "best" method
    methods    = ["ward", "median", "average", "single", "complete"]
    bestVal    = 0.0
    bestMethod = " "
    for mm in methods:
        Z = linkage(distArray, method=mm, optimal_ordering=True)

        # test the goodness of cluster with cophenetic correl coefficient
        c, cophDist = cophenet(Z, distArray)
        print("[ {0:10s} ] Cophenetic = {1:5.2f}".format(mm, c))
        if c > bestVal:
            bestVal    = c
            bestMethod = mm

    # repeat with best method
    Z = linkage(distArray, method=bestMethod, optimal_ordering=True)
    #  print(Z)
    # note: The Z gives the distances at which each cluster was merged

    # get the cluster for each point
    #  maxD   = 0.95
    maxD   = 0.3
    labels = fcluster(Z, maxD, criterion="distance")
    labels = labels - [1]*len(labels)  #  start from 0

    if withDendrogram:
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            Z,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            show_leaf_counts=True,
            get_leaves=True,
            #  truncate_mode="level",
            #  p =5,
        )
        plt.axhline(y=maxD, c='k')
        plt.savefig("dendrogram.png")
        print("Dendrogram saved on disk ('dendrogam.png')")

    return labels

def get3dPCA(data):
    """
    Given a set of n-dimensional point, find and store the first 3 principal
    components. These are typically used to create a 3-d chart.
    """

    return PCA(n_components = 3).fit_transform(data)


def create3dChart(centerCoord, pcaData, labels):


    nPoints = len(pcaData)
    nClusters = len(centerCoord)
    data = []
    scatter = dict(
        mode = "text+markers",
        name = "docs",
        type = "scatter3d",    
        x = pcaData[:,0], y = pcaData[:,1], z = pcaData[:,2], 
        text = list(range(nPoints)),
        marker = dict(size=5, color=labels, colorscale="Jet" )
        #  marker = dict(size=5, color=labels, colorscale="Viridis" )
    )
    data.append(scatter)

    centers = dict(
        mode = "markers",
        name = "centers",
        type = "scatter3d",
        line = dict( width = 2, color = 'gray'),
        opacity = 0.75,
        x = [centerCoord[i][0] for i in range(nClusters)],
        y = [centerCoord[i][1] for i in range(nClusters)],
        z = [centerCoord[i][2] for i in range(nClusters)],
        marker = dict( size=3, color="red")
        )
    data.append(centers)
        
    for i in range(nPoints):
        trace = go.Scatter3d(
            showlegend=False,
            name ="lines",
            x = [centerCoord[labels[i]][0], pcaData[i][0]],
            y = [centerCoord[labels[i]][1], pcaData[i][1]],
            z = [centerCoord[labels[i]][2], pcaData[i][2]],
            mode="lines", line=dict(color="red",width=2))
        data.append(trace)

    layout = dict(
        title = 'Documents Clustering',
        showlegend = False,
        scene = dict(
            xaxis = dict( zeroline=False ),
            yaxis = dict( zeroline=False ),
            zaxis = dict( zeroline=False ),
        )
    )
    fig = go.Figure( data=data, layout=layout )
    plotly.offline.plot(fig, filename='PCA_Based_Clustering.html') 


def transportationProblem(model, corpus, d1, d2):

    from gensim.corpora import Dictionary
    target= "bring demand capacity in line with line sales demand demand"

    target = nltkPreprocessing(target)
    print(target, " ********")
    nTop = 6
    wmd_corpus = []
    wmd_corpus.append("reduce demand cost structure in order to improve our profitability")
    wmd_corpus.append("this is the first document finance")
    wmd_corpus.append("this is the second sentence")
    wmd_corpus.append("this is the third sentence")
    wmd_corpus.append("this is the fourth sentence")
    wmd_corpus.append("this is the fifth sentence")
    wmd_corpus.append("this is the sixth sentence")
    pre = []
    for doc in wmd_corpus:
        pre.append(nltkPreprocessing(doc))


    print("Sentences : ", pre[0], " AND ", target)
    print("== == == OFFICIAL WMD = ", model.wmdistance(pre[0], target))

    dctTarget = Dictionary([target])
    d1 = pre[0]
    dctd1 = Dictionary([d1])


    print("supplier vs customer ", d1,  " ", target)
    capacity = dctd1.doc2bow(d1)
    demand = dctTarget.doc2bow(target)
    print("LENGHTS ", len(capacity), " ... ", len(demand))

    nS = len(capacity)
    nD = len(demand)

    print("S and D = ", capacity, " " , demand)
    for i in range(nS):
        print("v2 = ", dctd1[i], " = ", capacity[i][1])
    print("*"*50)
    for j in range(nD):
        print("v2 = ", dctTarget[j], " = ", demand[j][1])


    cap = np.array([capacity[i][1] for i in range(nS)])
    cap = cap/sum(cap)
    dem = np.array([demand[i][1] for i in range(nD)])
    dem = dem/sum(dem)

    print("Cap vector ", cap)
    print("Dem vector ", dem)

    # distances
    S = [model.wv[dctd1[i]] for i in range(nS)]
    D = [model.wv[dctTarget[i]] for i in range(nD)]
    dd = pairwise_distances(S, D, metric="euclidean")
    for i in range(nS):
        print([dd[i][j] for j in range(nD)])

    [z, xSol] = solveTransport(dd, cap, dem)
    
    indexMax = [np.argmax(xSol[i]) for i in range(nS)]
    thickness = [xSol[i][indexMax[i]] for i in range(nS)]
    indexMax = [indexMax[i] + nS for i in range(nS)]

        
    X = []
    tt = []
    for i in range(nS):
        X.append(model.wv[dctd1[i]])
        tt.append("doc")
    for j in range(nD):
        X.append(model.wv[dctTarget[j]])
        tt.append("target")

    seed = np.random.RandomState(seed=27)
    mds = manifold.MDS(n_components = 2, metric = True, max_iter = 3000, eps =
    1e-9, random_state = seed, dissimilarity = "euclidean", n_jobs = nCores)
    sol = mds.fit(X).embedding_


    colors = ["blue", "red", "green", "magenta"]
    typesLabels = ["doc", "target"]
    nPoints = len(X)
    print("TT is = ", tt)
    data = []

    words = [dctd1[i] for i in range(nS)]
    ids =[i for i in range(nPoints) if tt[i] == "doc"]
    sizes = cap*100
    trace = go.Scatter(x = sol[ids,0],
                       y = sol[ids,1],
                       showlegend = False,
                       hoverinfo="skip",
                       text = words,
                       textposition = "bottom",
                       mode = "text+markers",
                       marker=dict(color="red",size=cap*100))
    data.append(trace)
    words = [dctTarget[i] for i in range(nD)]
    ids = range(nS, nS+nD)
    trace = go.Scatter(x = sol[ids,0],
                       y = sol[ids,1],
                       showlegend = False,
                       text = words,
                       textposition = "bottom",
                       mode = "text+markers",
                       marker=dict(color="blue",size=dem*100))
    data.append(trace)
    annotations = []
    for i in range(nS):
        annot = dict(x=sol[i,0],
                     y=sol[i,1],
                     xref = "x",
                     yref = "y",
                     text=round(thickness[i],2),
                     ax = -20,
                     ay = -20,
                     #  ax = (sol[indexMax[i],0]-sol[i,0])*500,
                     #  ay = (sol[indexMax[i],1]-sol[i,1])*500,
                     showarrow=False)
        annotations.append(annot)



        trace = go.Scatter(x = [sol[i,0], sol[indexMax[i],0]],
                           y = [sol[i,1], sol[indexMax[i],1]],
                           showlegend = False,
                           opacity = 0.25,
                           mode="lines",
                           text=round(thickness[i],2),
                           #  textposition="top center",
                           line=dict(color="red",width=thickness[i]*20))
        data.append(trace)


    layout = go.Layout(title="MDS of Two Sentences",
    annotations = annotations,
    xaxis = dict(zeroline=False),
    yaxis = dict(zeroline=False))
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig)






    exit(123)

def solveTransport(matrixC, cap, dem):
    
    nS = len(cap)
    nD = len(dem)
    cpx = cplex.Cplex()
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
        print("Capacity constraint supplier ", i, " ... rhs = ", cap[i] )
        index = [x_ilo[i][j] for j in range(nD)]
        value = [1.0]*nD
        capacity_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [capacity_constraint],
                                   senses   = ["L"],
                                   rhs      = [cap[i]])

    # demand constraints
    #  for j in dctTarget:
    for j in range(nD):
        print("Demand constraint customer ", j, " ... rhs = ", dem[j])
        index = [x_ilo[i][j] for i in range(nS)]
        #  index = [x_ilo[i][j] for i in dctd1]
        #  value = [1.0]*len(dctd1)
        value = [1.0]*nS
        demand_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [demand_constraint],
                                   senses   = ["G"],
                                   rhs      = [dem[j]])

    cpx.solve()

    z = cpx.solution.get_objective_value()

    print("z* = ", z)
    x_sol = []
    for i in range(nS):
        x_sol.append(cpx.solution.get_values(x_ilo[i]))
        print([round(x_sol[i][j],2) for j in range(nD)])

    return [z, x_sol]


def main(argv):
    '''
    Entry point.
    '''

    myClusters = Clusters()

    docsAux    = [] #  the original documents, as read from disk
    docs       = [] #  the documents in format required by doc2vec
    corpus     = [] #  the documents after preprocessing (not fit for doc2vec)
    distMatrix = [] #  the distance matrix used by hierarchical clustering

    parseCommandLine(argv)
    
    target = readTargetSentence(targetFile)
    print("Target sentence : ", target)

    vocabularyBuilding(prefix)

    targetTokenized = docPreprocessing(target)
    print("Now: ", targetTokenized)

    if os.path.exists("11preprocessedSentences.txt"):
        print("\n\n>>> preprocessed sentences read from disk (skipping preprocessing)")
        with open('preprocessedSentences.txt', 'r') as f:
            docs = json.load(f)
            corpus = docs[:]

        printSummary(docs)

    else:
        docsAux = readEcco(prefix)

        print("## Init Document Preprocessing [p={0}] ...".format(nCores))
        start  = timer()
        #  p      = Pool(nCores)
        #  docs   = p.starmap(nltkPreprocessing, docs, sentences, docsAux))
        #  printSummary(docs)
        #  corpus = docs[:]
        #  p.close()
        #  p.join()
        for doc in docsAux:
            nltkPreprocessing(docs, corpus, doc)

        fsentences = open("corpusSentences.txt", "w")
        for i in range(len(corpus)):
            fsentences.write("{0} \t {1}\n".format(i, corpus[i]))
        fsentences.close()

        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

        with open('preprocessedSentences.txt', 'w') as outfile:
            json.dump(docs, outfile)

    printSummary(docsAux, docs)

    print("## Init Document Transformation for Doc2Vec...")
    start = timer()
    # add tags to documents (required by doc2vec)
    docs = transform4Doc2Vec(docs) 
    print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

    print("## Init Doc2Vec Model Creation ...")
    start = timer()
    modelDoc2Vec = applyDoc2Vec(docs)
    print("... Done in {0:5.2f} seconds.\n".format(timer() - start))
    
    similarityList = compileSimilarityList(modelDoc2Vec, targetTokenized, docs, 5)
    print("Target sentence : ", target)
    print("="*80)
    for i,j in similarityList:
        ss = ", ".join( repr(e) for e in corpus[i] )
        print("s[{0:5d}] = {1:5.2f} :: {2}".format(i, j, ss))
        print("="*80)

    exit(111)
    # REM: If we want to use WMD, we need to use a model that creates vectors
    #  for individual words. If we want to use doc2vec, then the vector is per
    # document, not per word. These are two different models!

    # Thus, wmd works on the corpus, to which only base preprocessing has
    # been applied (tokenization, punctuation, etc.) 
    # Every technique based on document distances, e.g., cosine similariy,
    # requires the use of doc2vec which, in turn, requires a further
    # transformation of the documents. All the doc2vec operations are applied
    # on docs, rather than on corpus.

    
    if distanceType == "1":
        print("## Computing docs similarity using doc2vec...")
        start = timer()
        distMatrix = computeDocsSimilarities(modelDoc2Vec, docs)
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))
    elif distanceType == "2":
        print("## Computing docs similarity using WMD ...")
        distMatrix = wmd4Docs(target, corpus)
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))
        

    # apply clustering algorithm on distMatrix
    if distanceType == "1" or distanceType == "2":
        myClusters.labels = hierarchicalClustering(distMatrix, withDendrogram=True)
    elif distanceType == "3":
        myClusters.labels = nltkClustering(modelDoc2Vec, nrClusters = 8)

    print("LABELS ARE = ", myClusters.labels)
    myClusters.updateClusterInfo()
    myClusters.createSetsFromLabels()
    myClusters.printSets()

    # This uses Multidimensional Scaling (works only if distMatrix is
    # symmetric)
    print("## Starting with MDS ")
    seed = np.random.RandomState(seed=3)
    start = timer()
    mds = manifold.MDS(n_components = 3, metric = True, max_iter = 3000,
    eps = 1e-9, random_state = seed, dissimilarity = "precomputed", n_jobs =
    nCores)
    #  embed3d = mds.fit(dd).embedding_
    embed3d = mds.fit(distMatrix).embedding_
    print("... Done in {0:5.2f} seconds.\n".format(timer() - start))
    myClusters.computeCenters3d(embed3d)
    create3dChart(myClusters.centers, embed3d, myClusters.labels)




if __name__ == '__main__':
    main(sys.argv[1:])
    #unittest.main()
