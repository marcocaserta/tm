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
 Ended   :

 Command line options (see parseCommandLine):
-i inputfile

NOTE: This code is based on the tutorial from:

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

"""

from multiprocessing import Pool
#  from itertools import product
import json
import csv
import sys, getopt
import cplex
import os
from os import path, listdir
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from timeit import default_timer as timer

from gensim.models import doc2vec
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from gensim.corpora.dictionary import Dictionary

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

from nltk.corpus import stopwords
from nltk import word_tokenize

stop_words = stopwords.words("english")

pressrelease_folders = ["data/NEWS_RELEASE_v3/"]
pressrelease_folders_txt = ["data/NEWS_RELEASE_TXT_v3/"]
#  pressrelease_folders_txt = ["NEWS_RELEASE_TXT_v2CLEANED/"]
prefix = path.expanduser("~/gdrive/research/nlp/")
edgar_models_folder = "data/edgar_models"
d2vmodelname = "doc2vec.model"

results_folder = "results"
project_folder = "edgar/"

simDoc2VecMatrixFile = "similarityDoc2Vec.txt"
simWMDMatrixFile = "similarityWMD.csv"

inputfile    = ""
distanceType = -1
nCores = 4


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
    
    try:
        opts, args = getopt.getopt(argv, "ht:i:", ["help","type=", "ifile="])
    except getopt.GetoptError:
        print ("Usage : python doc2Vec.py -t <type> -i <inputfile> ")

        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ("Usage : python doc2Vec.py -t <type> -i <inputfile> ")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--type"):
            distanceType = arg

    if distanceType == -1:
        print("Error : Distance type not defined. Select one using -t : ")
        print("\t 1 : Cosine similarity (document-wise)")
        print("\t 2 : Word Mover's Distance (word-wise)")
        sys.exit(2)



def readDocuments(docs, prefix):
    """
    Read press release files (all files in a folder).
    """

    fmap = open("mapping.txt", "w")


    i = -1
    for folder in pressrelease_folders_txt:
        i += 1
        fullpath = path.join(prefix, folder)
        totFilesInFolder = len(fnmatch.filter(os.listdir(fullpath),
        '*.txt'))
        countFiles = 0
        for f in listdir(path.join(prefix, folder)):
            fmap.write("{0}\t {1:5d}\n".format(f, countFiles))
            countFiles += 1
            fullname = fullpath + f
            #  text = open(fullname).readlines()
            ff = open(fullname)
            docs.append(ff.read())

            print("{0:5d}/{1:5d} :: Reading file {2:10s} ".format(countFiles,
            totFilesInFolder, f))

            #  if countFiles > 4:
                #  return


    fmap.close()

def printSummary(docs):
    print("\n\nmarco caserta (c) 2018 ")
    print("====================================================")
    print("Nr. Docs              : {0:>25d}".format(len(docs)))
    print(" * Distance Type      : {0:>25s}".format(distanceType))
    print("====================================================")

    return


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


def nltkPreprocessing(doc):
    """
    Document preprocessing using NLTK:
    - tokenize document
    - remove stopwords (currently using stopwords list from nltk.corpus)
    - remove numbers and punctuation (using a function from nltk)
    - remove infrequent words
    """

    doc = doc.lower()
    doc = word_tokenize(doc)
    doc = [w for w in doc if w not in stop_words] # remove stopwords
    doc = [w for w in doc if w.isalpha()] # remove numbers and pkt

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
    
    fullpath = path.join(prefix, edgar_models_folder, d2vmodelname)
    if os.path.exists(fullname):
        print(">>> model was read from disk ")
        model = doc2vec.Doc2Vec.load(fullname)
        return model

    # instantiate model (note that min_count=2 eliminates infrequent words)
    model = doc2vec.Doc2Vec(size = 300, window = 300, min_count = 2, iter
    = 300, workers = nCores, dm=0)

    # we can also build a vocabulary from the model
    model.build_vocab(docs)
    #  print("Vocabulary was built : ", model.wv.vocab.keys(), " ----> this is a voc")

    # train the model
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
    model.init_sims(replace=True) #  normalize vectors

    # if we want to save the model on disk (to reuse it later on without
    # training)
    model.save(fullname)

    # this can be used if we are done training the model. It will reelase some
    # RAM. The model, from now on, will only be queried
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    return model


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


    fullpath = path.join(prefix,results_folder, project_folder)
    fullname = fullpath + simDoc2VecMatrixFile
    # save similarity matrix on disk
    f = open(fullname, "w")
    for i in range(nDocs):
        for j in range(nDocs):
            f.write("{0:4.2f}\t".format(vals[i][j]))
        f.write("\n")
    f.close()

    print("... similarity written on disk file '",simDoc2VecMatrixFile,"' ")

    return vals


def trainingModel4wmd(corpus):
    """
    Training a model to be used in WMD.
    """
    model = Word2Vec(corpus, workers = nCores, size = 100, window = 300,
    min_count = 2, iter = 250)
    #  model = Word2Vec(corpus)

    # use the following if we want to normalize the vectors
    model.init_sims(replace=True)

    return model


def wmd4Docs(docs):
    """
    Use Word Mover's Distance for the entire corpus. We create a corpus based
    on (a subset of) the documents in docs. Next, we train the model using the
    gensim function. The model is stored into "model"
    """

    fullpath = path.join(prefix,results_folder, project_folder)
    fullname = fullpath + simWMDMatrixFile
    if os.path.exists(fullname):
        print(" ... reading WMD matrix from disk ...")
        start = timer()
        vals = []

        reader = csv.reader(open(fullname), delimiter=",")
        x = list(reader)
        vals = np.array(x).astype("float")
        print(" ... Done in {0:5.2f} seconds.\n".format(timer()-start))
        return vals

    wmd_corpus = []
    for doc in docs:
        #  doc = nltkPreprocessing(doc) # already done outside
        wmd_corpus.append(doc)


    print("## Building model for WMD .... ")
    start = timer()
    modelWord2Vec = trainingModel4wmd(wmd_corpus)
    print("... Done in {0:5.2f} seconds.\n".format(timer()-start))

    print ("## Computing distances using WMD (parallel version [p={0}])\
    ...".format(nCores))

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

    with open(fullname, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(vals)

    print("Word Mover's Distances written on disk file '",fullname, "' ")
    
    return vals

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

    if os.path.exists("preEdgar/preprocessedDocs.txt"):
        print("\n\n>>> preprocessed docs read from disk (skipping preprocessing)")
        with open('preEdgar/preprocessedDocs.txt', 'r') as f:
            docs = json.loads(f.read())
        printSummary(docs)

    else:
        readDocuments(docsAux, prefix)
        printSummary(docsAux)

        print("## Init Document Preprocessing [p={0}] ...".format(nCores))
        start  = timer()
        p      = Pool(nCores)
        docs   = p.map(nltkPreprocessing, docsAux)
        corpus = docs[:]
        p.close()
        p.join()
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

        with open('preEdgar/preprocessedDocs.txt', 'w') as outfile:
            json.dump(docs, outfile)



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
        print("## Init Document Transformation for Doc2Vec...")
        start = timer()
        # add tags to documents (required by doc2vec)
        docs = transform4Doc2Vec(docs) 
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

        print("## Init Doc2Vec Model Creation ...")
        start = timer()
        modelDoc2Vec = applyDoc2Vec(docs)
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

        print("## Computing docs similarity using doc2vec...")
        start = timer()
        distMatrix = computeDocsSimilarities(modelDoc2Vec, docs)
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

    elif distanceType == "2":
        print("## Computing docs similarity using WMD ...")
        distMatrix = wmd4Docs(corpus)
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

    # This uses Principal Component Analysis
    #  pcaData = get3dPCA(modelDoc2Vec.docvecs)
    #  myClusters.computeCenters3d(pcaData)
    #  create3dChart(myClusters.centers, pcaData, myClusters.labels)

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
