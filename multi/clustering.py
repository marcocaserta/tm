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

 Clustering algorithm on Press Releases.

 Author: Marco Caserta (marco dot caserta at ie dot edu)
 Started : 05.04.2018
 Ended   :

 Command line options (see parseCommandLine):
-i inputfile

NOTE: This code is based on the tutorial from:

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

"""

from multiprocessing import Pool
import csv
import sys, getopt
import os
from os import path, listdir
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import logging
from collections import namedtuple
import glob

from gensim.models import doc2vec

from sklearn import manifold
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import fcluster

from nltk.corpus import stopwords
from nltk import word_tokenize
stop_words = stopwords.words("english")


prefix = path.expanduser("~/gdrive/research/nlp/")
vocab_folder       = "data/google_vocab/"
perDayFolders = "data/v4/"
folder        = "data/NEWS_TXT_v4"
task        = -1
clusterType = -1
nCores      =  4
yearTarget  = "-1"


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
            #  print("Label of point ", i, " is ", self.labels[i])
            for j in range(3):
                self.centers[self.labels[i]][j] += data[i][j]

        for c in range(self.n):
            for j in range(3):
                self.centers[c][j] /= self.tots[c]

def summary(task, clusterType, yearTarget):
    """
    Print out some summmary statistics.
    """

    print("\n\n")
    print("*"*80)
    print("*\t marco caserta (c) 2018 {0:>48s}".format("*"))
    print("*"*80)
    if task == "0":
        msg = "Folder Restructuring"
    elif task == "1":
        msg = "Preprocessing"
    elif task == "2":
        msg = "Doc2Vec Creation"
    elif task == "3":
        msg = "Distance Matrix"
    elif task == "4":
        msg = "Clustering (Type " + str(clusterType) + ")"
    print(" Task type   :: {0:>60s} {1:>3s}".format(msg,"*"))
    print(" Target year :: {0:>60s} {1:>3s}".format(yearTarget,"*"))
    print("*"*80)
    print("\n\n")

def parseCommandLine(argv):
    """
    Parse command line.
    """

    global task
    global clusterType
    global yearTarget
    
    try:
        opts, args = getopt.getopt(argv, "ht:c:y:", ["help","task=",
        "clusterType=", "year="])
    except getopt.GetoptError:
        print ("Usage : python clustering.py -t <task> -c <clustering> -y <year>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ("Usage : python doc2Vec.py -t <type> -i <inputfile> ")
            sys.exit()
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-c", "--clusterType"):
            clusterType = arg
        elif opt in ("-y", "--year"):
            yearTarget = arg

    if task == -1:
        print("Error : Task type not defined. Select one using -t : ")
        print("\t 0 :  Folder Restructuring")
        print("\t 1 :  Preprocessing")
        print("\t 2 :  Doc2Vec Creation")
        print("\t 3 :  Distance Matrix")
        print("\t 4 :  Clustering (define type using -c)")
        sys.exit(2)

    if yearTarget == "-1":
        print("Error : Target year not defined. Use -y ")
        sys.exit(2)

def folderStructure():
    """
    Move each file in the corresponding year folder and, within the year, the
    corresponding day and month. For example, a file issued on January 21st,
    2015 will be stored in the folder:
    - 2015/01_21/fullname.txt
    We also create a global mapping for all the files in the dataset. The
    mapping gives a unique identifier for each file in the dataset and it is
    saved in the file mapping.txt.
    """
    
    fmap = open("mapping.txt","w")
    fmap.write("name,cik,year,month,day,tag\n")

    fullpath = path.join(prefix,folder)
    count = 0
    for f in listdir(fullpath):
        fName = f[:-4]
        fAux = fName.split("_")
        year = fAux[1]
        dirLevel1 = path.join(prefix,perDayFolders) + year
        if not os.path.exists(dirLevel1):
            os.makedirs(dirLevel1)
        dirLevel2 = dirLevel1 + "/" + fAux[2] + "_" + fAux[3]
        fmap.write("{0}\t {1}\t {2}\t {3}\t {4}\n".format(f,fAux[0],fAux[1],fAux[2],fAux[3]))
        if not os.path.exists(dirLevel2):
            os.makedirs(dirLevel2)
        origin = fullpath + "/" + f
        dest   = dirLevel2 + "/" + f
        shutil.copy2(origin, dest)
        if count % 1000 == 0:
            print("[{0:7d}] Copying from {1} to {2}".format(count, origin, dest))
        count += 1
    
    fmap.close()

def preprocessing(year0, yearT):
    """
    We now get into each directory of the current year, and add each file
    within that directory to the corpus. In addition, we write a preprocessed
    version of the file (suffix .pre).

    Let us first import the mapping filename  →→  tag. Here we are reading the
    whole list, i.e., the entire dataset. We define two structures to query the
    filename and the absolute id value:

    - file2tag: Given a file name, it returns the corresponding unique identifier
    - tag2file: Given a unique identifier, i.e., a tag, it returns the full
      file name (cik_year_month_day.txt)

    We also create a dataframe, dfMap, which contains all the information
    associated to a specific file, i.e., full name, cik, year, month, day,
    tag.

    For the preprocessing phase, we use the following criteria:

    - exclude all words with length  ≤≤  2
    - exclude all words that are not alphabetic (isalpha() from nltk)
    - exclude stopwords
    - exclude all words which are not in the google news dictionary (cut to the
      500k most frequent words)
    """

    # read google dictionary (to filter some words)
    fullpath = path.join(prefix,vocab_folder)
    name_vocab = fullpath + "embed500.vocab"
    with open(name_vocab) as f:
        vocab_list = map(str.strip,f.readlines())
    vocab_dict = {w:k for k,w in enumerate(vocab_list)}

    # create global mapping
    print("## Creating Maps ")
    start = timer()
    dfMap, file2tag, tag2file = createMaps()
    print("... Done in {0:5.2f} seconds.\n".format(timer()- start))

    #print(file2tag["11544_1996_12_13.txt"])
    #print(tag2file["10"])

    # create preprocessed file in time window
    yearSet = list(range(int(year0), int(yearT)+1))

    for year in yearSet:
        print("*"*5, " Year = ", year, " ", "*"*5)
        
        dirFull = path.join(prefix,perDayFolders) + str(year)
        print(dirFull)

        for ff in listdir(dirFull):
            print("\t Entering folder ", ff)
            filteredList = path.join(dirFull,ff) + "/*.txt"
            for f in glob.glob(filteredList):

                fullname = path.join(dirFull,ff,f)
                # do not create, if preprocessed file already exists
                if os.path.isfile(fullname+".pre"):
                    continue

                fTxt = open(fullname)
                doc = fTxt.read()
                words = [w.lower() for w in word_tokenize(doc) if w not in stop_words]
                words = [w for w in words if len(w) > 2 and w.isalpha() and w in vocab_dict]

                preprocName = fullname+".pre"
                fPre = open(preprocName, "w")
                writerDoc = csv.writer(fPre)
                writerDoc.writerows([words])
                fPre.close()


def createMaps():
    nRows = sum(1 for line in open('mapping.txt'))
    dfMap = pd.DataFrame(index=np.arange(0, nRows), columns=('name', 'cik', 'year', 'month','day','tag') )

    #  with open("mapping.txt", "r") as f:
    #      row = 0
    #      for line in f:
    #          if row % 1000 == 0:
    #              print("Processing row {0:7d}".format(row))
    #          line=line.rstrip() # remove \n from each row
    #          dfMap.loc[row] = line.split(",")
    #          row += 1
    dfMap = pd.read_csv("mapping.txt")

    print("... dfMap created ")
    dfMap["year"] = dfMap["year"].astype("int")
    dfMap["month"] = dfMap["month"].astype("int")
    dfMap["day"] = dfMap["day"].astype("int")
    dfMap["tag"] = dfMap["tag"].astype("int")
    print("... creating of file2tag")
    file2tag = {f:t for f,t in zip(dfMap["name"],dfMap["tag"])}
    print("... creating of tag2file")
    tag2file = {t:f for t,f in zip(dfMap["tag"],dfMap["name"])}


    return dfMap, file2tag, tag2file

def createCorpus(year0, yearT):
    """
    Construct the corpus needed to build the doc2vec model for a given year.
    The model is built using all the documents produced in the preceding years,
    up to year T.
    """
    dfMap, file2tag, tag2file = createMaps()

    yearSet = list(range(int(year0), int(yearT)+1))

    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument','words tags')
    i = 0
    for year in yearSet:
        print("Considering year ", year)
        dirFull = path.join(prefix,perDayFolders) + str(year)
     
        for ff in listdir(dirFull):
            filteredList = path.join(dirFull,ff) + "/*.txt.pre"
            for f in glob.glob(filteredList):
                name = f.split("/")[-1]
                name = name[:-4]
                    
                fullname = path.join(dirFull, ff, f)
                fToken = open(fullname, "r")
                readerDoc = csv.reader(fToken)
                doc = next(readerDoc)
                fToken.close()
                tags = [str(file2tag[name])] 
                docs.append(analyzedDocument(doc,tags))

                i = i + 1
        print("Corpus with {0:5d} documents.".format(len(docs)))

    return docs


def trainDoc2Vec(docs, year):

    # creating doc2vec model for the current year
    model = doc2vec.Doc2Vec(size=300, window=100, min_count=2, iter=100, workers=8, dm=1, negative=10)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
    model.init_sims(replace=True)

    # save model on disk
    nameModel = "doc2vec.model." + str(year)
    fullname = path.join(prefix,perDayFolders,str(year),nameModel)
    print("Saving model ", fullname)
    model.save(fullname)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        

def createDoc2VecModel(year0, yearT):
    """
    We now need to read all the files belonging to a given year (later, up to
    that year) and build a doc2vec model to be used with the documents of that
    year. The doc2vect model is stored within the folder of that year and a
    suffix `.year` is added to each model.
    """

    docs = createCorpus(year0, yearT)
    trainDoc2Vec(docs, yearT)

def buildDistanceMatrix(year):
    """
    Here, we upload a doc2vec model for a specific year and use it to build a
    distance matrix. In addition, to separate model creation from model use, we
    need to find out which documents belong to the corpus.

    We can get the tag of each document in the current corpus, and directly
    obtain the embedding of that document using modelDoc2Vec.docvecs[tag].

    Let us first get a list of documents tag to be used to compute the distance
    matrix. Which documents should be included here depends on the strategy
    used. For now, assume we want to use all the documents of a given year.

    Note: In this case, we do not even need to load the corpus. We just need to
    create a list of tags corresponding to the documents we want to use in the
    distance matrix computation.
    """

    dfMap, file2tag, tag2file = createMaps()

    dirFull = path.join(prefix,perDayFolders) + year
    fullname = dirFull + "/doc2vec.model." + year
    print("Loading model ", fullname)
    modelDoc2Vec = doc2vec.Doc2Vec.load(fullname)
    print("The model contains {0} vectors.".format(len(modelDoc2Vec.docvecs)))
    
    # get absolute tag for each document in the current time window
    tagList = []
    nRows = len(dfMap)
    for row in range(nRows):
        #print(dfMap.iloc[row])
        if dfMap.iloc[row]["year"] == int(year):
            #print("file ", dfMap.iloc[row]["name"], " selected")
            tagList.append(dfMap.iloc[row]["tag"])
    print(tagList)

    # compute and store distance matrix
    nDocs = len(tagList)
    nameMatrix = "doc2vecDistMatrix.txt." + str(year)
    fullname = path.join(dirFull,nameMatrix)

    f = open(fullname, "w")
    writer = csv.writer(f)
    writer.writerow([nDocs])
    writer.writerow(tagList)
    vals = [ [0.0 for i in range(nDocs)] for j in range(nDocs)]
        
    for i in range(nDocs):
        tag_i = tagList[i]
            
        for j in range(i,nDocs):
            tag_j = tagList[j]
            val = round(1.0-modelDoc2Vec.docvecs.similarity(str(tag_i),str(tag_j)), 4)
            writer.writerow([tag_i, tag_j, val])

    f.close()
    print("Distance Matrix saved :: ", fullname)



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

def nltkClustering(vectors, nrClusters):
    """
    Use the clustering function of nltk to define my own distance function.
    """

    import nltk
    from nltk.cluster import KMeansClusterer

    num_clusters = nrClusters
    kclusterer = KMeansClusterer(num_clusters, 
        distance = nltk.cluster.util.cosine_distance,
        #  distance = nltk.cluster.util.euclidean_distance,
        repeats = 1000)
    labels = kclusterer.cluster(vectors, assign_clusters=True)

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
    maxD   = 0.15
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


def create3dChart(centerCoord, pcaData, labels, tagList, period):


    nPoints = len(pcaData)
    nClusters = len(centerCoord)
    data = []
    scatter = dict(
        mode = "text+markers",
        name = "docs",
        type = "scatter3d",    
        x = pcaData[:,0], y = pcaData[:,1], z = pcaData[:,2], 
        text = tagList,
        #  text = list(range(nPoints)),
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
        title = 'Documents Clustering for Period ' + str(period),
        showlegend = False,
        scene = dict(
            xaxis = dict( zeroline=False ),
            yaxis = dict( zeroline=False ),
            zaxis = dict( zeroline=False ),
        )
    )
    fig = go.Figure( data=data, layout=layout )
    plotly.offline.plot(fig, filename='MDS_Based_Clustering.html') 

def readDistanceMatrix(year):

    nameMatrix = "doc2vecDistMatrix.txt." + year
    print("... Reading distance matrix ", nameMatrix)
    fullname   = path.join(prefix,perDayFolders,year,nameMatrix)
    f = open(fullname, "r")
    reader = csv.reader(f)
    nDocs = int(next(reader)[0])
    distMatrix = [ [0.0 for i in range(nDocs)] for j in range(nDocs)]
    auxList = next(reader)
    tagList = [int(auxList[i]) for i in range(len(auxList))]

    i = -1
    j =  0
    for row in reader:
        if j % nDocs == 0:
            i += 1
            j  = i
        ix = int(row[0])
        jx = int(row[1])
        d  = float(row[2])
        distMatrix[i][j] = d
        distMatrix[j][i] = d
        j += 1
    f.close()

    return tagList, distMatrix

def loadDoc2VecModel(year):

    dirFull = path.join(prefix,perDayFolders,year)
    fullname = dirFull + "/doc2vec.model." + year
    print("Loading model ", fullname)
    modelDoc2Vec = doc2vec.Doc2Vec.load(fullname)
    print("The model contains {0} vectors.".format(len(modelDoc2Vec.docvecs)))

    return modelDoc2Vec

def clusteringAlgorithm(year, clusterType=1):
    """
    We now use the distance matrix saved on disk within each folder to carry
    out a specific type of analysis. For now, we use hierarchical clustering.
    The goal, though, is to see how clusters change over time and other type of
    analysis to study correlations among PR.
    """
    myClusters = Clusters()

    # read distance matrix from disk
    tagList, distMatrix = readDistanceMatrix(year)
    print(tagList)

    # apply clustering algorithm on distMatrix
    clusterType = "1"
    if clusterType== "1" or clusterType == "2":
        myClusters.labels = hierarchicalClustering(distMatrix, withDendrogram=True)
    elif clusterType == "3":
        modelDoc2Vec = loadDoc2VecModel(year)
        vectors = [modelDoc2Vec.docvecs[str(i)] for i in tagList]
        print("## Starting with NLTK Clustering ")
        start = timer()
        myClusters.labels = nltkClustering(vectors, nrClusters = 4)
        print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

    print("LABELS ARE = ", myClusters.labels)
    myClusters.updateClusterInfo()
    myClusters.createSetsFromLabels()
    myClusters.printSets()

    # This uses Multidimensional Scaling (works only if distMatrix is
    # symmetric)
    print("## Starting with MDS ")
    seed = np.random.RandomState(seed=3)
    start = timer()
    mds = manifold.MDS(n_components = 3, metric = True, max_iter = 10000,
    eps = 1e-9, random_state = seed, dissimilarity = "precomputed", n_jobs =
    nCores)
    #  embed3d = mds.fit(dd).embedding_
    embed3d = mds.fit(distMatrix).embedding_
    print("... Done in {0:5.2f} seconds.\n".format(timer() - start))
    myClusters.computeCenters3d(embed3d)
    create3dChart(myClusters.centers, embed3d, myClusters.labels, tagList,
    year)



def main(argv):
    '''
    Entry point. Five types of tasks, controlled via command line.
    '''
    parseCommandLine(argv)
    summary(task, clusterType, yearTarget)

    if task == "0":
        folderStructure()
    elif task == "1":
        preprocessing("1995", yearTarget)
    elif task == "2":
        createDoc2VecModel("1995", yearTarget)
    elif task == "3":
        buildDistanceMatrix(yearTarget)
    elif task == "4":
        clusteringAlgorithm(yearTarget, clusterType)



if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )

    main(sys.argv[1:])
