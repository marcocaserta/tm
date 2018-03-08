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

 Author: Marco Caserta (marco dot caserta at ie dot edu)
 Started : 07.03.2018
 Updated : 
 Ended   :

"""
import sys
import csv
import json
import itertools
from collections import namedtuple
import os
from os import path, listdir
import fnmatch
from timeit import default_timer as timer

from gensim.models import doc2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
from gensim.corpora.dictionary import Dictionary

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords

stop_words        = stopwords.words("english")
prefix            = path.expanduser("~/gdrive/research/nlp/data/")
ecco_folders      = ["ecco/ecco_portions/normed/96-00/"]
#  ecco_folders    = ["/home/marco/gdrive/research/nlp/data/temp/"]
vocab_folder      = "google_vocab/"
ecco_models_folder = "ecco_models/"
modelnameD2V       = "doc2vec.model.all" #  the Word2Vec model used in WMD

premiumList       = "premium/ecco-donut_freqs.txt"

preprocListDocs   = "preproc/preprocListDocs.csv"
preprocListCorpus = "preproc/preprocListCorpus.csv"
premiumDocs       = "preproc/premiumDocs.csv"
premiumCorpus     = "preproc/premiumCorpus.csv"
premiumDocsXRow   = "preproc/premiumDocsXRow.csv"
premiumCorpusXRow = "preproc/premiumCorpusXRow.csv"

targetFile = "target.txt"


nCores            = 4

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
def readPremiumList():
    global premium
    
    premium = []
    with open(path.join(prefix,premiumList), "r") as f:
        for line in f:
            w = line.split()[0]
            if len(w) > 2 and w not in stop_words and w in vocab_dict:
                premium.append(w)
    print("len premium = ", len(premium))
    
    return premium

def readEcco():

    fDocs = open(premiumDocsXRow, "w")
    writerDocs = csv.writer(fDocs)

    fCorpus = open(premiumCorpusXRow, "w")
    writerCorpus = csv.writer(fCorpus)

    docs      = []
    sentences = []
    tot       = 0
    totFiles  = 0
    discarded = 0
    for folder in ecco_folders:
        fullpath = path.join(prefix, folder)
        totFiles += len(fnmatch.filter(os.listdir(fullpath), '*.txt'))
        countFiles = 0
        for f in listdir(path.join(prefix, folder)):
            countFiles  += 1
            fullname     = fullpath + f
            ff           = open(fullname)
            sents        = sent_tokenize(ff.read())
            for sent in sents:
                #  print("-"*80)
                #  print("Initial sentence = ", sent)
                tot += 1
                words   = [w.lower() for w in word_tokenize(sent) if w not in
                stop_words]
                initLen = len(words)
                #  print(initLen, " = ", words)
                words    = [w for w in words if w.isalpha()]
                lenAlpha = len(words)
                #  print(lenAlpha, "After alpha = ", words)
                # keep if at least 50% words have all characters from alphabet
                if lenAlpha >= 0.5*initLen:
                    words = [w for w in words if w in vocab_dict and w in premium]
                    #  print("next ", words)
                    ll = len(words)
                    if ll > 2 and ll > int(0.5*lenAlpha):
                        #  print("kept = ", words)
                        writerDocs.writerows([words])
                        writerCorpus.writerows([[sent]])
                        #  docs.append(ss)
                        #  corpus.append(sCorpus)
                    else:
                        #  print("discarded ... ")
                        discarded += 1
                else:
                    #  print("discarded ... ")
                    discarded += 1

            print("{0:5d}/{1:5d} :: Reading file {2:10s} ".format(countFiles,
            totFiles, f))
            if countFiles > 4:
                break

    print("Discarded {0} sentences out of {1}".format(discarded, tot))

    fDocs.close()
    fCorpus.close()

    return docs, sentences, totFiles


def writeListsPerSentence():
    with open(premiumCorpus, 'r') as f:
        corpus = json.load(f)
    with open(premiumDocs, 'r') as f:
        docs = json.load(f)

    with open(premiumDocsXRow, "w") as f:
        writer =csv.writer(f)
        writer.writerows(docs)
    with open(premiumCorpusXRow, "w") as f:
        writer =csv.writer(f)
        writer.writerows(corpus)

    exit(111)
    

def readPremiumLists():
    
    fCorpus = open (premiumCorpusXRow, "r")
    readerCorpus = csv.reader(fCorpus)

    fDocs = open(premiumDocsXRow, "r")
    totSents = sum(1 for _ in fDocs)
    print("Tot sentences = ", totSents)
    fDocs = open(premiumDocsXRow, "r")
    readerDocs   = csv.reader(fDocs)
   
    corpus = []
    docs   = []
    for row in itertools.islice(readerCorpus, 0, totSents):
        #  print(readerCorpus.line_num, "Row corpus = ", row)
        corpus.append(row)

    for row in itertools.islice(readerDocs, 0, totSents):
        #  print(readerDocs.line_num, "Row docs = ", row)
        docs.append(row)


    return docs, corpus
    #  return corpus, docs


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


def createDoc2Vec(docs):
    
    print("## Init Doc2Vec Model Creation ...")

    start = timer()
    # instantiate model (note that min_count=2 eliminates infrequent words)
    model = doc2vec.Doc2Vec(size = 300, window = 300, min_count = 2, iter = 300, workers = nCores, dm=0)

    # we can also build a vocabulary from the model
    model.build_vocab(docs)
    #  print("Vocabulary was built : ", model.wv.vocab.keys(), " ----> this is a voc")

    # train the model
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
    model.init_sims(replace=True)

    # if we want to save the model on disk (to reuse it later on without
    # training)
    model.save("doc2vec.model")

    # this can be used if we are done training the model. It will reelase some
    # RAM. The model, from now on, will only be queried
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

    return model


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
    
def targetPreprocessing(doc):

    doc = word_tokenize(doc)
    doc = [w.lower() for w in doc if w.lower() not in stop_words] # remove stopwords

    doc = [w for w in doc if w.isalpha()] # remove numbers and pkt
    #  doc = [w for w in doc if w.isalpha() and w in vocab_dict] # remove numbers and pkt

    return doc

def main(argv):

    vocabularyBuilding(prefix)
    if buildDoc2VecModel:
        premium = readPremiumList()
        #  # activate this part if we want to read the original files, preprocess them
        #  # and store one sentence per row
        docs, corpus, totoFiles = readEcco()
        #
        #  # activate this part to create a Doc2Vec model from the list of sentences
        docs, corpus = readPremiumLists()
        docs = transform4Doc2Vec(docs)
        createDoc2Vec(docs)

    
    if useDoc2VecModel: 
        docs, corpus = readPremiumLists()  #  NEEDED ???
        fullpath = path.join(prefix,ecco_models_folder)
        fullname = fullpath + modelnameD2V 
        modelDoc2Vec  = doc2vec.Doc2Vec.load(fullname)
        target, nTop = readTargetSentence(targetFile)
        targetTokenized = targetPreprocessing(target)
        inferred_vector = modelDoc2Vec.infer_vector(targetTokenized)
        sims = modelDoc2Vec.docvecs.most_similar([inferred_vector], topn=10)
        idx = [i for i,j in sims]
        print(idx)
        print(sims)
        for i in idx:
            print("="*80)
            print(corpus[i-1])
            print("-"*80)
            print(corpus[i])
            print("-"*80)
            print(corpus[i+1])
            print("="*80)

if __name__ == '__main__':
    main(sys.argv[1:])
    #unittest.main()

