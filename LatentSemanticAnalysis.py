import os
import sys
import gensim
import pandas as pd
import nltk

import math
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords




doc = open("Comments.txt", "r+")

A = doc.read()
B = A.split(" ")

#print(B)


doc2 = open("EvalCorpus.txt", "w")
doc2.write(str(B))
doc2.close()


doc3 = open("EvalCorpus.txt", "r+")


C = doc3.read()
#print(C)


doc4 = set(B)
#converted into the set to remove the duplicates
#print(doc4)

wordDict = dict.fromkeys(doc4 , 0)



for word in B:
    wordDict[word] += 1


def computeTF(wordDict, doc):
    tfDict = {}
    bowCount = len(doc)

    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict

tfBoW = computeTF(wordDict, B)
print(tfBoW)


def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    #Count the number of documents that contain the word w
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word,val in doc.items():
            if val > 0:
                try:
                    idfDict[word] += 1
                except:
                    idfDict[word] = 1
    #Divide N by denominator above and take log of that
    for word, val in idfDict.items():
        idfDict[word] = math.log(N/float(val))

    return idfDict

idfs = computeIDF([wordDict])


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidfBowA = computeTFIDF(tfBoW, idfs)



postDocs = [str(A)]

stopset = set(stopwords.words('english'))
stopset.update([])
print(len(stopset))

vectorizer = TfidfVectorizer(stop_words=stopset, use_idf=True, ngram_range=(1,2), smooth_idf=True, sublinear_tf=True)

X = vectorizer.fit_transform(postDocs)

print("Matrix")

print(X[0])


lsa = TruncatedSVD(n_components=2, n_iter=100)
lsa.fit(X)



print("The matrix V's first row")

print(lsa.components_[0])

terms = vectorizer.get_feature_names()

print("Terms")

print(terms)

for i, comp in enumerate(lsa.components_):
    termsInCmp = zip(terms, comp)
    sortedTerms = sorted(termsInCmp, key=lambda x:x[1], reverse=True)[:20]
    print("Concept %d:" %i)
    for term in sortedTerms:
        print(term[0])
    print(" ")
