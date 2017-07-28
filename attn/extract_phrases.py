from nltk.tree import *
from stanford_corenlp_pywrapper import CoreNLP
from scipy.io import savemat
import numpy as np
import argparse

def ExtractPhrases(myTree, phrase):
    myPhrases = []
    if (myTree.label() == phrase):
        myPhrases.append(myTree.copy(True))
    for child in myTree:
        if (type(child) is Tree):
            list_of_phrases = ExtractPhrases(child, phrase)
            if (len(list_of_phrases) > 0):
                myPhrases.extend(list_of_phrases)
    return myPhrases

def main(model, dataset):
    f_cap = '../cap/f30k/' + model + '.' + dataset + '.txt'
    f_mat = '../cap/f30k/' + model + '.NP.' + dataset + '.mat'

    caps = []
    with open(f_cap, 'rb') as f:
        for line in f:
            caps.append(line.strip())

    proc = CoreNLP("parse", corenlp_jars=["../external/stanford-corenlp-full-2015-12-09/*"])

    NP = np.zeros((len(caps), ), dtype = np.object)
    for i in range(len(caps)): # for each caption
        p = proc.parse_doc(caps[i])
        assert len(p[u'sentences']) == 1, 'More than one sentence'

        tree = Tree.fromstring(p[u'sentences'][0][u'parse'].encode("ascii"))
        NP_trees = ExtractPhrases(tree, 'NP')
        NP_phrases = np.zeros((len(NP_trees), ), dtype = np.object)
        for j in range(len(NP_trees)): # for each noun phrase
            # print NP_trees[j]
            # NP_phrases[j] = ' '.join(NP_trees[j].leaves())
            NP_phrases[j] = np.array(NP_trees[j].leaves(), dtype = np.object)
        NP[i] = NP_phrases

    savemat(f_mat, mdict = {'NP': NP})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str)
    parser.add_argument('-d', type=str, default='test')

    args = parser.parse_args()
    main(model = args.m, dataset = args.d)