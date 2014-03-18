__author__ = 'stephen'
import re
import string
import numpy as np
import sys
import logging

import myhmm
from confusionmatrix import plotconf

FORMAT = "[%(asctime)s] : %(filename)s.%(funcName)s():%(lineno)d - %(message)s"
DATEFMT = '%H:%M:%S, %m/%d/%Y'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATEFMT)
logger = logging.getLogger(__name__)


def readandclean(fname):
    '''
    This reads a file name and returns a piece of text that contains only lowercase letters
    and spaces. This also returns a sorted list of characters in the text.
    '''

    with open(fname, 'r') as f:
        txt = f.read()

    let = set()

    txt = txt.translate(string.maketrans("", ""), string.punctuation).lower()
    txt = re.sub("[^a-z ]", "", txt)
    txt = re.sub("\s+", " ", txt).strip().lower()

    for c in txt:
        let.add(c)

    return txt, sorted(list(let))


def normalize(m):
    '''
    Given a 2d matrix, this returns the same matrix normalized by row.
    '''
    row_sums = m.sum(axis=1)
    return m / row_sums[:, np.newaxis]

def main():
    # Apparently this is a very bad idea.
    sys.setrecursionlimit(1500)


    txt, chars = readandclean("textdata.txt")

    # this is a dictionary mapping characters to indices (I know I could do this
    # with ord(c) - k, but this is more flexible to new character sets)
    cdict = dict([(x, i) for i, x in enumerate(chars)])

    #  this is the list of observations
    Olist = [cdict[c] for c in txt]

    # This where we set N
    N = 6
    M = len(chars)

    # number of observations to pass
    seqlen = 1000
    Olist = Olist[:seqlen]

    np.random.seed(10001)

    # this is a vector
    #pi = np.ones(N) / N
    pi = np.random.rand(N)
    pi = pi / sum(pi)

    # these are matrices
    #A = np.ones((N, N))
    A = np.random.rand(N, N)
    #B = np.ones((N, M))
    B = np.random.rand(N, M)
    #A0 = np.random.rand(N, N)
    #B0 = np.random.rand(N, M)

    A = normalize(A)
    B = normalize(B)

    mine = True

    if mine:
        newPi, newA, newB = myhmm.learn_HMM(pi, A, B, Olist, threshold=0.0001)
        plotconf(newA, title="A Matrix")
        plt = plotconf(newB, xlab=chars, title="B Matrix")
        plt.show()
    else:
        hmmguess = coloHMM.HMM()
        hmmguess.pi = pi
        hmmguess.A = A
        hmmguess.B = B

        hmmguess.train(np.array(Olist), 0.0001, graphics=False)

        plt = plotconf(hmmguess.A, title="A Matrix")
        plt = plotconf(hmmguess.B, xlab=chars, title="B Matrix")
        plt.show()


if __name__ == "__main__":
    main()


