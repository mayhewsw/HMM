__author__ = 'stephen'

from caveneuwirth import *
import numpy as np

def pitest():
    with open("pi.txt") as f:
        nums = f.read()

    Olist = map(int, filter(lambda n: n != "\n", list(nums)))

     # This where we set N
    N = 6
    M = 10

    chars = range(11)

    # number of observations to pass
    seqlen = 10000
    Olist = Olist[:seqlen]

    # this is a vector
    #pi = np.ones(N) / N
    pi = np.random.rand(N)
    pi = pi / sum(pi)

    # these are matrices
    A = np.random.rand(N, N)
    B = np.random.rand(N, M)

    A = normalize(A)
    B = normalize(B)

    newPi, newA, newB = myhmm.learn_HMM(pi, A, B, Olist, threshold=0.01)
    plotconf(newA, title="A Matrix")
    plt = plotconf(newB, xlab=chars, title="B Matrix")
    plt.show()

if __name__ == "__main__":
    pitest()
