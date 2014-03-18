import numpy as np

import logging
import math


FORMAT = "[%(asctime)s] : %(filename)s.%(funcName)s():%(lineno)d - %(message)s"
DATEFMT = '%H:%M:%S, %m/%d/%Y'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATEFMT)
logger = logging.getLogger(__name__)

np.set_printoptions(linewidth=200, precision=3, suppress=True)

debug=False

def learn_HMM(pi, A, B, O, iterlimit=10000, threshold=0.0001):
    ''' This estimates the parameters lambda of an HMM using only an emitted
    symbol sequence O (unsupervised) and an inition guess lambda_0.'''

    converged = False

    T = len(O)
    N = len(A)
    M = len(B[0])
    cnt = 0
    logPold = -np.infty

    iters = 0

    while not converged and iters < iterlimit:

        alpha = np.zeros((T, N))
        beta = np.zeros((T, N))
        gamma = np.zeros((T, N))
        xi = np.zeros((N, N, T-1))
        c = np.zeros(T)

        iters += 1

        # Compute alpha
        for i in range(N):
            alpha[0, i] = pi[i] * B[i, O[0]]
        c[0] = 1. / np.sum(alpha[0, :])
        alpha[0, :] *= c[0]

        for t in range(1, T):
            for j in range(N):
                sm = 0
                for i in range(N):
                    sm += alpha[t - 1][i] * A[i, j]
                alpha[t, j] = sm * B[j, O[t]]
            c[t] = 1. / np.sum(alpha[t, :])
            alpha[t, :] *= c[t]



        # Compute beta
        for j in range(N):
            beta[T - 1, j] = 1.
        beta[T-1, :] *= c[T-1]

        for t in range(T - 2, -1, -1):
            for i in range(N):
                sm = 0
                for j in range(N):
                    sm += A[i, j] * B[j, O[t + 1]] * beta[t + 1, j]
                beta[t, i] = sm
            beta[t, :] *= c[t]

        if debug:
            print "alpha: (first few lines)"
            print alpha[:5, :]

            print "\nbeta: (first few lines)"
            print beta[:5, :]

        # compute gammas
        for t in range(T):
            rowsum = np.dot(alpha[t, :], beta[t, :])  # does this sum? I hope so.
            for i in range(N):
                gamma[t, i] = alpha[t, i] * beta[t, i] / rowsum

        # compute xi
        for t in range(T-1):
            xisum = 0
            for i in range(N):
                for j in range(N):
                    xi[i, j, t] = alpha[t, i] * A[i, j] * B[j, O[t+1]] * beta[t+1, j]
                    xisum += xi[i, j, t]

            xi[:, :, t] /= xisum

        # Following after description in Levinson book
        # Update A
        for i in range(N):
            for j in range(N):
                num = 0
                denom = 0
                for t in range(T - 1):
                    num += xi[i, j, t]
                    denom += gamma[t, i]
                A[i, j] = num / denom

        # update B
        for j in range(N):
            denom = np.sum(gamma[:, j])
            for k in range(M):
                num = 0
                for t in range(T):
                    if O[t] == k:
                        num += gamma[t, j]
                B[j, k] = num / denom

        # update pi
        pi = gamma[0, :]

        print "Iteration " + str(cnt + 1)

        logP = -1 * np.sum(map(lambda ct: math.log(ct), c))
        print "logP: ", logP
        #P = sum(alpha[T - 1, :])

        diff = logP - logPold
        print "change in prob (should be positive): ", diff, "\n"
        if diff < 0:
            "diff is not positive!"
            break

        if diff < threshold:
            print "We have reached our goal. diff=", diff
            break

        logPold = logP

        if debug:
            print "pi:"
            print pi, sum(pi)

            print "A:"
            for a in A:
                print a, sum(a)

            print "B:"
            for b in B:
                print b, sum(b)

        cnt += 1

    return pi, A, B