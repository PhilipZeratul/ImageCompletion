import numpy as np
import math
import time


def ComputeTreeNodes(tree):
    n = len(tree)  # num of nodes (1, 2, 3, ...)
    h = int(math.ceil(np.log2(n + 1)))  # total layers (1, 2, 3, ...)
    seed = [1, -1, 1, -1, -1, 1, -1, 1]  # calculation sign seed

    for layer in xrange(1, h):
        delta = 2 ** int(math.floor((layer - 1) / 2))
        seedPointer = 0
        for node in xrange(2 ** layer - 1, min(n, 2 ** (layer + 1) - 1)):
            parent = int(math.floor((node - 1) / 2))

            # compute row |
            if layer % 2 == 0:
                p = tree[parent].shape[1]
                tree[node] = np.zeros((p, p), dtype=int)
                for i in xrange(0, p):
                    tree[node][i, :] = tree[parent][i, :] + tree[parent][i + delta, :] * seed[seedPointer]

            # compute col ->
            else:
                p = tree[parent].shape[0]
                tree[node] = np.zeros((p, p - delta), dtype=int)

                for i in xrange(0, p - delta):
                    tree[node][:, i] = tree[parent][:, i] + tree[parent][:, i + delta] * seed[seedPointer]

            seedPointer = (seedPointer + 1) % 8


# Construct WHT compute tree
def WHTTree(patch, patchSize, numOfBase):
    h = 2 * np.log2(patchSize)  # Levels of binary tree
    length = int(2 ** h - 1) + numOfBase  # Total length of full binary tree
    tree = np.empty(length, dtype=object)
    tree[0] = patch

    ComputeTreeNodes(tree)
    return tree


def main():
    m = 3
    p = 2 ** m  # window(patch) size
    numOfBase = 16

    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 600
    sourceImage = np.random.randint(255, size=(IMAGE_HEIGHT, IMAGE_WIDTH))

    startTime = int(round(time.time() * 1000))
    #
    WHTTree(sourceImage, p, numOfBase)
    #
    endTime = int(round(time.time() * 1000))
    elapse = endTime - startTime
    print(elapse)


if __name__ == '__main__':
    main()
