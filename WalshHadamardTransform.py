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
def WHTTree(patch, numOfBase):
    # Check if input is valid
    m = np.log2(patch.shape[0])
    if not m.is_integer:
        print("Error: WHTTree input shape is not power of 2!")
        return
    if not m == np.log2(patch.shape[1]):
        print("Error: WHTTree input is not square matrix!")
        return

    h = 2 * m  # Levels of binary tree
    length = int(2**h - 1) + numOfBase  # Total length of full binary tree
    tree = np.empty(length, dtype=object)
    tree[0] = patch

    ComputeTreeNodes(tree)
    return tree


def main():
    m = 3
    p = 2 ** m  # window(patch) size
    numOfBase = 16

    IMAGE_WIDTH = 80
    IMAGE_HEIGHT = 60
    sourceImage = np.random.randint(255, size=(IMAGE_HEIGHT, IMAGE_WIDTH))

    startTime = int(round(time.time() * 1000))
    #
    for i in xrange(0, IMAGE_HEIGHT - p):
        for j in xrange(0, IMAGE_WIDTH - p):
            patch = sourceImage[i:i + p, j:j + p]
            # Can be simplified to better performance
            tree = WHTTree(patch, numOfBase)
            np.vstack(tree[-numOfBase:]).flatten()
    #
    endTime = int(round(time.time() * 1000))
    elapse = endTime - startTime
    print(elapse)


if __name__ == '__main__':
    main()
