from scipy.stats import entropy
import numpy as np


def ShannonEntropy(matrix, base=2):
    _, counts = np.unique(matrix, return_counts=True)
    return entropy(counts, base=base)


def SAD(referenceMacroblock, targetMacroblock):
#     print (targetMacroblock.shape, referenceMacroblock.shape)
    return np.sum(np.abs(targetMacroblock - referenceMacroblock))
    