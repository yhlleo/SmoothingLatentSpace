"""
The metric has to measure how smooth is the latent space. 
Smoothness is loosely defined, but we assume that, given two 
latent points A and B, a model should be:
  - efficient: interpolating from A to B the model has to take 
    the shortest path possible;
  - equality: interpolating from A to B the model should do steps 
    of equal size
We measure efficiency through the fraction between the shortest 
perceptual path A-B and the sum of the interpolation perceptual 
distances between A and B. We measure inequality through the GINI 
coefficient of the interpolation perceptual distances between 
A and B. As we are intesrested in equality, we will use 1-gini.
"""

import numpy as np

def gini(array):
    """
    Calculate the Gini coefficient of a numpy array. Based on:
    http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    All values are treated equally, arrays can be 1d or 2d
    [d1, d2, d3... dn]
    or
    [[d11, d12, d13... d1n],
    ...
    [d21, d22, d23... dmn]]
    """
    # 
    assert np.all(array >= 0)
    if len(array.shape) == 1:
        array = array.reshape(1, -1)

    # Values not to be 0
    array += 1e-6     
    # Values must be sorted:
    array = np.sort(array, axis=1)
    # Index per array element:
    index = np.repeat(np.arange(1, array.shape[1]+1).reshape(1, -1), array.shape[0], axis=0)
    n = array.shape[1]

    return ((np.sum((2 * index - n  - 1) * array, axis=1)) / (n * np.sum(array, axis=1)))

def equality(array):
    """
    Computes the equality, which is 1-gini inequality.
    All values are treated equally, arrays can be 1d or 2d
    [d1, d2, d3... dn]
    or
    [[d11, d12, d13... d1n],
    ...
    [d21, d22, d23... dmn]]
    """
    return 1-gini(array)

def efficiency(shortest_paths, distances):
    """
    Computes the efficiency, defined as the ration between the
    shortest path between A and B and the sum of the segment distances 
    between A and B
    """
    score = shortest_paths/distances.sum(1)
    # the perceptual metric seems that it does not obey to triangle inequality
    # here we assume that if the shortest path is bigger than the sum of the
    # interpolation distances, the efficiency is capped to 1.
    score[score > 1] = 1
    return score

def PPS(shortest_paths, distances):
    """
    Computes the Perceptual Path Smoothness as the harmonic mean
    between equality and efficiency
    """
    i = equality(distances)
    e = efficiency(shortest_paths, distances)
    return 2*(i*e)/(i+e)

