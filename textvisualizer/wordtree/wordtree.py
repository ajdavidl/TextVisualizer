"""
Module for generating word tree diagrams.
"""

import wordtree


def wordTree(corpus, keyword, maxNr=5):
    """
    Generate a word tree diagram.

    It uses the wordtree package under the hood.

    Parameters
    ----------
    corpus : list of strings
        The corpus of text to be used to draw the word tree.

    keyword : string
        The word to be the source of the tree.

    maxNr : integer
        Maximum number of words to be shown in a leaf of the tree.

    Returns
    -------
    graphviz.graphs.Digraph
    """
    return wordtree.search_and_draw(corpus=corpus, keyword=keyword, max_n=maxNr)
