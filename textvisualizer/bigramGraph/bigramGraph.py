"""
Module to make a bigram graph plot.
"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def bigramGraph(listText, stopwords=None, total_bigrams=15):
    """
    Make a graph of bigrams.

    It receives a list of text and make a graph of bigrams.
    It uses the package networkx under the hood.

    Parameters
    ----------
    listText : list of strings
        The corpus of text.

    stopwords : list of strings, default=None
        That list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    total_bigrams : integer
        The number of bigrams that will appear on the graph.

    """
    listText = [txt.lower() for txt in listText]
    if stopwords is not None:
        for i in range(0, len(listText)):
            words = listText[i].split(" ")
            words_new = [w for w in words if w not in stopwords]
            listText[i] = ' '.join(words_new)

    text = " ".join(listText)
    bigrams = [(a, b) for l in [text]
               for a, b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    bigrams = [(a, b) for a, b in bigrams if ((a != '') and (b != ''))]

    # Create counter of words in clean bigrams
    bigram_counts = Counter(bigrams)

    # Create a network plot of grouped terms
    bigram_df = pd.DataFrame(bigram_counts.most_common(total_bigrams),
                             columns=['bigram', 'count'])

    # Create dictionary of bigrams and their counts
    d = bigram_df.set_index('bigram').T.to_dict('records')

    # Create network plot
    G = nx.DiGraph()

    # Create connections between nodes
    for k, v in d[0].items():
        G.add_edge(k[0], k[1], weight=(v * 10))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Hide grid lines
    ax.grid(False)

    pos = nx.circular_layout(G)
    # Plot networks
    nx.draw_networkx(G, pos,
                     font_size=16,
                     width=3,
                     edge_color='gray',
                     node_color='darkblue',
                     node_shape='',
                     with_labels=True,
                     ax=ax)
