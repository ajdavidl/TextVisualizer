"""
Module for phraseNet plot
"""

import re
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def phraseNet(listText, connectors, number_of_pairs=20):
    """
    Plot the Phrase net of a list of texts.

    It plots a phrase net graph based on given connectors from a list of texts.

    It uses networkx draw function to plot the grpah.

    Parameters
    ----------
    listText : list of strings
        List of text to be used as text source in the construction of the graph.

    connectors : list of strings
        List of connectors to be used in the construction of the graph.

    number_of_pairs : int
        Number of pairs of words to create the graph.
    """
    for i in range(len(connectors)):
        if " " in connectors[i]:
            conOrig = connectors[i]
            conNew = re.sub(" ", "_", conOrig)
            connectors[i] = conNew
            for j in range(len(listText)):
                listText[j] = re.sub(conOrig, conNew, listText[j])
        connectors[i] = " %s " % connectors[i]

    trigram_count_vect = CountVectorizer(analyzer='word', ngram_range=(3, 3))
    trigram_count_vect.fit(listText)
    bag_of_trigrams = trigram_count_vect.transform(listText)
    sum_trigrams = bag_of_trigrams.sum(axis=0)
    trigrams_freq = [(trigram, sum_trigrams[0, idx]) for trigram, idx in trigram_count_vect.vocabulary_.items()]
    trigrams_freq =sorted(trigrams_freq, key = lambda x: x[1], reverse=True)
    y_pos = np.arange(number_of_pairs)
    objects = []
    performance = []
    count = 0
    i=0

    G = nx.DiGraph()
    while (count < number_of_pairs) and (i < len(trigrams_freq)):
        aux = trigrams_freq[i]

        if sum([connector in aux[0] for connector in connectors])>0:
            aux = aux[0].split()
            # Create connections between nodes
            G.add_edge(aux[0], aux[2], weight=1)
            count += 1
        i += 1

    fig, ax = plt.subplots(figsize=(12, 8))

    # Hide grid lines
    ax.grid(False)

    #pos = nx.spring_layout(G, k=5.5)
    #pos = nx.kamada_kawai_layout(G)
    pos = nx.circular_layout(G)
    
    # Plot network
    nx.draw_networkx(G, pos,
                    font_size=16,
                    width=3,
                    edge_color='gray',
                    node_color='darkblue',
                    node_shape = '',
                    with_labels = True,
                    ax=ax)

    plt.show()

def phraseNetPlotly(listText, connectors, number_of_pairs=20):
    """
    Plot the Phrase net of a list of texts using Plotly.

    It plots a phrase net graph based on given connectors from a list of texts.

    It uses Plotly package to draw the graph.
    
    Parameters
    ----------
    listText : list of strings
        List of text to be used as text source in the construction of the graph.

    connectors : list of strings
        List of connectors to be used in the construction of the graph.

    number_of_pairs : int
        Number of pairs of words to create the graph.

    Returns
    -------
    plotly.graph_objs._figure.Figure
    """
    for i in range(len(connectors)):
        if " " in connectors[i]:
            conOrig = connectors[i]
            conNew = re.sub(" ", "_", conOrig)
            connectors[i] = conNew
            for j in range(len(listText)):
                listText[j] = re.sub(conOrig, conNew, listText[j])
        connectors[i] = " %s " % connectors[i]

    trigram_count_vect = CountVectorizer(analyzer='word', ngram_range=(3, 3))
    trigram_count_vect.fit(listText)
    bag_of_trigrams = trigram_count_vect.transform(listText)
    sum_trigrams = bag_of_trigrams.sum(axis=0)
    trigrams_freq = [(trigram, sum_trigrams[0, idx])
                     for trigram, idx in trigram_count_vect.vocabulary_.items()]
    trigrams_freq = sorted(trigrams_freq, key=lambda x: x[1], reverse=True)
    y_pos = np.arange(number_of_pairs)
    objects = []
    performance = []
    count = 0
    i = 0

    G = nx.DiGraph()
    while (count < number_of_pairs) and (i < len(trigrams_freq)):
        aux = trigrams_freq[i]

        if sum([connector in aux[0] for connector in connectors]) > 0:
            aux = aux[0].split()
            # Create connections between nodes
            G.add_edge(aux[0], aux[2], weight=1)
            count += 1
        i += 1

    #pos = nx.spring_layout(G, k=5.5)
    #pos = nx.kamada_kawai_layout(G)
    pos = nx.circular_layout(G)

    # Create Edges
    edge_x = []
    edge_y = []
    nodes = dict(pos.items())
    for edge in G.edges():
        x0, y0 = nodes[edge[0]]
        x1, y1 = nodes[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_list = []
    for node in G.nodes():
        x, y = nodes[node]
        node_x.append(x)
        node_y.append(y)
        node_list.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='text',
        text=node_list,
        textfont=dict(size=20, color='black'),
        hoverinfo='text',
    )
    # Color Node Points
    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        node_text.append(node)

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
        title='Entity coocurrence graph',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    )

    return fig


