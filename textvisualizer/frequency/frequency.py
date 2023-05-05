"""
Module for frequency plot 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from collections import Counter


def __countvectorize(listText, stopwords=None, ngramRange=(1, 1), vocabulary=None):
    count_vect = CountVectorizer(
        analyzer='word',
        stop_words=stopwords,
        ngram_range=ngramRange,
        vocabulary=vocabulary
    )
    count_vect.fit(listText)
    return count_vect


def frequencyPlot(listText, number_of_words=20, stopwords=None, ngramRange=(1, 1), vocabulary=None):
    """
    Plot a bar graph with the token frequencies.

    It receives a list of text, count the tokens and plot a bar graph with the frequencies.

    It uses matplotlib under the hood.

    Parameters
    ----------
    listText : list of strings

    number_of_words : int
        Number of words to be plotted.

    stopwords : list of strings, default=None
        That list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    ngramRange : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer is not callable``.

    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    """
    count_vect = __countvectorize(listText, stopwords, ngramRange, vocabulary)
    bag_of_words = count_vect.transform(listText)
    sum_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_words[0, idx])
                 for word, idx in count_vect.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    yPos = np.arange(number_of_words)
    objects = []
    performance = []
    for i in range(number_of_words):
        aux = word_freq[i]
        objects.append(aux[0])
        performance.append(aux[1])
    plt.barh(yPos, performance, align='center', alpha=0.5)
    plt.yticks(yPos, objects)
    plt.xlabel('Frequency')
    plt.ylabel('Tokens')
    plt.title('Frequency of tokens')
    plt.show()


def frequencyPlotly(listText, number_of_words=20, stopwords=None, ngramRange=(1, 1), vocabulary=None):
    """
    Plot a bar graph with the token frequencies.

    It receives a list of text, count the tokens and plot a bar graph with the frequencies.

    It uses Plotly under the hood.
    Parameters
    ----------
    listText : list of strings

    number_of_words : int
        Number of words to be plotted.

    stopwords : list of strings, default=None
        That list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    ngramRange : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer is not callable``.

    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    Returns
    -------
    plotly.graph_objs._figure.Figure
    """
    count_vect = __countvectorize(listText, stopwords, ngramRange, vocabulary)
    bag_of_words = count_vect.transform(listText)
    sum_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_words[0, idx])
                 for word, idx in count_vect.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    objects = []
    performance = []
    for i in range(number_of_words):
        aux = word_freq[i]
        objects.append(aux[0])
        performance.append(aux[1])
    data = [go.Bar(x=objects, y=performance,
                   name='Frequency Plot', orientation='v')]
    layout = go.Layout(title="Frequencies", xaxis=dict(
        title="tokens"), yaxis=dict(title="quantity"))
    fig = go.Figure(data, layout)
    return fig


def frequencyPlotYellowbrick(listText, number_of_words=20, stopwords=None, ngramRange=(1, 1), vocabulary=None):
    """
    Plot a bar graph with the token frequencies.

    It receives a list of text, count the tokens and plot a bar graph with the frequencies.

    It uses yellowbrick under the hood.

    Parameters
    ----------
    listText : list of strings

    number_of_words : int
        Number of words to be plotted.

    stopwords : list of strings, default=None
        That list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    ngramRange : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer is not callable``.

    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.
    """
    count_vect = __countvectorize(listText, stopwords, ngramRange, vocabulary)
    docs = count_vect.transform(listText)
    features = count_vect.get_feature_names_out()

    visualizer = FreqDistVisualizer(
        features=features, orient='v', n=number_of_words)
    visualizer.fit(docs)
    visualizer.show()


def frequencyTreeMap(listText, number_of_words=100, stopwords=None, ngramRange=(1, 1), vocabulary=None):
    """
    Plot a tree map with the token frequencies.

    It receives a list of text, count the tokens and plot a tree map with the frequencies.

    It uses plotly express under the hood.

    Parameters
    ----------
    listText : list of strings

    number_of_words : int
        Number of words to be plotted.

    stopwords : list of strings, default=None
        That list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    ngramRange : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer is not callable``.

    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    Returns
    -------
        plotly.graph_objs._figure.Figure
    """
    count_vect = __countvectorize(listText, stopwords, ngramRange, vocabulary)
    count_vect.fit(listText)
    bag_of_words = count_vect.transform(listText)
    sum_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_words[0, idx])
                 for word, idx in count_vect.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    y_pos = np.arange(number_of_words)
    objects = []
    performance = []
    for i in range(number_of_words):
        aux = word_freq[i]
        objects.append(aux[0])
        performance.append(aux[1])
    df_words = pd.DataFrame({'words': objects, 'count': performance})
    fig = px.treemap(df_words, path=["words"],
                     values='count',
                     color='count',
                     color_continuous_scale='viridis',
                     color_continuous_midpoint=np.average(df_words['count'])
                     )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig


def frequencyDonutChart(listText, number_of_words=20, stopwords=None, ngramRange=(1, 1), vocabulary=None):
    """
    This function takes a text as input and plots a donut chart with the word frequencies using Plotly.

    Parameters
    ----------
    listText : list of strings

    number_of_words : int
        Number of words to be plotted.

    stopwords : list of strings, default=None
        That list is assumed to contain stop words, all of which will be removed from the resulting tokens.

    ngramRange : tuple (min_n, max_n), default=(1, 1)
        The lower and upper boundary of the range of n-values for different
        word n-grams or char n-grams to be extracted. All values of n such
        such that min_n <= n <= max_n will be used. For example an
        ``ngram_range`` of ``(1, 1)`` means only unigrams, ``(1, 2)`` means
        unigrams and bigrams, and ``(2, 2)`` means only bigrams.
        Only applies if ``analyzer is not callable``.

    vocabulary : Mapping or iterable, default=None
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents. Indices
        in the mapping should not be repeated and should not have any gap
        between 0 and the largest index.

    Returns
    -------
        plotly.graph_objs._figure.Figure
    """
    count_vect = __countvectorize(listText, stopwords, ngramRange, vocabulary)
    count_vect.fit(listText)
    bag_of_words = count_vect.transform(listText)
    sum_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_words[0, idx])
                 for word, idx in count_vect.vocabulary_.items()]
    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
    top_words = {k: v for k, v in word_freq[:number_of_words]}

    # Create the trace for the donut chart
    trace = go.Pie(
        labels=list(top_words.keys()),
        values=list(top_words.values()),
        hole=0.4,
        hoverinfo='label+percent',
        marker=dict(colors=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
                    '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3', '#8dd3c7', '#bebada']),
    )

    # Create the layout for the donut chart
    layout = go.Layout(
        title="Token Frequencies",
        margin=dict(t=30, b=10, l=10, r=10),
    )

    # Create the figure and plot the donut chart
    fig = go.Figure(data=[trace], layout=layout)
    return fig
