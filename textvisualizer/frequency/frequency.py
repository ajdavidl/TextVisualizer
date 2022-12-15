"""
Module for frequency plot 
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer


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
    count_vect = CountVectorizer(
        analyzer='word',
        stop_words=stopwords,
        ngram_range=ngramRange,
        vocabulary=vocabulary
    )
    count_vect.fit(listText)
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
    count_vect = CountVectorizer(
        analyzer='word',
        stop_words=stopwords,
        ngram_range=ngramRange,
        vocabulary=vocabulary
    )
    count_vect.fit(listText)
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
