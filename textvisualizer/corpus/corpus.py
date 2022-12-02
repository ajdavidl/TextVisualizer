"""
Class Corpus
"""
from ..textvisualizer import *


class Corpus:
    """
    Class Corpus to centralize functions
    Attributes
    ----------
        listText : list of str
            List of the corpus text.
    """

    def __init__(self, listText):
        """
        Constructor of the class Corpus
        Parameters
        ----------
        listText : list of str
            List of the corpus text.
        """
        self.listText = listText

    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)

    def __str__(self):
        return "Object of class {self.__class__.__name__}".format(self=self)

    def frequencyPlot(self, number_of_words=20, stopwords=None, ngramRange=(1, 1), vocabulary=None):
        """
        Plot a bar graph with the token frequencies.

        Parameters
        ----------
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
        fig = frequencyPlot(self.listText, number_of_words=number_of_words,
                            stopwords=stopwords, ngramRange=ngramRange, vocabulary=vocabulary)
        return fig

    def phraseNet(self, connectors, number_of_pairs=20):
        """
        Plot the Phrase net of a list of texts.

        It plots a phrase net graph based on given connectors from a list of texts.

        Parameters
        ----------
        connectors : list of strings
            List of connectors to be used in the construction of the graph.

        number_of_pairs : int
            Number of pairs of words to create the graph.

        Returns
        -------
        plotly.graph_objs._figure.Figure
        """
        fig = phraseNet(self.listText, connectors=connectors,
                        number_of_pairs=number_of_pairs)
        return fig

    def wordcloudPlot(self, stopwords=None, max_font_size=50, max_words=100, background_color="white"):
        """
        Generate Word cloud figure.

        It uses the wordcloud package under the hood.

        Parameters
        ----------
        text : string
            List of text to be used as text source in the construction of the graph.

        stopwords : list of strings, default=None
            That list is assumed to contain stop words, all of which will be removed from the resulting tokens.

        max_font_size : int or None (default=50)
            Maximum font size for the largest word. If None, height of the image is used.

        max_words : number (default=100)
            The maximum number of words.

        """
        return wordcloudPlot(' '.join(self.listText), stopwords=stopwords, max_font_size=max_font_size, max_words=max_words, background_color=background_color)
