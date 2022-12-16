"""
Class Corpus
"""
from ..textvisualizer import *
from ..phraseNet.phraseNet import *
from ..frequency.frequency import *
from ..wordcloud.wordcloud import *
import pandas as pd


class Corpus:
    """
    Class Corpus to centralize functions.
    Attributes
    ----------
        listText : list of str
            List of the corpus text.
        lisLabels : list of str
            List of corpus labels.
    """

    def __init__(self, listText, listLabels=None):
        """
        Constructor of the class Corpus
        Parameters
        ----------
        listText : list of str
            List of the corpus text.
        listLabels : list of str
            List of corpus labels.
        """
        if (listLabels is not None):
            if len(listText) != len(listLabels):
                raise BaseException(
                    "Mismatch in lengths of listLabels and listText")

        self.listText = listText
        self.listLabels = listLabels

    def __repr__(self):
        return "{self.__class__.__name__}()".format(self=self)

    def __str__(self):
        return "Object of class {self.__class__.__name__}".format(self=self)

    def __mountDataframe(self, labels):
        """
        Private method to mount a dataframe with the corpus and labels.

        It check if the given label is valid.

        Raise an error if self.listLabels is None or parameter label is not a string or a list of string.

        Returns a pandas DataFrame if it's everything ok with the labels.

        Parameters
        ----------
        labels : str or list of str, default=None
            Labels to be used to filter the text. 

        Returns
        -------
        Pandas DataFrame.
        """
        if self.listLabels is not None:
            df = pd.DataFrame(
                {"text": self.listText, "label": self.listLabels})
        else:
            raise BaseException("The listLabels is None.")
        if type(labels) == str:
            df = df[df.label == labels]
        elif type(labels) == list:
            df = df[df.label.isin(labels)]
        else:
            raise BaseException("labels must be string or list of string")
        return df

    def frequencyPlot(self, number_of_words=20, stopwords=None, ngramRange=(1, 1), vocabulary=None, labels=None, plotly=False):
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

        labels : str or list of str, default=None
            Labels to be used to filter the text. 

        plotly : bolean 
            Flag to indicate the use of the plotly package. 
            Default = False

        Returns
        -------
        matplotlib plot.
        if plotly is true, it returns plotly.graph_objs._figure.Figure
        """
        if labels is None:
            if plotly:
                return frequencyPlotly(self.listText, number_of_words=number_of_words,
                                       stopwords=stopwords, ngramRange=ngramRange, vocabulary=vocabulary)
            else:
                return frequencyPlot(self.listText, number_of_words=number_of_words,
                                     stopwords=stopwords, ngramRange=ngramRange, vocabulary=vocabulary)
        else:
            df = self.__mountDataframe(labels=labels)
            if plotly:
                return frequencyPlotly(df.text.tolist(), number_of_words=number_of_words,
                                       stopwords=stopwords, ngramRange=ngramRange, vocabulary=vocabulary)
            else:
                return frequencyPlot(df.text.tolist(), number_of_words=number_of_words,
                                     stopwords=stopwords, ngramRange=ngramRange, vocabulary=vocabulary)
        return fig

    def phraseNet(self, connectors, number_of_pairs=20, labels=None, plotly=False):
        """
        Plot the Phrase net of a list of texts.

        It plots a phrase net graph based on given connectors from a list of texts.

        Parameters
        ----------
        connectors : list of strings
            List of connectors to be used in the construction of the graph.

        number_of_pairs : int
            Number of pairs of words to create the graph.

        labels : str or list of str, default=None
            Labels to be used to filter the text. 

        plotly : bolean 
            Flag to indicate the use of the plotly package. 
            Default = False

        Returns
        -------
        networkx draw figure.
        if plotly is true, it returns plotly.graph_objs._figure.Figure
        """
        if labels is None:
            if plotly:
                return phraseNetPlotly(self.listText, connectors=connectors,
                                       number_of_pairs=number_of_pairs)
            else:
                return phraseNet(self.listText, connectors=connectors,
                                 number_of_pairs=number_of_pairs)

        else:
            df = self.__mountDataframe(labels=labels)
            if plotly:
                return phraseNetPlotly(df.text.tolist(), connectors=connectors,
                                       number_of_pairs=number_of_pairs)
            else:
                return phraseNet(df.text.tolist(), connectors=connectors,
                                 number_of_pairs=number_of_pairs)

    def wordcloudPlot(self, stopwords=None, max_font_size=50, max_words=100, background_color="white", labels=None):
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

        labels : str or list of str, default=None
            Labels to be used to filter the text. 
        """
        if labels is None:
            return wordcloudPlot(' '.join(self.listText), stopwords=stopwords, max_font_size=max_font_size, max_words=max_words, background_color=background_color)

        else:
            df = self.__mountDataframe(labels=labels)
            return wordcloudPlot(' '.join(df.text.tolist()), stopwords=stopwords, max_font_size=max_font_size, max_words=max_words, background_color=background_color)

    def vennWordcloudPlot(self, labels, stopwords=None):
        """
        Generate Venn Word cloud figure.

        It uses the matplotlib_venn_wordcloud and wordcloud package under the hood.

        Parameters
        ----------
        labels : list of string
            Labels to be used in the venn groups. The list has to have 2 or 3 elements.

        stopwords : list of strings, default=None
            That list is assumed to contain stop words, all of which will be removed from the resulting tokens.
        """
        return vennWordcloudPlot(self.listText, self.listLabels, labels, stopwords)
