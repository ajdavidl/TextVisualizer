"""
Module to make a correlation plot.
"""
from yellowbrick.text.correlation import WordCorrelationPlot


def wordCorrelationsPlot(listText, words):
    """
    Make a correlation plot.

    It receives a list of text and a list of words and make a correlation plot.
    It uses the package yellowbrick under the hood.

    Parameters
    ----------
    listText : list of strings
        The corpus of text.

    words : list of strings
        The list of words to be used to plot the figure.

    """
    # Instantiate the visualizer and draw the plot
    viz = WordCorrelationPlot(words)
    viz.fit(listText)
    viz.show()
