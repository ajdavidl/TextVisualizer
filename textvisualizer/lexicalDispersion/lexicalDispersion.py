"""
Module to make a lexical dispersion plot.
"""

from yellowbrick.text import DispersionPlot


def lexicalDispersionPlot(listText, targetWords):
    """
    Make a lexical dispersion plot.

    It receives a list of text and a list of words and maek a lexical dispersion plot.
    It uses the package yellowbrick under the hood.

    Parameters
    ----------
    listText : list of strings
        The corpus of text.

    targetWords : list of strings
        The list of words to be used to plot the figure.

    """
    # Create a list of words from the corpus text
    text = [doc.split() for doc in listText]

    # Create the visualizer and draw the plot
    visualizer = DispersionPlot(targetWords)
    visualizer.fit(text)
    visualizer.show()
