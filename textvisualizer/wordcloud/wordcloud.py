"""
Module for wordcloud plot
"""

import wordcloud
import matplotlib.pyplot as plt


def wordcloudPlot(text, stopwords=None, max_font_size=50, max_words=100, background_color="white"):
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
    cloud = wordcloud.WordCloud(stopwords=stopwords, max_font_size=max_font_size,
                                max_words=max_words, background_color=background_color).generate(text.lower())

    # Display the generated image:
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
