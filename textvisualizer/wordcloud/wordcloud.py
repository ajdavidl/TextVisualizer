"""
Module for wordcloud plot
"""

import wordcloud
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn_wordcloud import venn2_wordcloud, venn3_wordcloud


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


def vennWordcloudPlot(listText, listLabels, labels, stopwords=None):
    """
    Generate Venn Word cloud figure.

    It uses the Vmatplotlib_venn_wordcloud and wordcloud package under the hood.

    Parameters
    ----------
    listText : list of string
        List of text to be used as text source in the construction of the graph.

    listLabels : list of string with the labels of the text.
        List of the labels of each element of listText.
    
    labels : list of string
        Labels to be used in the venn groups. The list has to have 2 or 3 elements.

    stopwords : list of strings, default=None
        That list is assumed to contain stop words, all of which will be removed from the resulting tokens.
        
    """
    df = pd.DataFrame({"text" : listText, "label" : listLabels} )
    if len(labels) < 2:
        raise BaseException('Insuficient Numbers of labels. You need to give 2 or 3 labels.')
    elif len(labels) == 2 or len(labels) == 3:
        df = df[df.label.isin(labels)]
    else:
        df = df[df.label.isin(labels[:3])]
    str1 =  ' '.join(df[df['label']==labels[0]].text.tolist())
    str2 =  ' '.join(df[df['label']==labels[1]].text.tolist())
    listText2 = [str1, str2]
    if len(labels) == 3:
        str3 =  ' '.join(df[df['label']==labels[2]].text.tolist())
        listText2.append(str3)

    sets = [] 
    for string in listText2:
        # get a word list
        words = string.split(' ')
        # remove non alphanumeric characters
        words = [''.join(ch for ch in word if ch.isalnum()) for word in words]
        # convert to all lower case
        words = [word.lower() for word in words]
        sets.append(set(words))
    
    wordcloudArgs = {'max_font_size':100,'max_words' : 200, 'stopwords' : stopwords}
    
    fig, ax = plt.subplots(1,1,figsize=(30, 30))

    if len(labels) == 2:
        venn2_wordcloud(sets,
                        set_labels = labels,
                        ax=ax,
                        wordcloud_kwargs=wordcloudArgs
                        )
    elif len(labels) == 3:
        venn3_wordcloud(sets,
                        set_labels = labels,
                        ax=ax,
                        wordcloud_kwargs=wordcloudArgs
                        )
    fig.show()
