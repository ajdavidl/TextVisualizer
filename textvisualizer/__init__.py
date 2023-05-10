"""
Module for text data visualization
"""

from .corpus.corpus import *
from .phraseNet.phraseNet import *
from .frequency.frequency import *
from .wordcloud.wordcloud import *
from .wordtree.wordtree import *
from .bubbleChart.bubbleChart import bubbleChart
from .lexicalDispersion.lexicalDispersion import lexicalDispersionPlot
from .correlation.correlation import wordCorrelationsPlot
from .bigramGraph.bigramGraph import bigramGraph

__version__ = '0.2.0'
