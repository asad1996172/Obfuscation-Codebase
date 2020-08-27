"""
Writeprints Static feature set from Jstylo

Reference
---------
https://www1.icsi.berkeley.edu/~sadia/papers/adversarial_stylometry.pdf
"""
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import nltk
import spacy
import re
from sortedcontainers import SortedDict
import os
from keras.preprocessing import text


def calculateFeatures(input_text, jfe):

    featureList = []

    """ Lexical -- Word Level """
    featureList.extend([jfe.total_words(input_text)])
    featureList.extend([jfe.average_word_length(input_text)])
    featureList.extend([jfe.frequency_of_short_words(input_text)])
    
    """ Lexical -- Character Level """
    featureList.extend([jfe.total_number_of_characters(input_text)])
    featureList.extend([jfe.digit_characters_percentage(input_text)])
    featureList.extend([jfe.uppercase_characters_percentage(input_text)])

    """ Lexical -- Special Characters """
    featureList.extend(jfe.special_characters(input_text))

    """ Lexical -- Letters """
    featureList.extend(jfe.frequency_of_letters(input_text))

    """ Lexical -- Digits """
    featureList.extend(jfe.digits(input_text))

    """ Lexical -- Vocabulary Richness """
    featureList.extend([jfe.hapax_legomena(input_text)])
    featureList.extend([jfe.hapax_dislegomena(input_text)])

    """ Lexical -- corpus level """
    featureList.extend(jfe.frequency_of_top_letter_bigrams(input_text))
    featureList.extend(jfe.frequency_of_top_letter_trigrams(input_text))

    """ Syntactic -- Function Words """
    featureList.extend(jfe.function_words(input_text))

    """ Syntactic -- POS tags """
    featureList.extend(jfe.POS_tags(input_text))

    """ Syntactic -- Punctuation """
    featureList.extend(jfe.punctuation(input_text))


    return featureList

