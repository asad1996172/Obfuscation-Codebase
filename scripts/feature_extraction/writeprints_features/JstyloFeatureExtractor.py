"""
This module contains a list of features that make different
feature sets in Jstylo.

Reference
---------
https://github.com/psal/jstylo/tree/master/jsan_resources/feature_sets
"""
import sys
import os
import io
import numpy as np
import nltk
import spacy
import re
from sortedcontainers import SortedDict
import os
import string
from keras.preprocessing import text
import operator
import pickle
import collections
import itertools
import pandas as pd
import math

nlp = spacy.load('en_core_web_sm')


class jstyloFeaturesExtractor:

    def __init__(self, dataset_path = None, word_unigrams_limit=60, 
        word_bigrams_limit=40, word_trigrams_limit=23, letter_bigrams_limit=39, 
        letter_trigrams_limit=20):
        if dataset_path != None:
            self.dataset_path = dataset_path
            self.corpus_top_letter_bigrams = self.__corpus_top_letter_bigrams(letter_bigrams_limit)
            self.corpus_top_letter_trigrams = self.__corpus_top_letter_trigrams(letter_trigrams_limit)
            self.corpus_vocabulary = self.__corpus_vocabulary()
            self.corpus_top_unigram_words = self.__corpus_top_word_unigrams(
                word_unigrams_limit)
            self.corpus_top_bigram_words = self.__corpus_top_word_bigrams(
                word_bigrams_limit)
            self.corpus_top_trigram_words = self.__corpus_top_word_trigrams(
                word_trigrams_limit)

    def __corpus_top_word_unigrams(self, word_unigrams_limit):
        """ Sorted list of top 60 (default) word uni-grams in corpus on the basis of frequencies"""

        corpus = []
        uni_gram_frequencies = {}
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".txt"):
                    corpus.append(os.path.join(root, file))

        for text_file in corpus:
            input_text = io.open(text_file, "r", errors="ignore").readlines()
            input_text = ''.join(str(e) + "" for e in input_text)
            input_text = str(input_text).lower()
            all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
            for word in all_words:
                if word in uni_gram_frequencies.keys():
                    uni_gram_frequencies[word] += 1
                else:
                    uni_gram_frequencies[word] = 1

        final_dict = sorted(uni_gram_frequencies.items(), key=operator.itemgetter(1), reverse=True)
        res_list = [x[0] for x in final_dict]

        return res_list[:word_unigrams_limit]

    def __corpus_top_word_bigrams(self, word_bigrams_limit):
        """ Sorted list of top 40 (default) word bi-grams in corpus on the basis of frequencies"""

        corpus = []
        bi_gram_frequencies = {}
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".txt"):
                    corpus.append(os.path.join(root, file))

        for text_file in corpus:
            input_text = io.open(text_file, "r", errors="ignore").readlines()
            input_text = ''.join(str(e) + "" for e in input_text)
            input_text = str(input_text).lower()

            all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
            all_bigrams_words = list(nltk.bigrams(all_words))
            for bigram in all_bigrams_words:

                if bigram in bi_gram_frequencies.keys():
                    bi_gram_frequencies[bigram] += 1
                else:
                    bi_gram_frequencies[bigram] = 1

        final_dict = sorted(bi_gram_frequencies.items(), key=operator.itemgetter(1), reverse=True)
        res_list = [x[0] for x in final_dict]
        return res_list[:word_bigrams_limit]

    def __corpus_top_word_trigrams(self, word_trigrams_limit):
        """ Sorted list of top 23 (default) word tri-grams in corpus on the basis of frequencies"""

        corpus = []
        tri_gram_frequencies = {}
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".txt"):
                    corpus.append(os.path.join(root, file))

        for text_file in corpus:
            input_text = io.open(text_file, "r", errors="ignore").readlines()
            input_text = ''.join(str(e) + "" for e in input_text)
            input_text = str(input_text).lower()

            all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
            all_trigrams_words = list(nltk.trigrams(all_words))
            for trigram in all_trigrams_words:

                if trigram in tri_gram_frequencies.keys():
                    tri_gram_frequencies[trigram] += 1
                else:
                    tri_gram_frequencies[trigram] = 1

        final_dict = sorted(tri_gram_frequencies.items(), key=operator.itemgetter(1), reverse=True)
        res_list = [x[0] for x in final_dict]
        return res_list[:word_trigrams_limit]

    def __corpus_top_letter_bigrams(self, letter_bigrams_limit):
        """ Top 39 (default) letter bigrams from the whole corpus """

        regex = re.compile('[A-Za-z]+')

        corpus = []
        bigram_dictionary = {}
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".txt"):
                    corpus.append(os.path.join(root, file))

        for text_file in corpus:
            input_text = io.open(text_file, "r", errors="ignore").readlines()
            input_text = ''.join(str(e) + "" for e in input_text)
            input_text = str(input_text).lower()
            all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
            for word in all_words:
                all_bigrams = list(nltk.bigrams(word))
                for bigram in all_bigrams:
                    bigram = ''.join(bigram)

                    if regex.fullmatch(bigram):
                        if bigram in bigram_dictionary.keys():
                            bigram_dictionary[bigram] = bigram_dictionary[bigram] + 1
                        else:
                            bigram_dictionary[bigram] = 1

        final_dict = sorted(bigram_dictionary.items(), key=operator.itemgetter(1), reverse=True)
        res_list = [x[0] for x in final_dict]
        return res_list[:letter_bigrams_limit]
    
    def __corpus_vocabulary(self):
        """ A set of unique words in the corpus """
        corpus = []
        corpus_vocabulary = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".txt"):
                    corpus.append(os.path.join(root, file))

        for text_file in corpus:
            input_text = io.open(text_file, "r", errors="ignore").readlines()
            input_text = ''.join(str(e) + "" for e in input_text)
            input_text = str(input_text).lower()
            all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
            for word in all_words:
                corpus_vocabulary.append(word)
        return len(list(set(corpus_vocabulary)))

    def __corpus_top_letter_trigrams(self, letter_trigrams_limit):
        """ Top 20 (default) letter bigrams from the whole corpus """

        regex = re.compile('[A-Za-z]+')

        corpus = []
        trigram_dictionary = {}
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".txt"):
                    corpus.append(os.path.join(root, file))

        for text_file in corpus:
            input_text = io.open(text_file, "r", errors="ignore").readlines()
            input_text = ''.join(str(e) + "" for e in input_text)
            input_text = str(input_text).lower()
            all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
            for word in all_words:
                all_trigrams = list(nltk.trigrams(word))
                for trigram in all_trigrams:
                    trigram = ''.join(trigram)

                    if regex.fullmatch(trigram):
                        if trigram in trigram_dictionary.keys():
                            trigram_dictionary[trigram] = trigram_dictionary[trigram] + 1
                        else:
                            trigram_dictionary[trigram] = 1

        final_dict = sorted(trigram_dictionary.items(), key=operator.itemgetter(1), reverse=True)
        res_list = [x[0] for x in final_dict]

        return res_list[:letter_trigrams_limit]

    """ All Write-prints Features """

    """ Lexical => Character-based features """
    def total_number_of_characters(self, input_text):
        """ The total number of characters in the document """

        return len(input_text)
    
    def alphabetic_characters_percentage(self, input_text):
        """ Percentage of letters out of the total character count in the document """

        alphabetic_characters = len(re.findall(r"[A-Za-z]", input_text))
        chars_count = self.total_number_of_characters(input_text)
        try:
            return round(alphabetic_characters / chars_count, 2)
        except ZeroDivisionError:
            return 0
    
    def uppercase_characters_percentage(self, input_text):
        """ Percentage of Uppercase letters out of the total character count in the document """

        chars_count = self.total_number_of_characters(input_text)
        uppercase_characters = len(re.findall(r"[A-Z]", input_text))

        try:
            return round(uppercase_characters / chars_count, 2)
        except ZeroDivisionError:
            return 0
    
    def digit_characters_percentage(self, input_text):
        """ Percentage of digits out of the total character count in the document """

        chars_count = self.total_number_of_characters(input_text)
        digits_count = len(re.findall(r"[0-9]", input_text))
        try:
            return round(digits_count / chars_count, 2)
        except ZeroDivisionError:
            return 0
    
    def whitespace_characters_percentage(self, input_text):
        """ Percentage of white spaces out of the total character count in the document """

        chars_count = self.total_number_of_characters(input_text)
        whitespaces_count = len(re.findall(r"[\s]", input_text))
        try:
            return round(whitespaces_count / chars_count, 2)
        except ZeroDivisionError:
            return 0
    
    def tabspaces_characters_percentage(self, input_text):
        """ Percentage of Tab Spaces out of the total character count in the document """

        tabspaces_characters = len(re.findall(r"[\t]", input_text))
        chars_count = self.total_number_of_characters(input_text)
        try:
            return round(tabspaces_characters / chars_count, 2)
        except ZeroDivisionError:
            return 0
    
    def frequency_of_letters(self, input_text):
        """ Frequency of letters (a-z, case insensitive) """

        input_text = str(input_text).lower()
        letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u',
                   'v', 'w', 'x', 'y', 'z'}
        letters_frequency_dict = {}
        for letter in letters:
            letters_frequency_dict[letter] = 0

        for letter in str(input_text):
            if letter in letters:
                letters_frequency_dict[letter] = letters_frequency_dict[letter] + 1

        letter_counts = SortedDict(letters_frequency_dict)
        letter_counts = np.array(letter_counts.values())

        return list(letter_counts)
    
    def frequency_of_top_letter_bigrams(self, input_text):
        """ Frequency of top 39 (default) letter bi-grams (e.g. aa, ab, etc.), case insensitive and only within words. """

        bigrams = self.corpus_top_letter_bigrams
        bigrams = set(bigrams)
        input_text = str(input_text).lower()

        letters_frequency_dict = {}
        for bigram in bigrams:
            letters_frequency_dict[bigram] = 0

        all_words = text.text_to_word_sequence(input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
        for word in all_words:
            all_bigrams = list(nltk.bigrams(word))
            for bigram in all_bigrams:
                bigram = ''.join(bigram)
                if bigram in bigrams:
                    letters_frequency_dict[bigram] = letters_frequency_dict[bigram] + 1

        bigram_counts = SortedDict(letters_frequency_dict)
        bigram_counts = np.array(bigram_counts.values())

        return list(bigram_counts)
    
    def frequency_of_top_letter_trigrams(self, input_text):
        """ Frequency of top 20 (default) letter tri-grams (e.g. aaa, aab, etc.), case insensitive and only within words. """

        trigrams = self.corpus_top_letter_trigrams
        trigrams = set(trigrams)
        input_text = str(input_text).lower()

        letters_frequency_dict = {}
        for trigram in trigrams:
            letters_frequency_dict[trigram] = 0

        all_words = text.text_to_word_sequence(input_text, filters=",.?!\"'`;:-()&$", lower=True, split=" ")
        for word in all_words:
            all_trigrams = list(nltk.trigrams(word))
            for trigram in all_trigrams:
                trigram = ''.join(trigram)
                if trigram in trigrams:
                    letters_frequency_dict[trigram] = letters_frequency_dict[trigram] + 1

        trigram_counts = SortedDict(letters_frequency_dict)
        trigram_counts = np.array(trigram_counts.values())

        return list(trigram_counts)
    
    def special_characters(self, input_text):
        """ Frequencies of special characters, e.g., ~,@etc. """

        special_characters_file = open(
            "{}/writeprints_special_chars.txt".format(os.environ['WRITEPRINTS_RESOURCES']), "r")
        special_characters = special_characters_file.readlines()
        special_characters = [s.strip("\n") for s in special_characters]
        special_characters_frequency_dict = {}
        for i in range(0, len(special_characters)):
            special_char = special_characters[i]
            special_characters_frequency_dict[special_char] = 0

        for char in str(input_text):
            if char in special_characters_frequency_dict.keys():
                special_characters_frequency_dict[char] = special_characters_frequency_dict[char] + 1

        spec_chars_len_counts = []
        for i in range(0, len(special_characters)):
            special_char = special_characters[i]
            spec_chars_len_counts.append(special_characters_frequency_dict[special_char])

        special_characters_file.close()
        return list(spec_chars_len_counts)
    
    def digits(self, input_text):
        """ Frequency of digits (0-9) """

        digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        digits_counts = {}
        for digit in digits:
            digits_counts[str(digit)] = 0

        alldigits = re.findall('\d', input_text)
        for digit in alldigits:
            digits_counts[digit] += 1

        digits_counts = SortedDict(digits_counts)
        digits_counts = np.array(digits_counts.values())
        return digits_counts
    
    def two_digit_numbers(self, input_text):
        """ Frequency of digits (0-9) """
        digits = []
        for i in range(10, 100):
            digits.append(i)
        digits = set(digits)

        digits_counts = {}
        for digit in digits:
            digits_counts[str(digit)] = 0

        alldigits = re.findall('\d\d', input_text)
        for digit in alldigits:
            try:
                digits_counts[digit] += 1
            except KeyError as ke:
                print(ke)

        digits_counts = SortedDict(digits_counts)
        digits_counts = np.array(digits_counts.values())
        return digits_counts
    
    def three_digit_numbers(self, input_text):
        """ Frequency of digits (0-9) """
        digits = []
        for i in range(100, 1000):
            digits.append(i)
        digits = set(digits)

        digits_counts = {}
        for digit in digits:
            digits_counts[str(digit)] = 0

        alldigits = re.findall('\d\d\d', input_text)
        for digit in alldigits:
            try:
                digits_counts[digit] += 1
            except KeyError as ke:
                print(ke)
        digits_counts = SortedDict(digits_counts)
        digits_counts = np.array(digits_counts.values())
        return digits_counts

    """ Lexical => Word-based features """
    def total_words(self, input_text):
        """ Total number of words in the text """

        return len(text.text_to_word_sequence(input_text, filters="", lower=True, split=" "))
    
    def frequency_of_large_words(self, input_text):
        """ Frequency of words having length >= 4 characters """
        words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        frequency = 0
        for word in words:
            if len(word) >= 4:
                frequency += 1

        return frequency

    def frequency_of_short_words(self, input_text):
        """ Frequency of words having length < 4 characters """
        words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        frequency = 0
        for word in words:
            if len(word) < 4:
                frequency += 1

        return frequency
        
        
    def characters_in_words_per_total_characters(self, input_text):
        """ Average number of characters per word in the document """
        words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        word_characters = 0
        for word in words:
            word_characters += len(word)
        try:
            return round(word_characters / self.total_number_of_characters(input_text), 2)
        except ZeroDivisionError:
            return 0
    
    def average_word_length(self, input_text):
        """ Average number of characters per word """

        words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        frequencies = []
        for word in words:
            frequencies.append(len(word))

        if len(frequencies) != 0:
            return np.mean(frequencies)
        else:
            return 0
    
    def average_sentence_length_characters(self, input_text):
        """ Average sentence length in terms of characters """
        frequencies = []
        sent_text = nltk.sent_tokenize(input_text)
        for sent in sent_text:
            frequencies.append(len(str(sent)))

        if len(frequencies) != 0:
            return np.mean(frequencies)
        else:
            return 0
    
    def average_sentence_length_words(self, input_text):
        """ Average sentence length in terms of words """
        frequencies = []
        sent_text = nltk.sent_tokenize(input_text)
        for sent in sent_text:
            frequencies.append(self.total_words(str(sent)))

        if len(frequencies) != 0:
            return np.mean(frequencies)
        else:
            return 0
    
    def average_unique_words(self, input_text):
        """ Total number of unique words per total words """
        words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        unique_words = len(set(words))
        try:
            return round(unique_words / self.total_words(input_text), 2)
        except ZeroDivisionError:
            return 0
    
    def hapax_legomena(self, input_text):
        """ Frequency of once occuring words """
        words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        frequencies = nltk.FreqDist(word for word in words)
        hapaxes = [key for key, val in frequencies.items() if val == 1]

        if len(hapaxes) != 0:
            return len(hapaxes)
        else:
            return 0
    
    def hapax_dislegomena(self, input_text):
        """ Frequency of twice occuring words """
        words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        frequencies = nltk.FreqDist(word for word in words)
        hapaxes = [key for key, val in frequencies.items() if val == 2]

        if len(hapaxes) != 0:
            return len(hapaxes)
        else:
            return 0
    
    def yules_K_measure(self, input_text):
        """ Calculate Yule’s K (1944). """

        tokens = re.split(r"[^0-9A-Za-z\-'_]+", input_text)
        num_tokens = len(tokens)
        bow = collections.Counter(tokens)
        freq_spectrum = pd.Series(collections.Counter(bow.values()))

        a = freq_spectrum.index.values / num_tokens
        b = 1 / num_tokens
        return 10 ** 4 * ((freq_spectrum.values * a ** 2) - b).sum()
    
    def simpsons_D_measure(self, input_text):
        """ simpson's D measure """
        tokens = re.split(r"[^0-9A-Za-z\-'_]+", input_text)
        num_tokens = len(tokens)
        bow = collections.Counter(tokens)
        freq_spectrum = pd.Series(collections.Counter(bow.values()))

        a = freq_spectrum.values / num_tokens
        b = freq_spectrum.index.values - 1
        return (freq_spectrum.values * a * (b / (num_tokens - 1))).sum()
    
    def sichels_S_measure(self, input_text):
        """ Calculate Sichel’s S (1975). """
        tokens = re.split(r"[^0-9A-Za-z\-'_]+", input_text)
        num_tokens = len(tokens)
        bow = collections.Counter(tokens)
        freq_spectrum = pd.Series(collections.Counter(bow.values()))

        num_types = self.corpus_vocabulary
        return freq_spectrum[2] / num_types
    
    def brunets_W_measure(self, input_text):
        """ Calculate Brunet’s W (1978). """

        tokens = re.split(r"[^0-9A-Za-z\-'_]+", input_text)
        num_tokens = len(tokens)
        bow = collections.Counter(tokens)
        freq_spectrum = pd.Series(collections.Counter(bow.values()))

        num_types = self.corpus_vocabulary

        a = -0.172
        return num_tokens ** (num_types ** -a)
    
    def honores_R_measure(self, input_text):
        """ Calculate honore's_R measure """

        tokens = re.split(r"[^0-9A-Za-z\-'_]+", input_text)
        num_tokens = len(tokens)
        bow = collections.Counter(tokens)
        freq_spectrum = pd.Series(collections.Counter(bow.values()))

        num_types = self.corpus_vocabulary

        return 100 * (math.log(num_tokens)
                      / (1 - ((freq_spectrum[1]) / (num_types))))
    
    def word_lengths_frequency(self, input_text, length_limit=21):
        """ Frequency of words of different lengths (excluding punctuation) """

        lengths = range(1, length_limit)
        word_length_frequencies = {}
        for l in lengths:
            word_length_frequencies[l] = 0

        words = text.text_to_word_sequence(input_text, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True,
                                           split=" ")
        for word in words:
            if (word[0] in string.punctuation) and (word[-1] in string.punctuation):
                temp_word = word[1:-1]
            elif word[0] in string.punctuation:
                temp_word = word[1:]
            elif word[-1] in string.punctuation:
                temp_word = word[:-1]
            else:
                temp_word = word

            word_length = len(temp_word)
            if word_length in word_length_frequencies:
                word_length_frequencies[word_length] = word_length_frequencies[word_length] + 1

        word_len_counts = SortedDict(word_length_frequencies)
        word_len_counts = np.array(word_len_counts.values())

        return word_len_counts
    
    def misspelled_words(self, input_text):
        """ Frequencies of misspelled words out of a list of 5,513 common misspellings """

        misspelled_words = open(
            "{}/writeprints_misspellings.txt".format(os.environ['WRITEPRINTS_RESOURCES']), "r").readlines()
        misspelled_words = [f.strip("\n") for f in misspelled_words]

        misspelled_words_frequency_dict = {}
        for i in range(0, len(misspelled_words)):
            misspelled_word = misspelled_words[i]
            misspelled_words_frequency_dict[misspelled_word] = 0

        input_text = input_text.lower()
        words = text.text_to_word_sequence(input_text, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True,
                                           split=" ")

        for word in words:
            if (word[0] in string.punctuation) and (word[-1] in string.punctuation):
                temp_word = word[1:-1]
            elif word[0] in string.punctuation:
                temp_word = word[1:]
            elif word[-1] in string.punctuation:
                temp_word = word[:-1]
            else:
                temp_word = word

            if temp_word in misspelled_words:
                misspelled_words_frequency_dict[temp_word] = misspelled_words_frequency_dict[temp_word] + 1


        mispel_words_len_counts = SortedDict(misspelled_words_frequency_dict)
        mispel_words_len_counts = np.array(mispel_words_len_counts.values())

        return mispel_words_len_counts
    
    def words(self, input_text):
        """ Frequencies of top 60 (default) words in the text, case insensitive and without punctuations """

        all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        unigram_frequencies = [0] * len(self.corpus_top_unigram_words)
        for word in all_words:
            try:
                unigram_frequencies[self.corpus_top_unigram_words.index(word)] += 1
            except ValueError:
                pass
        return unigram_frequencies
    
    def word_bigrams(self, input_text):
        """ Frequencies of top 40 (default) word bigrams in the text, case insensitive and without punctuations """

        all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        all_bigrams_words = list(nltk.bigrams(all_words))
        bigram_frequencies = [0] * len(self.corpus_top_bigram_words)
        for bigram_word in all_bigrams_words:
            try:
                bigram_frequencies[self.corpus_top_bigram_words.index(bigram_word)] += 1
            except ValueError:
                pass
        return bigram_frequencies
    
    def word_trigrams(self, input_text):
        """ Frequencies of top 23 (default) word trigrams in the text, case insensitive and without punctuations """

        all_words = text.text_to_word_sequence(input_text, filters="", lower=True, split=" ")
        all_trigrams_words = list(nltk.trigrams(all_words))
        trigram_frequencies = [0] * len(self.corpus_top_trigram_words)
        for trigram_word in all_trigrams_words:
            try:
                trigram_frequencies[self.corpus_top_trigram_words.index(trigram_word)] += 1
            except ValueError:
                pass
        return trigram_frequencies

    """ Syntactic features """
    
    def function_words(self, input_text):
        """ 512 common function words used by Koppel et al. in Koppel, 2005 """

        function_words = open(
            "{}/functionWord.txt".format(os.environ['WRITEPRINTS_RESOURCES']), "r").readlines()
        function_words = [f.strip("\n") for f in function_words]

        function_words_frequency_dict = {}
        for i in range(0, len(function_words)):
            function_word = function_words[i]
            function_words_frequency_dict[function_word] = 0

        input_text = input_text.lower()
        words = text.text_to_word_sequence(input_text, filters=' !#$%&()*+,-./:;<=>?@[\\]^_{|}~\t\n"', lower=True,
                                           split=" ")

        for word in words:
            if (word[0] in string.punctuation) and (word[-1] in string.punctuation):
                temp_word = word[1:-1]
            elif word[0] in string.punctuation:
                temp_word = word[1:]
            elif word[-1] in string.punctuation:
                temp_word = word[:-1]
            else:
                temp_word = word

            if temp_word in function_words:
                function_words_frequency_dict[temp_word] = function_words_frequency_dict[temp_word] + 1

        func_words_len_counts = SortedDict(function_words_frequency_dict)
        func_words_len_counts = np.array(func_words_len_counts.values())

        return func_words_len_counts
    
    def punctuation(self, input_text):
        """ Frequencies of punctuations, e.g. . , ! etc. """

        punctuations = open("{}/writeprints_punctuation.txt".format(
            os.environ['WRITEPRINTS_RESOURCES']), "r").readlines()
        punctuations = [s.strip("\n") for s in punctuations]
        punctuations_frequency_dict = {}
        for i in range(0, len(punctuations)):
            punctuation = punctuations[i]
            punctuations_frequency_dict[punctuation] = 0

        for char in str(input_text):
            if char in punctuations_frequency_dict.keys():
                punctuations_frequency_dict[char] = punctuations_frequency_dict[char] + 1

        punctuation_len_counts = SortedDict(punctuations_frequency_dict)
        punctuation_len_counts = np.array(punctuation_len_counts.values())

        return punctuation_len_counts
    
    def POS_tags(self, input_text):
        """ Frequencies of Part-Of_Speech tags extracted by spacy """

        pos_tags = []
        doc = nlp(str(input_text))
        for token in doc:
            pos_tags.append(str(token.pos_))

        tagset = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                  'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
        tags = [tag for tag in pos_tags]
        return list(tuple(tags.count(tag) for tag in tagset))
    
    def POS_bigrams(self, input_text):
        """ Frequencies of Part-Of_Speech bigrams extracted by spacy """

        pos_tags = []
        doc = nlp(str(input_text))
        for token in doc:
            pos_tags.append(str(token.pos_))

        tagset = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                  'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
        tagset = list(itertools.combinations(tagset, 2))
        tagset_freq = [0] * len(tagset)

        for i in range(1, len(pos_tags)):
            if ((pos_tags[i - 1], pos_tags[i]) in tagset) or ((pos_tags[i], pos_tags[i - 1]) in tagset):
                try:
                    tagset_freq[tagset.index((pos_tags[i - 1], pos_tags[i]))] += 1
                except:
                    tagset_freq[tagset.index((pos_tags[i], pos_tags[i - 1]))] += 1

        final = [val for val in tagset_freq]
        return final
    
    def POS_trigrams(self, input_text):
        """ Frequencies of Part-Of_Speech trigrams extracted by spacy """

        pos_tags = []
        doc = nlp(str(input_text))
        for token in doc:
            pos_tags.append(str(token.pos_))

        tagset = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                  'PUNCT', 'SCONJ', 'SYM', 'VERB', 'SPACE', 'X']
        tagset = list(itertools.combinations(tagset, 3))
        tagset_freq = [0] * len(tagset)

        for i in range(2, len(pos_tags)):
            all_posibilities = list(itertools.permutations([pos_tags[i - 2], pos_tags[i - 1], pos_tags[i]]))
            for possibility in all_posibilities:
                if possibility in tagset:
                    tagset_freq[tagset.index(possibility)] += 1

        final = [val for val in tagset_freq]
        return final

