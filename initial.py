# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:19:44 2016

@author: Shen.Xu
"""

import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

def evaluate_features(feature_select):
    #reading pre-labeled input and splitting into lines
    posSentences = open('polarityData\\rt-polaritydata\\rt-polarity-pos.txt', 'r')
    negSentences = open('polarityData\\rt-polaritydata\\rt-polarity-neg.txt', 'r')
    posSentences = re.split(r'\n', posSentences.read())
    negSentences = re.split(r'\n', negSentences.read())
 
    posFeatures = []
    negFeatures = []
    #http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    #breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i)
        posWords = [feature_select(posWords), 'pos']
        posFeatures.append(posWords)
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i)
        negWords = [feature_select(negWords), 'neg']
        negFeatures.append(negWords)
        
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
    
    classifier = NaiveBayesClassifier.train(trainFeatures)
    
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)
        
    print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print ('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    print ('pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos']))
    print ('pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos']))
    print ('neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg']))
    print ('neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg']))
    classifier.show_most_informative_features(10)

def make_full_dict(words):
    return dict([(word, True) for word in words])
    print ('using all words as features')
    evaluate_features(make_full_dict)
    
def create_word_scores():
    #splits sentences into lines
    posSentences = open('polarityData\\rt-polaritydata\\rt-polarity-pos.txt', 'r')
    negSentences = open('polarityData\\rt-polaritydata\\rt-polarity-neg.txt', 'r')
    posSentences = re.split(r'\n', posSentences.read())
    negSentences = re.split(r'\n', negSentences.read())
 
    #creates lists of all positive and negative words
    posWords = []
    negWords = []
    for i in posSentences:
        posWord = re.findall(r"[\w']+|[.,!?;]", i)
        posWords.append(posWord)
    for i in negSentences:
        negWord = re.findall(r"[\w']+|[.,!?;]", i)
        negWords.append(negWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd.inc(word.lower())
        cond_word_fd['pos'].inc(word.lower())
    for word in negWords:
        word_fd.inc(word.lower())
        cond_word_fd['neg'].inc(word.lower())
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
 
    return word_scores

def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda ws:ws[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])
    numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
    for num in numbers_to_test:
        print( 'evaluating best %d word features' % (num))
        best_words = find_best_words(word_scores, num)
        evaluate_features(best_word_features)