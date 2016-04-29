# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:15:47 2016

@author: Shen.Xu
"""


import pandas as pd
import numpy as np
from sklearn.datasets import load_files
import csv, re, nltk, sys, random, codecs
import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import glob
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import jieba
from pandas import ExcelWriter
import string
from sklearn import svm

   

def read_training_data():      
    valid_comments=[]    
    valid_sentiment=[]
    utf_comments=[]
    post_pos=[]  
    post_neg=[]
    post_neu=[]
    sent_list=['neg', 'neu', 'pos','POS','NEG','NEU']
    pos_list=['pos','Pos','POS','Positive']
    neg_list=['neg','Neg','NEG','Negitive']
    neu_list=['neu','Neu','NEU','Neutual']
    path =r'E:/Dropbox/Python codes/interview/trainingdata/'    
    files=glob.glob(path + "/*.csv")    
    files_xlsx= glob.glob(path + "/*.xlsx")
    fieldnames = ['Post#', 'Post','Post Relevance','Sentiment','AD']
    file_list=[]    
    for fl in files:
        df = pd.read_csv(fl,encoding='utf-8')        
        file_list.append(df)
    for fl in files_xlsx:
        df = pd.read_excel(fl,encoding='gbk')
        file_list.append(df)
    training1=pd.concat(file_list,axis=0)
    training=training1.fillna('n')       
    relevant=training[training['Sentiment'].str.contains('neu|neg|pos',case=False)][['Post','Sentiment']]
    data_captured=relevant[relevant['Sentiment'].isin(sent_list)][['Post','Sentiment']]
    post1=data_captured['Post']    
    sen1=data_captured['Sentiment']    
    for row in post1:
        comments=row.encode('utf-8','replace')        
        utf_comments.append(comments)
    for row in sen1:
        sentiment=row.lower()
        valid_sentiment.append(sentiment) 
    post_pos=relevant[relevant['Sentiment'].isin(pos_list)][['Post','Sentiment']]
    #post_pos=post_pos.append(post_pos1)
    post_neg=relevant[relevant['Sentiment'].isin(neg_list)][['Post','Sentiment']]
        #post_neg=post_neg.append(post_neg1)
    post_neu=relevant[relevant['Sentiment'].isin(neu_list)][['Post','Sentiment']]
        #post_neu=post_neg.append(post_neu1)
    
    print 'pos: %d, neg: %d, neu: %d' %(len(post_pos), len(post_neg), len(post_neu))
    print 'Training Data Parsed:' 
    print (len(utf_comments) == len(valid_sentiment))
    for i in xrange(len(utf_comments)):
        valid_comment=utf_comments[i].decode('utf-8')
        valid_comments.append(valid_comment)
    return valid_comments,valid_sentiment
    
def read_predict():
    test_comments=[]    
    test_sentiment=[] 
    post_pos=[]  
    post_neg=[]
    post_neu=[]
    sent_list=['neg', 'neu', 'pos','POS','NEG','NEU']
    pos_list=['pos','Pos','POS','Positive']
    neg_list=['neg','Neg','NEG','Negitive']
    neu_list=['neu','Neu','NEU','Neutual']
    path =r'E:/Dropbox/Python codes/interview/test/'    
    files=glob.glob(path + "/*.csv")    
    file_list=[]    
    for fl in files:
        df = pd.read_csv(fl)
        file_list.append(df)
    training1=pd.concat(file_list)
    training=training1.fillna('n')    
    relevant=training[training['Sentiment'].str.contains('neu|neg|pos',case=False)][['Post','Sentiment']]
    data_captured=relevant[relevant['Sentiment'].isin(sent_list)][['Post','Sentiment']]
    post1=data_captured['Post']    
    sen1=data_captured['Sentiment']    
    for row in post1:
        comments=row.decode('utf-8')        
        test_comments.append(comments)
    for row in sen1:
        sentiment=row.lower()
        test_sentiment.append(sentiment) 
    post_pos=relevant[relevant['Sentiment'].isin(pos_list)][['Post','Sentiment']]
    #post_pos=post_pos.append(post_pos1)
    post_neg=relevant[relevant['Sentiment'].isin(neg_list)][['Post','Sentiment']]
        #post_neg=post_neg.append(post_neg1)
    post_neu=relevant[relevant['Sentiment'].isin(neu_list)][['Post','Sentiment']]
        #post_neu=post_neg.append(post_neu1)
        
    print 'pos: %d, neg: %d, neu: %d' %(len(post_pos), len(post_neg), len(post_neu))
    print (len(test_comments))
    
    return test_comments,test_sentiment
    
def preprocess():
    
    return None
    
def processing_comments(text):
        
    brand_name =[u'清扬',u'海飞丝',u'欧莱雅',u'潘婷',u'飘柔',u'施华蔻',u'维达沙宣',u'沙宣',
                 u'Clear',u'Head&shoulders',u'Oreal',u'Pantene',u'Rejoice',u'Schwarzkopf',u'Vidal Sassoon']    
    for brand in brand_name:
        text = re.sub(brand,' __BRAND__ ',text, re.U | re.I)
    
    
    URL = r"""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"""
    RE_URL = re.compile(URL, re.VERBOSE | re.I | re.UNICODE)   
    urls = RE_URL.findall(text)
    if urls:
        text = RE_URL.sub("", text)          
    
    
    clean_text = re.sub("[\W\d]+", "", text, flags = re.U)
    #clean_text=text.translate(identify,delCStr)
    return clean_text

def feature_extraction():
    valid_comments,valid_sentiment=read_training_data()    
    sent_list=['neg', 'neu', 'pos','POS','NEG','NEU']    
    count_vect = CountVectorizer(encoding = u'utf-8',
        decode_error = 'strict',
        strip_accents = None,
        analyzer = u'word',
        preprocessor = None,
        tokenizer = segementation,
        ngram_range = (1,3),
        stop_words = None,
        lowercase = False,
        token_pattern = None,
        max_df = 1.0,
        min_df = 1,
        max_features = None,
        vocabulary = None,
        binary = False,
        dtype = np.float64,               
        )     
    trans_vect = TfidfVectorizer(
        encoding = u'utf-8',
        decode_error = 'strict',
        strip_accents = None,
        analyzer = u'word',
        preprocessor = None,
        tokenizer = segementation,
        ngram_range = (1,3),
        stop_words = None,
        lowercase = False,
        token_pattern = None,
        max_df = 1.0,
        min_df = 1,
        max_features = None,
        vocabulary = None,
        binary = False,
        dtype = np.float64,
        norm = u'l2',
        use_idf = True,
        smooth_idf = True,
        sublinear_tf = True
        )    
    #commentsss=[]
    #for i in xrange(len(valid_comments)):
        #valid_comment=valid_comments[i].decode('utf-8')
        #commentsss.append(valid_comment)
    X_train_counts = count_vect.fit_transform(valid_comments)    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)    
    
    X_train_tf = tf_transformer.transform(X_train_counts)    
    X_train_tfidf = trans_vect.fit_transform(valid_comments)    
    
    terms=count_vect.get_feature_names()
    freqs=X_train_tf.sum(axis=0).A1
    terms_trans=trans_vect.get_feature_names()
    result=zip(terms,freqs)
    
        
    return result,valid_sentiment,X_train_tfidf,X_train_counts,count_vect,trans_vect

def segementation(text):
    stopwords = []
    clean_text=processing_comments(text)
    segments = [s for s in jieba.cut(clean_text.encode("utf-8")) if s]
    #feats = fe.feats + [w for w in segments if w.lower() not in stopwords]
    feats = [w for w in segments if w.lower() not in stopwords]
    #delCStr = '《》（）￥【】，。！？：；、'.decode()     
    #feats=[w for w in feats if w not in delCStr]    
    return feats

def export2excel():
    #result,valid_sentiment, X_train_tfidf=feature_extraction()   
    test_comments, predicted_nb,t_pred=naive_bayesian_multi()
    predicted_sgd=  stochastic_gradient_descent()
    predicted_sgd_gs=grid_search()
    #out_path= "outfile.csv"
    #out_file = open(out_path, 'wb')
    path =r'E:/Dropbox/Python codes/interview/results/'
    fieldnames = ['Bigram', 'Score']
    #writer = csv.writer(out_file, lineterminator='\n')
    #for val in data:                
        #writer.writerows([val])
    
    #my_df=pd.DataFrame(result)
    #my_df.columns=fieldnames
    #my_df.to_csv('ngram.csv',encoding='utf-8',index=False,header=fieldnames)
    #my_df.to_excel(path+'ngram.xlsx')
        
    posts=[]    
    for i in xrange(len(test_comments)):
        comment=test_comments[i].encode('gbk','replace')
        posts.append(comment)    
    df1=pd.DataFrame(posts,columns=['Post'])
    df2=pd.DataFrame(predicted_nb,columns=['Predicted Sentiment NB_MULTI'])      
    df3=pd.DataFrame(predicted_sgd,columns=['Predicted Sentiment SGD'])
    df4=pd.DataFrame(predicted_sgd_gs,columns=['Predicted Sentiment SGD_GS'])
    df5=pd.DataFrame(t_pred,columns=['Actual Sentiment'])
    list=[df1,df2,df3,df4,df5]    
    df=pd.concat([df1,df2,df3,df4,df5],axis=1)
    
    df.to_excel(path+'results.xlsx')
    df.to_csv(path+'results.csv')    


def naive_bayesian_multi():
    valid_comments,valid_sentiment=read_training_data()    
    result,valid_sentiment, X_train_tfidf, X_train_counts,count_vect,trans_vect=feature_extraction()
    test_comments,test_sentiment=read_predict()
    t_pred=test_sentiment
    clf = MultinomialNB().fit(X_train_tfidf, valid_sentiment)
    """
    docs_new = np.array(['欧莱雅好,自信','欧莱雅不好用','欧莱雅非常不好','用海飞丝,就是那么自信'])
    comments_new=[]    
    for row in docs_new:
        comments=row.decode()        
        comments_new.append(comments) 
        
    #X_train_counts = count_vect.fit_transform(valid_comments)
    X_train_counts.shape    
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)    
    X_train_tf = tf_transformer.transform(X_train_counts)    
    X_train_tf.shape    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape        
    X_new_counts = count_vect.transform(comments_new)       
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)    
    predicted_nb = clf.predict(X_new_tfidf)
    for doc, category in zip(comments_new, predicted_nb):
        print('%r => %s' % (doc.decode('utf-8'), category))    
    t_pred=['pos','neg','neg','pos']
    """
          
    X_new_tfidf = trans_vect.transform(test_comments)    
    predicted = clf.predict(X_new_tfidf)
    #for doc, category in zip(test_comments, predicted_nb):
        #print('%r => %s' % (doc, category))
       
    
    
    predicted_nb=[]    
    for i in xrange(len(predicted)):
        comment=predicted[i].encode('gbk','replace')
        predicted_nb.append(comment)
    np.mean(np.array(predicted_nb) == np.array(t_pred))            
    print(classification_report(t_pred, predicted_nb))
    print confusion_matrix(t_pred, predicted_nb)
    return test_comments,predicted_nb,t_pred
    
def stochastic_gradient_descent():
    valid_comments,valid_sentiment=read_training_data()    
    result,valid_sentiment, X_train_tfidf, X_train_counts,count_vect,trans_vect=feature_extraction()
    test_comments,test_sentiment=read_predict()
    t_pred=test_sentiment    
    clf=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
    clf1=clf.fit(X_train_tfidf, valid_sentiment)
    X_new_tfidf = trans_vect.transform(test_comments)    
    predicted = clf1.predict(X_new_tfidf)
    #for doc, category in zip(test_comments, predicted_svm):
        #print('%r => %s' % (doc, category))    
    predicted_sgd=[]    
    for i in xrange(len(predicted)):
        comment=predicted[i].encode('gbk','replace')
        predicted_sgd.append(comment)    
    np.mean(np.array(predicted_sgd) == np.array(t_pred))
    print(classification_report(t_pred,predicted_sgd))
    print confusion_matrix(t_pred, predicted_sgd)
    
    """
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])


    text_clf.fit(valid_comments, valid_sentiment)
    """
    return predicted_sgd
def grid_search():    
    valid_comments,valid_sentiment=read_training_data()    
    result,valid_sentiment, X_train_tfidf, X_train_counts,count_vect,trans_vect=feature_extraction()
    test_comments,test_sentiment=read_predict()
    t_pred=test_sentiment    
    
    clf=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
    clf1=clf.fit(X_train_tfidf, valid_sentiment)
    
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])

    
    
    parameters = {'vect__ngram_range': [(1, 1), (1, 2),(1,3)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-5),}
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(valid_comments, valid_sentiment)
    
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    predicted=gs_clf.predict(test_comments)
    #for doc, category in zip(test_comments, predicted_gs):
        #print('%r => %s' % (doc, category))
    predicted_sgd_gs=[]    
    for i in xrange(len(predicted)):
        comment=predicted[i].encode('gbk','replace')
        predicted_sgd_gs.append(comment)      
    np.mean(np.array(predicted_sgd_gs) == np.array(t_pred))
    print(classification_report(t_pred,predicted_sgd_gs))
    print confusion_matrix(t_pred,predicted_sgd_gs) 
    """    
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
"""
    return predicted_sgd_gs



def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

def getfeaturevector(text):
    featureVector = []
    stopWords=[]
    #split tweet into words
    words=segementation(text)
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search('[\x80-\xff].', w.encode('gbk','replace'))
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

def make_feature_list():
    valid_comments,valid_sentiment=read_training_data()    
    stopWords=[]    
    feature_list=[]
    for line in valid_comments:
        words=getfeaturevector(line)
        for w in words:            
            w = replaceTwoOrMore(w)            
            if(w in stopWords):
                continue
            else:
                feature_list.append(w.lower())
    return feature_list

def featurewithsen():
    featurewithsen=[]
    valid_comments,valid_sentiment=read_training_data()
    for line,row in zip(valid_comments, valid_sentiment):
        featurevector=getfeaturevector(line)
        sentiment=row
        featurewithsen.append((featurevector,sentiment))
    return featurewithsen

def extract_features(text):
    tweet_words = set(text)
    features = {}
        
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
def naive_bayes():
    valid_comments,valid_sentiment=read_training_data()    
    result,valid_sentiment, X_train_tfidf, X_train_counts,count_vect,trans_vect=feature_extraction()
    test_comments,test_sentiment=read_predict()
    t_pred=test_sentiment
    featureList=make_feature_list()
    featurewithsen=featurewithsen()
    training_set = nltk.classify.util.apply_features(extract_features, featurewithsen)
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set) 
    print NBClassifier.show_most_informative_features(10)
    predicted_nb=[]    
    for line in test_comments:
        predicted=NBClassifier.classify(extract_features(getfeaturevector(line)))
        predicted_nb.append(predicted)
    np.mean(np.array(predicted_nb) == np.array(t_pred))
    print(classification_report(t_pred,predicted_nb))
    print confusion_matrix(t_pred, predicted_nb)
    
    
def maxium_entropy():
    valid_comments,valid_sentiment=read_training_data()    
    result,valid_sentiment, X_train_tfidf, X_train_counts,count_vect,trans_vect=feature_extraction()
    test_comments,test_sentiment=read_predict()
    t_pred=test_sentiment
    featureList=make_feature_list()
    commentswithfeature=featurewithsen()
    training_set = nltk.classify.util.apply_features(extract_features, commentswithfeature)
    MaxEntClassifier=nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', trace=3, \
                    encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 10)
    predicted_ME=[]
    for line in test_comments:
        predicted=MaxEntClassifier.classify(extract_features(getfeaturevector(line)))
        predicted_ME.append(predicted)
    np.mean(np.array(predicted_ME) == np.array(t_pred))
    print(classification_report(t_pred,predicted_ME))
    print confusion_matrix(t_pred, predicted_ME)
    print MaxEntClassifier.show_most_informative_features(10)


def svm():
    valid_comments,valid_sentiment=read_training_data()    
    result,valid_sentiment, X_train_tfidf, X_train_counts,count_vect,trans_vect=feature_extraction()    
    test_comments,test_sentiment=read_predict()
    t_pred=test_sentiment    
    clf_svm=svm.SVC(        
        kernel='linear',
        
        
        tol=1e-3,
        class_weight='auto',
        
        )
    trained_svm=clf_svm.fit(X_train_tfidf,valid_sentiment)
    y_pred = trained_svm.predict(X_train_tfidf)
    cm = confusion_matrix(valid_sentiment, y_pred)
    cr=classification_report(valid_sentiment, y_pred)
    print cm,cr
    X_new_tfidf = trans_vect.transform(test_comments)    
    predicted = trained_svm.predict(X_new_tfidf)
    predicted_nb=[]    
    for i in xrange(len(predicted)):
        comment=predicted[i].encode('gbk','replace')
        predicted_nb.append(comment)
    np.mean(np.array(predicted_nb) == np.array(t_pred))            
    print(classification_report(t_pred, predicted_nb))
    print confusion_matrix(t_pred, predicted_nb)
    return None
"""    
def evaluate_features(feature_select):
    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polaritydata\\rt-polarity-pos.txt', 'r')
    negSentences = open('rt-polaritydata\\rt-polarity-neg.txt', 'r')
    posSentences = re.split(r'\n', posSentences.read())
    negSentences = re.split(r'\n', negSentences.read())
 
    posFeatures = []
    negFeatures = []
    #http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    #breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i)
        posWords=[make_full_dict(posWords),'pos'] #posWords = [feature_select(posWords), 'pos']
        posFeatures.append(posWords)
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i)
        negWords = [feature_select(negWords), 'neg']
        negFeatures.append(negWords)
#selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]
    classifier = NaiveBayesClassifier.train(trainFeatures)
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)
    for i, (features, label) in enumerate(testFeatures): 
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)
    
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10)
def make_full_dict(words):
    return dict([(word, True) for word in words])
print 'using all words as features'
evaluate_features(make_full_dict)
def create_word_scores():
    #splits sentences into lines
    posSentences = open('rt-polaritydata\\rt-polarity-pos.txt', 'r')
    negSentences = open('rt-polaritydata\\rt-polarity-neg.txt', 'r')
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
        word_fd[word.lower()] +=1
        cond_word_fd['pos'][word.lower()] +=1
    for word in negWords:
        word_fd[word.lower()] +=1
        cond_word_fd['neg'][word.lower()] +=1
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
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
    print 'evaluating best %d word features' % (num)
    word_scores=create_word_scores()    
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)
"""




    




