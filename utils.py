import math
import csv
import time
import numpy as np
from scipy import sparse
import pandas as pd
import os
import random
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer



def outputCSV(file_dir, output_list):
    with open(file_dir, "a", newline='') as fobj:
        # fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
        writer = csv.writer(fobj)
        for row in output_list:
            writer.writerow(row)


def compute_TU(topic_word, N):
    """
    :param topic_word: topic_word matrix
    :param N: top word count
    :return: average TU for the whole matrix
    """
    topic_size, word_size = np.shape(topic_word)
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)
    TU = 0
    cnt = [0 for i in range(word_size)]
    for topic in topic_list:
        for word in topic:
            cnt[word] += 1
    for topic in topic_list:
        TU_t = 0
        for word in topic:
            TU_t += 1/cnt[word]
        TU_t /= N
        TU += TU_t

    TU /= topic_size

    return TU


def compute_TU_list(topic_word, N):
    """
    :param topic_word: topic_word matrix
    :param N: top word count
    :return: TU for each individual topic
    """
    topic_size, word_size = np.shape(topic_word)
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)
    TU = []
    cnt = [0 for i in range(word_size)]
    for topic in topic_list:
        for word in topic:
            cnt[word] += 1
    for topic in topic_list:
        TU_t = 0
        for word in topic:
            TU_t += 1/cnt[word]
        TU_t /= N
        TU.append(TU_t)

    return TU


def print_topic_word(word_list, save_dir, topic_word, N):
    # print top N words of each topic
    topic_size, vocab_size = np.shape(topic_word)

    with open(save_dir, 'a', encoding='utf-8') as fout:
        print('-------------------- Topic words --------------------', file=fout)
        for topic_idx in range(topic_size):
            top_word_list = []
            print('['+str(topic_idx)+'] ', end='', file=fout)
            top_word_idx = np.argsort(topic_word[topic_idx, :])[-N:]
            for i in range(N):
                top_word_list.append(word_list[top_word_idx[i]])

            # print words
            for word in top_word_list:
                print(word, ' ', end='', file=fout)
            print('\n', end='', file=fout)
        print('\n', end='', file=fout)

    print('save done!')


def save_as_triple(Y, filename):
    Y_coo = sparse.coo_matrix(Y)
    result_matrix = open(filename, 'w')
    result_matrix.writelines('row_idx,col_idx,data')
    # Y_coo = Y.tocoo()
    for i in range(len(Y_coo.data)):
        result_matrix.writelines("\n" + str(Y_coo.row[i]) + "," + str(Y_coo.col[i]) + "," + str(Y_coo.data[i]))


def read_triple(filename, get_sparse=False):
    tp = pd.read_csv(open(filename))
    rows, cols, data = np.array(tp['row_idx']), np.array(tp['col_idx']), np.array(tp['data'])
    if get_sparse:
        return sparse.coo_matrix((data, (rows, cols)), shape=(max(rows)+1, max(cols)+1))
    return sparse.coo_matrix((data, (rows, cols)), shape=(max(rows)+1, max(cols)+1)).toarray()


# freeze random seeds
def freeze_seed(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


### LNTM_metrics ###
def softmax_np(x):
    x = np.array(x)
    soft_x = []
    for x_t in x:
        x_max = np.max(x_t)
        x_t = x_t - x_max
        x_exp = np.exp(x_t)
        x_exp_sum = np.sum(x_exp)
        soft_x_t = x_exp / x_exp_sum
        soft_x.append(soft_x_t)
    soft_x = np.array(soft_x)
    return soft_x

def perplexity(testset, topic_word, doc_topic):
    """calculate the perplexity of a lda-model"""
    # testset: BoW?

    print('doc size =',len(testset))
    print('vocab size =',len(testset[0]))
    print('topic num =',len(topic_word))
    
    num_topics = len(topic_word)
    
    #print('pre topic_word[0]=',topic_word[0])
    topic_word = softmax_np(topic_word)
    #print('softmax topic_word[0]=',topic_word[0])
    
    #print('pre doc_topic[0]=',doc_topic[0])
    doc_topic = softmax_np(doc_topic)
    #print('softmax doc_topic[0]=',doc_topic[0])
    #print('sum=',np.sum(doc_topic[0]))
    
    doc_word_prob = doc_topic.dot(topic_word)

    prep = 0.0
    prob_doc_sum = 0.0
    
    testset_word_num = np.sum(testset)
    for i in range(len(testset)):
        prob_doc = 0.0 # the probablity of the doc
        doc = testset[i]
        for word_id, num in enumerate(doc):
            if num == 0:
                continue
            # cal p(w) : p(w) = sumz(p(z)*p(w|z))
            prob_word = doc_word_prob[i][word_id] # the probablity of the word 
            prob_doc += math.log(prob_word)*num # p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
    
    #print(-prob_doc_sum/testset_word_num)
    prep = math.exp(-prob_doc_sum/testset_word_num) # perplexity = exp(-sum(p(d)/sum(Nd))

    return prep

def save_loss_trend(loss_list, save_path):
    f = open(save_path, "w")
    f.write(str(loss_list))
    f.close()
    
def tfidf(Dt_path):
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\S+\b", lowercase=False)
    transformer = TfidfTransformer()
    # Dt_pt = open(Dt_path)
    Dt_pt = load_pkl(Dt_path)
    X = vectorizer.fit_transform(Dt_pt)
    vocabulary = vectorizer.get_feature_names()
    tfidf = transformer.fit_transform(X)
    # Dt_pt.close()

    # y = []
    # with open(Dt_path+'_label') as fobj:
    #     for line in fobj.readlines():
    #         label = line.strip()
    #         y.append(label)

    # return tfidf.transpose(), vocabulary#, y
    return np.array(tfidf.toarray()), vocabulary#, y

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)