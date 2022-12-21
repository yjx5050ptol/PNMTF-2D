#coding=utf-8

import os
import sys
import time
import configparser as cp
import numpy as np
import heapq
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
import pickle
import utils

def get_top_words(topic_word_mat, vocabulary, k):
    
    # Acknowledgement: HenryLiu
    topic_word_id = []
    for line in topic_word_mat:
        max_num_index_list = map(line.tolist().index, heapq.nlargest(k, line.tolist()))
        topic_word_id.append(list(max_num_index_list))
    topic_words = []
    for line in topic_word_id:
        a = []
        for id in line:
            a.append(vocabulary[id])
        topic_words.append(a)
    # print(topic_words)
    return topic_words

# 数值最高的T个主题词
def topT(topic_word, T):
    topic = []
    for i in range(len(topic_word)):
        topic_now = np.argsort(-topic_word[i]).tolist()[0:T]
        topic.append(topic_now)
    return topic

def get_internal_texts(text_filename): # min_cnt2.txt
    # text_file = open(text_filename, "r")
    # texts = text_file.readlines()
    # for i, line in enumerate(texts):
    #     texts[i] = line.strip().split(",")[1].strip().split(" ")
    # text_file.close()
    with open(text_filename, 'rb') as f:
        texts = pickle.load(f)
    for i, line in enumerate(texts):
        texts[i] = line.strip().replace('\n', '').split(" ")
    texts = list(texts)
    vocab = corpora.Dictionary(texts)
    return texts, vocab


def cal_coherence(topic_words, methods, texts, vocab):  # topic_words: from get_top_words(); text, vocab: from text_filename();
    coherence = {}
    # methods = ["c_npmi"]
    metrics = ['u_mass','c_v','c_uci','c_npmi']
    new_methods = list(set(metrics).intersection(set(methods)))
    for method in new_methods:
        coherence[method] = CoherenceModel(topics=topic_words, texts=texts, dictionary=vocab, coherence=method).get_coherence()
    return coherence

# 计算连贯性, 来自祖传算法 。。。
# 以文档为窗口
def compute_coherence(topic_words, doc_word):
    topic_words = np.array(topic_words)
    topic_size, word_size = np.shape(topic_words)
    doc_size = np.shape(doc_word)[0]
    coherence = []
    for N in [5, 10, 15]:
        # find top words'index of each topic
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(
                topic_words[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)

        # compute coherence of each topic
        sum_coherence_score = 0.0
        for i in range(topic_size):
            word_array = topic_list[i]
            sum_score = 0.0
            for n in range(N):
                flag_n = doc_word[:, word_array[n]] > 0
                p_n = np.sum(flag_n) / doc_size
                for l in range(n + 1, N):
                    flag_l = doc_word[:, word_array[l]] > 0
                    p_l = np.sum(flag_l)
                    p_nl = np.sum(flag_n * flag_l)
                    if p_n * p_l * p_nl > 0:
                        p_l = p_l / doc_size
                        p_nl = p_nl / doc_size
                        sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
            sum_coherence_score += sum_score * (2 / (N * N - N))
        sum_coherence_score = sum_coherence_score / topic_size
        coherence.append(sum_coherence_score)

    # print(f"ours-npmi:{np.mean(coherence)}")

    return np.mean(coherence)

# 计算主题冗余度，以重复出现的词数量来统计
def evaluate_topic_diversity(topic_words):
    """topic_words is in the form of [[w11,w12,...],[w21,w22,...]]"""
    topic_words_np = np.array(topic_words)
    TU = 0.0
    for n in [5, 10, 15]:
        compute_topic_words = topic_words_np[:, :n].tolist()
        TU += compute_topic_diversity(compute_topic_words)

    TU /= 3
    # print(f"topic_diversity:{TU}")
    return TU

# 计算主题冗余度，以重复出现的词数量来统计
def compute_topic_diversity(topic_words):
    """topic_words is in the form of [[w11,w12,...],[w21,w22,...]]"""
    vocab = set(sum(topic_words, []))
    total = sum(topic_words, [])
    return len(vocab) / len(total)

def evaluate_coherence(topic_word_mat, k, methods, vocabulary, text_filename, doc_word):
    topic_words = get_top_words(topic_word_mat, vocabulary, k)
    texts, vocab = get_internal_texts(text_filename)
    metrics_dict = cal_coherence(topic_words, methods, texts, vocab)
    # npmi TU 都是前5、10、15个主题词的平均值
    metrics_dict['npmi'] = compute_coherence(topic_word_mat, doc_word)
    metrics_dict['tu'] = evaluate_topic_diversity(topic_words)
    return metrics_dict


def evaluate_topics(W, res_dir, metrics, top_n, vocabulary, text_file, doc_word):

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    coh = evaluate_coherence(W.T, top_n, metrics, vocabulary, text_file, doc_word)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(coh)
    out_file = os.path.join(res_dir, 'evaluate_topics_pnmtf-2D.csv')
    if not os.path.exists(out_file):
        utils.outputCSV(out_file, [metrics])
    out_list = [coh[metric] for metric in metrics]
    utils.outputCSV(out_file, [out_list])