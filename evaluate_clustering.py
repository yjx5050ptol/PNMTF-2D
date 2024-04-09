import os
import sys
import configparser as cp
from turtle import pos, shape
import numpy as np
from scipy import sparse

# from sklearn.metrics.cluster import rand_score  
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
# from sklearn.metrics.cluster import pair_confusion_matrix
# from sklearn.cluster import KMeans
# from coclust.clustering import SphericalKmeans

import utils
import pickle


# work_directory = os.path.dirname(os.path.abspath(__file__))
# exp_ini = sys.argv[1]

# # experiment.ini
# exp_config = cp.ConfigParser()
# exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding='utf-8')
# data_name = exp_config.get(exp_ini, 'data_name')
# model_name = exp_config.get(exp_ini, 'model_name')
# # dataset.ini
# data_config = cp.ConfigParser()
# data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding='utf-8')
# filename_prefix = data_config.get(data_name, 'filename_prefix')
# chunknum = int(data_config.get(data_name, 'chunk_num'))
# n_clusters = len(data_config.get(data_name, 'label').split(','))

# data_dir = os.path.join(work_directory, "Dataset", data_name)
# res_dir = os.path.join(work_directory, "res", data_name, model_name, exp_ini)

# label_file_list = [os.path.join(data_dir, filename_prefix+'_'+str(i+1)+'_label') for i in range(chunknum)]
# # if model_name == 'PNMTF-LTM' or model_name == 'NMTF-LTM':
# #     res_file_list = [os.path.join(res_dir, 'V_'+str(i)+'.csv') for i in range(chunknum)]
# # else:
# #     res_file_list = [os.path.join(res_dir, 'DT_' + str(i) + '.csv') for i in range(chunknum)]
# # res_file_list = [os.path.join(res_dir, 'V.csv') for i in range(chunknum)]
# res_file = os.path.join(res_dir, 'V.csv')

def get_true_label(filename):
    # res = []
    # label_id_mapping = {}
    # label_id_count = 0
    
    # for filename in filename_list:
    #     with open(filename) as fobj:
    #         for line in fobj.readlines():
    #             label = line.strip()
    #             if label in label_id_mapping:
    #                 res.append(label_id_mapping[label])
    #             else:
    #                 res.append(label_id_count)
    #                 label_id_mapping[label] = label_id_count
    #                 label_id_count += 1
    # return np.array(res)
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    return res


def accuracy(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


# def get_f_measure(labels_true, labels_pred, beta=5.):
#     (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
#     # convert to Python integer types, to avoid overflow or underflow
#     # tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
#     # Special cases: empty data or full agreement
#     # if fn == 0 and fp == 0:
#     #     return 1.0
#     p, r = tp / (tp + fp), tp / (tp + fn)
#     f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
#     return f_beta


def cal_indexes(labels_true, labels_pred):
    """
    ARI, NMI
    """
    # ri = rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred) #average_method='arithmetic'
    ari = adjusted_rand_score(labels_true, labels_pred)
    # f_beta = get_f_measure(labels_true, labels_pred)
    # purity = accuracy(labels_true, labels_pred)
    # return nmi, ri, f_beta, purity
    return nmi, ari

def cal_cluster_label(mat, post='max'):
    """
    each row of mat is a cluster indicator (probability distribution) vector
    """
    # res = np.array([])
    if mat.ndim <= 1:
        return mat
    if post == 'max':  # max
        res = np.argmax(mat, axis=1)
    else:
        print("[Warning] Unknown post processing, 'max' applied")
        res = np.argmax(mat, axis=1)
    return res


# if __name__ == '__main__':
def evaluate_cluster(file_true, V, res_dir, post='max'):
    # input_matrices = []
    # for filename in res_file_list:
    # input_matrices.append(utils.read_triple(res_file))
    # doc-topic
    # X = np.concatenate(input_matrices, axis=0)
    # X = utils.read_triple(res_file)
    V = sparse.csr_matrix(V)
    print(V.shape)

    out_file = os.path.join(res_dir, 'evaluate_clustering-PNMTF-2D.csv')
    if not os.path.exists(out_file):
        utils.outputCSV(out_file, [['NMI', 'ARI']])

    labels_true = get_true_label(file_true)

    # labels_pred follow 2022 TKDE PNMTF
    labels_pred = cal_cluster_label(V, post=post)
    labels_pred = np.asarray(labels_pred).flatten()
    nmi, ari = cal_indexes(labels_true, labels_pred)

    # print('NMI:',nmi, 'RI:', ri, 'F_beta', f_beta, 'Purity', purity)
    print('NMI:',nmi, 'ARI:', ari)
    
    # utils.outputCSV(out_file, [[nmi, ri, f_beta, purity]])
    utils.outputCSV(out_file, [[nmi, ari]])
