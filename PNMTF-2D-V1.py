# 24-1,24-2
import os
import sys
import time
import math

import numpy as np
from scipy import sparse
from mpi4py import MPI

from tqdm import tqdm
import configparser as cp
from sklearn.preprocessing import normalize

import utils
import argparse
from evaluate_clustering import evaluate_cluster
from evaluate_topics import evaluate_topics

model_name = 'PNMTF-2D-V1'


def merge_chunk(pipeline, merge_chunk_path):
    Dt_merge = []
    for path in pipeline:
        Dt_pt = open(path)
        Dt_lines = Dt_pt.readlines()
        Dt_merge.extend(Dt_lines)
        Dt_pt.close()
    with open(merge_chunk_path, 'w') as f:
        for line in Dt_merge:
            f.write(line)
    
    y_merge = []
    for path in pipeline:
        y_pt = open(path+'_label')
        y_lines = y_pt.readlines()
        y_merge.extend(y_lines)
        y_pt.close()
    merge_label_path = merge_chunk_path+"_label"
    with open(merge_label_path, 'w') as f:
        for line in y_merge:
            f.write(line)
    
def concurrence(merge_chunk_path, vocabulary, save_path, num_word):   # co-occurrence matrix construction
    window_size = 5
    K = sparse.lil_matrix((num_word, num_word), dtype='float64')
    word_to_id = {} # a dict mapping word to idx
    for id, word in enumerate(vocabulary):
        word_to_id[word] = id
    for line in tqdm(open(merge_chunk_path)):
        line_list = line.strip().split()
        for ind_focus, wid_focus in enumerate(line_list):  
            ind_lo = max(0, ind_focus-window_size)  
            ind_hi = min(len(line_list), ind_focus+window_size+1)  
            for ind_c in range(ind_lo, ind_hi):
                if ind_c == ind_focus:  
                    continue
                if line_list[ind_c] == wid_focus:    
                    continue
                focus_id = word_to_id[wid_focus]   
                c_id = word_to_id[line_list[ind_c]]                 
                K[focus_id, c_id] += 1
    utils.save_as_triple(K, save_path)


def cal_blocksize(n, size, rank):
    if rank < (n % size):
        return int(math.ceil(n / size))
    else:
        return int(math.floor(n / size))


def summation(C_local, comm, rank=-1, counts=None, pattern='Allreduce'):
    """
    collective communication
    input a numpy array;
    pattern='Allreduce' or 'Reduce_scatter';
    rank and counts should be passed if 'Reduce_scatter' is passed.
    """
    # C_local = A_col.dot(B_row)
    if pattern == 'Allreduce':
        C = np.empty(C_local.shape, dtype='float64')
        comm.Allreduce([C_local, MPI.DOUBLE], [C, MPI.DOUBLE], op=MPI.SUM)
        return C
    elif pattern == 'Reduce_scatter':
        buffersize_p = counts[rank]
        colcount = C_local.shape[1]
        rowcount_p = buffersize_p // colcount
        C_row = np.empty((rowcount_p, colcount), dtype='float64')
        comm.Reduce_scatter([C_local, MPI.DOUBLE], [C_row, MPI.DOUBLE], recvcounts=counts, op=MPI.SUM)
        return C_row
    else:
        print('Unknown pattern!')
        return None


def scatter_sparse(mat, comm, rank, counts_p, displ_p, dim, mode='row'):    
    ''' mode: 'row' or 'col' '''
    if rank == 0:
        # csr meta-data
        if mode == 'row':
            mat = sparse.csr_matrix(mat)
        elif mode == 'col':
            mat = sparse.csc_matrix(mat)
        else:
            print('Invalid mode for scatter_sparse_matrix')
            return None
        indptr = mat.indptr  # row/col offset
        indices = mat.indices  # col/row index
        data = mat.data #?
    else:
        indptr = np.empty(sum(counts_p)+1, dtype='i')  # +1: rf. scipy doc of indptr
        indices = None
        data = None

    # Broadcast indptr
    comm.Bcast(indptr, root=0)

    # Calculate counts and displ for indices and data
    displ_data = [indptr[start] for start in displ_p]   
    displ_data.append(indptr[-1])   
    counts_data = [displ_data[j] - displ_data[j-1] for j in range(1, len(displ_data))]  
    displ_data = displ_data[0:-1]  # remove the last ele of indptr

    # Scatterv for indices and data
    indices_p = np.empty(counts_data[rank], dtype='i')
    data_p = np.empty(counts_data[rank], dtype='float64')
    comm.Scatterv([indices, counts_data, displ_data, MPI.INT], indices_p, root=0)   
    comm.Scatterv([data, counts_data, displ_data, MPI.DOUBLE], data_p, root=0)      

    # construct mat_row
    indptr_p = indptr[displ_p[rank]: displ_p[rank] + counts_p[rank] + 1]  
    offset = indptr_p[0]
    indptr_p = indptr_p - offset    

    if mode == 'row':
        mat_p = sparse.csr_matrix((data_p, indices_p, indptr_p), shape=(counts_p[rank], dim))   # csr format
    elif mode == 'col':
        mat_p = sparse.csc_matrix((data_p, indices_p, indptr_p), shape=(dim, counts_p[rank]))   # csd format

    return mat_p

def load_vocabulary_list(path):
    f = open(path, "r")
    content = f.read()
    f.close()
    return eval(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_name', type=str, default='classic4', help='data set')
    parser.add_argument('--work_directory', type=str, default=r'./', help="word_directory")
    parser.add_argument('--exp_ini', type=str, required=True, help='')
    parser.add_argument('--pr', type=int, required=True, help="Number of row processors in the 2D processor grid")
    parser.add_argument('--pc', type=int, required=True, help="Number of columnn processors in the 2D processor grid")
    args = parser.parse_args()
        
    exp_ini = args.exp_ini #'CLASSIC4_FNMTF'
    work_directory = args.work_directory
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank() 
    comm_size = comm.Get_size() 

    # Cartesian topology start
    pr = args.pr  # Number of row processors in the 2D processor grid  
    pc = args.pc  # Number of columnn processors in the 2D processor grid  
    reorder = 0 
    dimSizes = [None] * 2   
    periods = [1, 1]    # is loop 
    nd = 2
    dimSizes[0] = pr
    dimSizes[1] = pc

    if dimSizes[0] * dimSizes[1] != comm_size:  
        if comm_rank == 0:
            print("Processor grid dimensions do not multiply to MPI_SIZE")
        comm.Barrier()
        comm.Abort(1)   # [errorcode]

    m_gridComm = comm.Create_cart(dimSizes, periods, reorder)

    remain_dims = [False, True]
    comm_cart_rows = m_gridComm.Sub(remain_dims)
    remain_dims = [True, False]
    comm_cart_cols = m_gridComm.Sub(remain_dims)

    m_row_size = comm_cart_rows.Get_size()
    m_row_rank = comm_cart_rows.Get_rank()
    m_col_size = comm_cart_cols.Get_size()
    m_col_rank = comm_cart_cols.Get_rank()

    if comm_rank == 0:  
        print("m_row_size:%d, m_col_size:%d" % (m_row_size, m_col_size))

    # Cartesian topology end



    T = None 
    max_step = None
    num_word_topic = None
    num_doc_topic = None
    eps = None
    loss_trend = 'False'
    seed_num = 0    

    if comm_rank == 0:

        # work_directory = os.path.dirname(os.path.abspath(__file__)) 
        # exp_ini = sys.argv[1]   
        # experiment.ini
        exp_config = cp.ConfigParser()
        exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding='utf-8')

        data_name = exp_config.get(exp_ini, 'data_name')
        model_name_input = exp_config.get(exp_ini, 'model_name')
        max_step = int(exp_config.get(exp_ini, 'max_step'))
        num_word_topic = int(exp_config.get(exp_ini, 'num_word_topic'))
        num_doc_topic = int(exp_config.get(exp_ini, 'num_doc_topic'))
        top_n = int(exp_config.get(exp_ini, 'top_n'))  # num of topic words
        eps = float(exp_config.get(exp_ini, 'eps'))
        coh_metrics = exp_config.get(exp_ini, 'coh_metrics').split(',')
        if exp_config.has_option(exp_ini, 'loss_trend'):
            loss_trend = exp_config.get(exp_ini, 'loss_trend')  
        if exp_config.has_option(exp_ini, 'seed_num'):  
            seed_num = int(exp_config.get(exp_ini, 'seed_num'))

        if model_name != model_name_input:
            print('Inconsistent model name')
            sys.exit(1)

        cur_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        # dataset.ini
        data_config = cp.ConfigParser()
        data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding='utf-8')
        DATA_DIR = data_config.get(data_name, "DATA_DIR")   
        RESULT_DIR = data_config.get(data_name, "RESULT_DIR")     
        RESULT_DIR = os.path.join(RESULT_DIR, data_name, model_name, exp_ini)
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        file_true = os.path.join(DATA_DIR, 'label.pkl')   
        vocab_file = os.path.join(DATA_DIR, 'vocab.pkl')
        text_file = os.path.join(DATA_DIR, 'text.pkl')
        tfidf_file = os.path.join(DATA_DIR, 'tfidf.pkl')
        ################## existence check ###########################
        if not os.path.exists(tfidf_file):
            tfidf, vocabulary = utils.tfidf(text_file)
            utils.save_pkl(tfidf, tfidf_file)
            utils.save_pkl(vocabulary, vocab_file)
        tfidf = np.array(utils.load_pkl(tfidf_file))
        vocabulary = utils.load_pkl(vocab_file)
        
        filename_prefix = data_config.get(data_name, 'filename_prefix')
        label = data_config.get(data_name, 'label').split(',')  

        print(exp_ini)
        print("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, top_n {top_n}, "
              "eps {eps}, p {p}, loss_trend {loss_trend}, seed_num {seed_num}.".format(
               max_step=max_step, num_word_topic=num_word_topic, num_doc_topic=num_doc_topic, top_n=top_n,
               eps=eps, p=comm_size, loss_trend=loss_trend, seed_num=seed_num) + "\n")   
        pipeline = os.path.join(DATA_DIR, 'chunks', filename_prefix)  

        log_file = os.path.join(RESULT_DIR, "log.txt")
        topic_words_local_file = os.path.join(RESULT_DIR, "topic_words_local.txt") # W
        topic_words_global_file = os.path.join(RESULT_DIR, "topic_words_global.txt")   #WS
        loss_trend_file = os.path.join(RESULT_DIR, "loss_trend.txt") # loss FILE
        loss_list = []  

        log_files = [log_file, topic_words_local_file, topic_words_global_file, loss_trend_file]

        for f in log_files: 
            with open(f, "a") as fobj:
                fobj.write(exp_ini + ' ')
                fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " PNMTF_2D Start" + "\n")
                fobj.write(data_name + ": " + str(pipeline) + "\n")
                fobj.write("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, "
                           "top_n {top_n}, "
                           "eps {eps}, p {p}, loss_trend {loss_trend}, seed_num {seed_num}.".format(max_step=max_step, num_word_topic=num_word_topic,
                                               num_doc_topic=num_doc_topic, top_n=top_n,
                                               eps=eps, p=comm_size, loss_trend=loss_trend, seed_num=seed_num) + "\n")


        # merge_chunk_path = os.path.join(work_directory, "Dataset", data_name, filename_prefix+"_all")
        # if not os.path.exists(merge_chunk_path):
        #     merge_chunk(pipeline, merge_chunk_path)

        # load_D_path = os.path.join(work_directory, "Dataset", data_name, "D.npz")
        # load_K_path = os.path.join(work_directory, "Dataset", data_name, "K.npz")
        # read_vocabulary_path = os.path.join(work_directory, "Dataset", data_name, "vocabulary.txt")
    ## comm_rank 0 end
    max_step = comm.bcast(max_step, root=0)
    num_word_topic = comm.bcast(num_word_topic, root=0)
    num_doc_topic = comm.bcast(num_doc_topic, root=0)
    eps = comm.bcast(eps, root=0)
    loss_trend = comm.bcast(loss_trend, root=0)
    seed_num = comm.bcast(seed_num, root=0)
    utils.freeze_seed(seed_num + comm_size+comm_rank)

    # INITILIZATION BY ZERO 
    if comm_rank == 0:

        # if os.path.exists(load_D_path) and os.path.exists(read_vocabulary_path):
        #     print("load D.npz and vocabulary.txt")
        #     D = sparse.load_npz(load_D_path)
            # vocabulary = load_vocabulary_list(read_vocabulary_path)
        # else:
            # D, vocabulary, y = tfidf(merge_chunk_path)
            # 将tfidf,label,text,vocab（pkl形式）存在DATA_DIR(可自行更改)
            # text_reader.save_PNMTF_Baseline_data(DATA_DIR)
            # D = np.array([[0 ,1 ,0 ,0 ,0  ,2 ,0 ,0 ,3],
            #                 [0 ,0 ,4 ,0 ,5  ,0 ,6 ,7 ,0],
            #                 [0 ,8 ,0 ,9 ,0  ,0 ,10,11,12],
            #                 [13,0 ,14,0 ,0  ,15,0 ,0 ,0],
            #                 [0 ,0 ,16,17,0  ,18,19,0 ,0],
            #                 [0 ,20,0 ,21,0  ,0 ,22,23,24],
            #                 [25,26,0 ,0 ,0  ,27,0 ,0 ,0],
            #                 [0 ,0 ,28,29,30 ,0 ,0 ,31,0],
            #                 [32,33,0 ,0 ,0  ,34,0 ,35,36],
            #                 [0 ,37,38,39,0  ,0 ,0 ,0 ,0],
            #                 [0 ,0 ,0 ,0 ,0  ,40,41,42,43]], dtype='float64')    # 
        # 
        # text_reader = reader.TextReader(args.data_name, args.data_dir)
        # D, _, y = text_reader.get_matrix(data_type='all', mode='tfidf')
        # D = D.T
        # vocabulary = text_reader.vocab
        
        # assert sorted(vocabulary) == vocabulary
        D = utils.load_pkl(tfidf_file).T
        vocabulary = utils.load_pkl(vocab_file)
        num_word, num_doc = D.shape
        print("num_word = ", num_word, "num_document = ", num_doc)

        #Initialize S
        S = np.random.rand(num_word_topic, num_doc_topic).astype('float64')

        V = np.empty((num_doc, num_doc_topic), dtype='float64')
        W = np.empty((num_word, num_word_topic), dtype='float64')

    else:  # comm_rank != 0
        num_word = None
        num_doc = None
        D_r = None
        D = None
        W = None
        V = None
        S = np.empty((num_word_topic, num_doc_topic), dtype='float64')

        

    '''Parallelization handling'''
    # constants
    num_word = comm.bcast(num_word, root=0)
    num_doc = comm.bcast(num_doc, root=0)

    comm.Bcast(S, root=0) # broadcast S

    

    
    rcount = cal_blocksize(num_doc, m_row_size, m_row_rank)
    ccount = cal_blocksize(num_word, m_col_size, m_col_rank)

    
    rcount_list = np.empty(m_row_size, dtype='i')   
    ccount_list = np.empty(m_col_size, dtype='i')   

    comm_cart_rows.Allgather(np.array([rcount], dtype='i'), rcount_list)
    comm_cart_cols.Allgather(np.array([ccount], dtype='i'), ccount_list)

    displ_r = np.insert(np.cumsum(rcount_list), 0, 0)[0:-1]
    displ_c = np.insert(np.cumsum(ccount_list), 0, 0)[0:-1]

    # scatter in cols
    if m_row_rank == 0:
        # scatterv
        D_r = scatter_sparse(D, comm_cart_cols, m_col_rank, ccount_list, displ_c, num_doc, mode='row')

    # scatter in rows
    Dij = scatter_sparse(D_r, comm_cart_rows, m_row_rank, rcount_list, displ_r, ccount, mode='col')


    # initialize W and V

    W_ccount = cal_blocksize(ccount, m_row_size, m_row_rank)
    Wij = np.random.rand(W_ccount, num_word_topic).astype('float64')
    # 
    V_rcount = cal_blocksize(rcount, m_col_size, m_col_rank)
    Vji = np.random.rand(V_rcount, num_doc_topic).astype('float64')

    # 
    # 1
    rcounts_Vji = np.empty(m_col_size, dtype='i')
    comm_cart_cols.Allgather(np.array([V_rcount], dtype='i'), rcounts_Vji)
    counts_VSTj = rcounts_Vji * num_word_topic
    displ_VSTj = np.insert(np.cumsum(counts_VSTj), 0, 0)[0:-1]
    # 2
    ccounts_Wij = np.empty(m_row_size, dtype='i')
    comm_cart_rows.Allgather(np.array([W_ccount], dtype='i'), ccounts_Wij)
    counts_Wij = counts_DVSTij = ccounts_Wij * num_word_topic
    # 3
    counts_WSi = ccounts_Wij * num_doc_topic
    displ_WSi = np.insert(np.cumsum(counts_WSi), 0, 0)[0:-1]
    # 4
    counts_Vji = counts_DTWSji = rcounts_Vji * num_doc_topic
    # 5
    displ_Vji = np.insert(np.cumsum(counts_Vji), 0, 0)[0:-1]
    # 6
    counts_DVij = counts_WSi # = ccounts_Wij * num_doc_topic

    if comm_rank == 0:
        time_start = time.time()  # timing  

    '''Iterative Updates'''
    # Update W, S, V
    for times in tqdm(range(max_step)):
    # for i in range(1):  
        '''update W'''
        # pij computes localVST
        localVST = Vji.dot(S.T)
        # pij computes Aij = localSVT dot (localSVT)T
        Aij = localVST.T.dot(localVST)
        
        # compute SVTSTV using all-reduce across all procs
        SVTSTV = summation(Aij, comm, pattern='Allreduce')
        # SVTSTV = comm.allreduce([Aij, MPI.DOUBLE], op=MPI.SUM)

        VSTj = np.empty((rcount, num_word_topic), dtype='float64')
        comm_cart_cols.Allgatherv(localVST, [VSTj, counts_VSTj, displ_VSTj, MPI.DOUBLE])

        # pij computes Bij = Dij (VST)j
        Bij = Dij.dot(VSTj)

        DVSTij = summation(Bij, comm_cart_rows, m_row_rank, counts_DVSTij, pattern='Reduce_scatter')

        Wij = Wij * ((eps + DVSTij) / (eps + Wij.dot(SVTSTV)))

        '''update V'''
        # pij computes local localWS
        localWS = Wij.dot(S)
        # pij computes Xij = localWS.T dot localWS
        Xij = localWS.T.dot(localWS)
        # compute STWTWS using all-reduce across all procs
        STWTWS = summation(Xij, comm, pattern='Allreduce')
        WSi = np.empty((ccount, num_doc_topic), dtype='float64')
        comm_cart_rows.Allgatherv(localWS, [WSi, counts_WSi, displ_WSi, MPI.DOUBLE])

        # pij computes Yij = Dij.T (WS)i
        Yij = Dij.T.dot(WSi)
        DTWSji = summation(Yij, comm_cart_cols, m_col_rank, counts_DTWSji, pattern='Reduce_scatter')

        Vji = Vji * ((eps + DTWSji) / (eps + Vji.dot(STWTWS)))

        '''update S'''
        # pij collects Vj using all-gather
        Vj = np.empty((rcount, num_doc_topic))
        # displ_Vji = np.insert(np.cumsum(counts_Vji), 0, 0)[0:-1]    
        comm_cart_cols.Allgatherv(Vji, [Vj, counts_Vji, displ_Vji, MPI.DOUBLE])

        # pij computes Eij = Dij dot Vj
        Eij = Dij.dot(Vj)
        DVij = summation(Eij, comm_cart_rows, m_row_rank, counts_DVij, pattern='Reduce_scatter')

        # Fij = WijT dot Eij
        Fij = Wij.T.dot(DVij)
        # compute WTDV using all-reduce Yij across all procs
        WTDV = summation(Fij, comm, pattern='Allreduce')

        # pij computes Gij = WijT dot Wij
        Gij = Wij.T.dot(Wij)
        # computes WTW using all-reduce
        WTW = summation(Gij, comm, pattern='Allreduce')
        # pij computes Hij = VjiT dot Vji
        Hij = Vji.T.dot(Vji)
        # computes VTV using all-reduce
        VTV = summation(Hij, comm, pattern='Allreduce')
        # computes WTWSVTV = WTW dot S dot VTV
        WTWSVTV = WTW.dot(S).dot(VTV)
        S = S * ((eps + WTDV) / (eps + WTWSVTV))

        '''compute loss'''
        if loss_trend == "True":
            # pij - ((WS)i)j = (Wi)j dot S
            WSij = Wij.dot(S)
            # pij - Allgather( ((WS)i)j )
            comm_cart_rows.Allgatherv(WSij, [WSi, counts_WSi, displ_WSi, MPI.DOUBLE])
            # pij - (WSVT)ij = (WS)i dot Vj.T
            WSVTij = WSi.dot(Vj.T)
            # pij -  Aij = Dij - (WSVT)ij， aij = norm(Aij)^2
            aij = np.linalg.norm(Dij - WSVTij)**2   
            # zero reduce
            a = comm.reduce(aij, op=MPI.SUM, root=0)

            break_flag = False
            if comm_rank == 0:
                loss = a
                loss_list.append(loss)
                f = open(loss_trend_file, "a")
                f.write(str(loss) + "\n")
                f.close()
                if len(loss_list) > 2 and abs(loss_list[-1] - loss_list[-2]) < 1e-4:    
                    break_flag = True
            break_flag = comm.bcast(break_flag, root=0)
            if break_flag:
                break   # 所有进程一起break
        # compute loss ---End，并行计算loss


    if comm_rank == 0:
        total_iter_time = time.time() - time_start  # timing    

    '''Result Gathering'''
    ccounts_gather_Wij = np.empty(comm_size, dtype='i')
    comm.Allgather(np.array([W_ccount], dtype='i'), ccounts_gather_Wij)
    counts_gather_Wij = ccounts_gather_Wij * num_word_topic
    displ_gather_Wij = np.insert(np.cumsum(counts_gather_Wij), 0, 0)[0:-1]
    comm.Gatherv(Wij, [W, counts_gather_Wij, displ_gather_Wij, MPI.DOUBLE], root=0)
    
    # 2-dimensional gather
    # gather to the first thread of col
    Vj = np.empty((rcount, num_doc_topic))
    displ_Vji = np.insert(np.cumsum(counts_Vji), 0, 0)[0:-1]
    comm_cart_cols.Gatherv(Vji, [Vj, counts_Vji, displ_Vji, MPI.DOUBLE], root=0)
    # additional gather on row
    if m_col_rank == 0:
        rcount_Vj = np.sum(counts_Vji)
        rcounts_Vj = np.empty(m_row_size, dtype='i')
        comm_cart_rows.Allgather(np.array([rcount_Vj], dtype='i'), rcounts_Vj)
        displ_Vj = np.insert(np.cumsum(rcounts_Vj), 0, 0)[0:-1]
        comm_cart_rows.Gatherv(Vj, [V, rcounts_Vj, displ_Vj, MPI.DOUBLE], root=0)


    '''Evaluation'''
    if comm_rank == 0:

        # write results
        utils.save_as_triple(W, os.path.join(RESULT_DIR, 'W'+'.csv'))
        utils.save_as_triple(S, os.path.join(RESULT_DIR, 'S'+'.csv'))
        utils.save_as_triple(V, os.path.join(RESULT_DIR, 'V'+'.csv'))

        utils.print_topic_word(vocabulary, topic_words_local_file, W.T, top_n)
        utils.print_topic_word(vocabulary, topic_words_global_file, (W.dot(S)).T, top_n)
        
        # # evaluate cluster
        # evaluate_cluster(file_true=file_true, V=V, res_dir=RESULT_DIR)
        # evaluate cluster
        evaluate_cluster(file_true=file_true, V=V, res_dir=RESULT_DIR)
        
        # evaluate topic
        evaluate_topics(W=W, res_dir=RESULT_DIR, metrics=coh_metrics,
                        top_n=top_n, vocabulary=vocabulary, text_file=text_file, doc_word=tfidf)

    '''Write Log'''
    if comm_rank == 0:
        with open(log_file, "a") as fobj:
            fobj.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " End" + "\n")
            fobj.write("Total iterative updates time:" + str(total_iter_time) + " s \n")
            fobj.write("Average Time (Seconds) Per Iteration:" + str(total_iter_time/max_step) + "\n")
    