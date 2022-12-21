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
    
def concurrence(merge_chunk_path, vocabulary, save_path, num_word):   # 构造词共现矩阵并以文件形式保存
    window_size = 5
    K = sparse.lil_matrix((num_word, num_word), dtype='float64')
    word_to_id = {} # 词到词id的映射字典
    for id, word in enumerate(vocabulary):
        word_to_id[word] = id
    for line in tqdm(open(merge_chunk_path)):
        line_list = line.strip().split()
        for ind_focus, wid_focus in enumerate(line_list):  # ind_focus:中心窗口下标，wid_focus中心窗口词
            ind_lo = max(0, ind_focus-window_size)  # 左窗口下标
            ind_hi = min(len(line_list), ind_focus+window_size+1)  # 右窗口下标
            for ind_c in range(ind_lo, ind_hi):
                if ind_c == ind_focus:  # 跳过遍历指示下标ind_c和中心下标一样的情况
                    continue
                if line_list[ind_c] == wid_focus:    # 跳过遍历指示词和中心词一样的情况
                    continue
                focus_id = word_to_id[wid_focus]    # 中心词的词id
                c_id = word_to_id[line_list[ind_c]]    # 遍历指示词的词id                
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


def scatter_sparse(mat, comm, rank, counts_p, displ_p, dim, mode='row'):    # 稀疏矩阵的scatter
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
    displ_data = [indptr[start] for start in displ_p]   # 获取每隔m/p行的对应行首元素相对于全元素的偏移
    displ_data.append(indptr[-1])   # 加入最后一个元素（元素数量）
    counts_data = [displ_data[j] - displ_data[j-1] for j in range(1, len(displ_data))]  # 获取每m/p行的元素数量
    displ_data = displ_data[0:-1]  # remove the last ele of indptr

    # Scatterv for indices and data
    indices_p = np.empty(counts_data[rank], dtype='i')
    data_p = np.empty(counts_data[rank], dtype='float64')
    comm.Scatterv([indices, counts_data, displ_data, MPI.INT], indices_p, root=0)   # scatter各rank的indices到indices_p中
    comm.Scatterv([data, counts_data, displ_data, MPI.DOUBLE], data_p, root=0)      # scatter各rank的data到data_p中

    # construct mat_row
    indptr_p = indptr[displ_p[rank]: displ_p[rank] + counts_p[rank] + 1]  # +1: rf. scipy doc of indptr # 根据总indptr构造各个rank的indptr_p
    offset = indptr_p[0]
    indptr_p = indptr_p - offset    # 各rank取总indptr中对应段，减去第一个元素的值，使第一个元素偏移为0。

    if mode == 'row':
        mat_p = sparse.csr_matrix((data_p, indices_p, indptr_p), shape=(counts_p[rank], dim))   # 根据接收的data_p, indices_p, indptr_p构造自己rank的csr稀疏矩阵
    elif mode == 'col':
        mat_p = sparse.csc_matrix((data_p, indices_p, indptr_p), shape=(dim, counts_p[rank]))   # 根据接收的data_p, indices_p, indptr_p构造自己rank的csc稀疏矩阵

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
    comm_rank = comm.Get_rank() # 编号
    comm_size = comm.Get_size() # 大小

    # 笛卡尔拓扑 start
    pr = args.pr  # Number of row processors in the 2D processor grid  # pr是列通信域进程数量
    pc = args.pc  # Number of columnn processors in the 2D processor grid  # pc是行通信域进程数量
    reorder = 0 # 是否重排
    dimSizes = [None] * 2   # 分配进程在不同维度上的数目
    periods = [1, 1]    # 子进程是否循环
    nd = 2
    dimSizes[0] = pr
    dimSizes[1] = pc

    if dimSizes[0] * dimSizes[1] != comm_size:  # 两维度进程数相乘等于总进程数
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

    if comm_rank == 0:  # 测试行、列通信子包含进程数量（行通信子2个进程，列通信子3个进程）
        print("m_row_size:%d, m_col_size:%d" % (m_row_size, m_col_size))

    # 笛卡尔拓扑 end



    T = None 
    max_step = None
    num_word_topic = None
    num_doc_topic = None
    eps = None
    loss_trend = 'False'
    seed_num = 0    # 初始化为0

    if comm_rank == 0:

        # work_directory = os.path.dirname(os.path.abspath(__file__)) # 获取当前目录路径
        # exp_ini = sys.argv[1]   # 获取命令行实验名参数

        # experiment.ini
        exp_config = cp.ConfigParser()
        exp_config.read(os.path.join(work_directory, 'experiment.ini'), encoding='utf-8')

        data_name = exp_config.get(exp_ini, 'data_name')
        model_name_input = exp_config.get(exp_ini, 'model_name')
        max_step = int(exp_config.get(exp_ini, 'max_step'))
        num_word_topic = int(exp_config.get(exp_ini, 'num_word_topic'))
        num_doc_topic = int(exp_config.get(exp_ini, 'num_doc_topic'))
        top_n = int(exp_config.get(exp_ini, 'top_n'))  # 主题词数目
        eps = float(exp_config.get(exp_ini, 'eps'))
        coh_metrics = exp_config.get(exp_ini, 'coh_metrics').split(',')
        if exp_config.has_option(exp_ini, 'loss_trend'):
            loss_trend = exp_config.get(exp_ini, 'loss_trend')  # 字符串类型'False'或者'True'
        if exp_config.has_option(exp_ini, 'seed_num'):  # 若exp_ini中有就读取，否则就为默认值0
            seed_num = int(exp_config.get(exp_ini, 'seed_num'))

        if model_name != model_name_input:
            print('Inconsistent model name')
            sys.exit(1)

        cur_date = time.strftime('%Y%m%d', time.localtime(time.time()))
        # dataset.ini
        data_config = cp.ConfigParser()
        data_config.read(os.path.join(work_directory, 'dataset.ini'), encoding='utf-8')
        DATA_DIR = data_config.get(data_name, "DATA_DIR")   # dataset 存放位置
        RESULT_DIR = data_config.get(data_name, "RESULT_DIR")     # 结果存放位置  /res
        RESULT_DIR = os.path.join(RESULT_DIR, data_name, model_name, exp_ini)
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        file_true = os.path.join(DATA_DIR, 'label.pkl')    # 读label
        vocab_file = os.path.join(DATA_DIR, 'vocab.pkl')
        text_file = os.path.join(DATA_DIR, 'text.pkl')
        tfidf_file = os.path.join(DATA_DIR, 'tfidf.pkl')
        ################## 如果检测到tfidf、vocab文件的存在就不再执行 ###########################
        if not os.path.exists(tfidf_file):
            tfidf, vocabulary = utils.tfidf(text_file)
            utils.save_pkl(tfidf, tfidf_file)
            utils.save_pkl(vocabulary, vocab_file)
        tfidf = np.array(utils.load_pkl(tfidf_file))
        vocabulary = utils.load_pkl(vocab_file)
        
        filename_prefix = data_config.get(data_name, 'filename_prefix')
        label = data_config.get(data_name, 'label').split(',')  # 所有数据类别名称

        print(exp_ini)
        print("max_step {max_step}, num_word_topic {num_word_topic}, num_doc_topic {num_doc_topic}, top_n {top_n}, "
              "eps {eps}, p {p}, loss_trend {loss_trend}, seed_num {seed_num}.".format(
               max_step=max_step, num_word_topic=num_word_topic, num_doc_topic=num_doc_topic, top_n=top_n,
               eps=eps, p=comm_size, loss_trend=loss_trend, seed_num=seed_num) + "\n")   # 可以去掉lambda_tm和lambda_c和lambda_kg
        pipeline = os.path.join(DATA_DIR, 'chunks', filename_prefix)  # 给数据文件加上目录

        log_file = os.path.join(RESULT_DIR, "log.txt")
        topic_words_local_file = os.path.join(RESULT_DIR, "topic_words_local.txt") # W
        topic_words_global_file = os.path.join(RESULT_DIR, "topic_words_global.txt")   #WS
        loss_trend_file = os.path.join(RESULT_DIR, "loss_trend.txt") # loss记录文件
        loss_list = []  # 每次迭代loss记录列表

        log_files = [log_file, topic_words_local_file, topic_words_global_file, loss_trend_file]

        for f in log_files: # 所有log中写入实验名、时间、类别、参数
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

    # 由rank0初始化（其实可以每个进程各自初始化）!
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
            #                 [0 ,0 ,0 ,0 ,0  ,40,41,42,43]], dtype='float64')    # 测试时dtype需要是float64否则mpi通信不正常数据不对齐
        # 读取数据
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

    comm.Bcast(S, root=0) # 每个进程保存S

    # 二维分配D，D是sparse.csc_matrix
    # 先由0号进程在它的col通信域以csr形式scatter
    # 然后各row_rank=0进程在col通信域以csc形式scatter

    # 计算数量
    rcount = cal_blocksize(num_doc, m_row_size, m_row_rank)
    ccount = cal_blocksize(num_word, m_col_size, m_col_rank)

    # 计算偏移
    rcount_list = np.empty(m_row_size, dtype='i')   # 创造存放每个进程的rcounts长度值的空间
    ccount_list = np.empty(m_col_size, dtype='i')   # 创造存放每个进程的ccounts长度值的空间

    comm_cart_rows.Allgather(np.array([rcount], dtype='i'), rcount_list)
    comm_cart_cols.Allgather(np.array([ccount], dtype='i'), ccount_list)

    displ_r = np.insert(np.cumsum(rcount_list), 0, 0)[0:-1]
    displ_c = np.insert(np.cumsum(ccount_list), 0, 0)[0:-1]

    # 先由0号进程在它的col通信域以csr形式scatter
    if m_row_rank == 0:
        # scatterv
        D_r = scatter_sparse(D, comm_cart_cols, m_col_rank, ccount_list, displ_c, num_doc, mode='row')

    # 然后各row_rank=0进程在row通信域以csc形式scatter
    Dij = scatter_sparse(D_r, comm_cart_rows, m_row_rank, rcount_list, displ_r, ccount, mode='col')


    # 创建W和V的空间并初始化。
    # 计算每个进程占用的W的W_ccount
    W_ccount = cal_blocksize(ccount, m_row_size, m_row_rank)
    Wij = np.random.rand(W_ccount, num_word_topic).astype('float64')
    # 计算每个进程占用的W的V_rcount
    V_rcount = cal_blocksize(rcount, m_col_size, m_col_rank)
    Vji = np.random.rand(V_rcount, num_doc_topic).astype('float64')

    # count和displ计算，避免在迭代中反复计算
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
        time_start = time.time()  # timing  # 测单个迭代时间，用总迭代时间除迭代次数

    '''Iterative Updates'''
    # Update W, S, V
    for times in tqdm(range(max_step)):
    # for i in range(1):  # test一次迭代
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
        # displ_Vji = np.insert(np.cumsum(counts_Vji), 0, 0)[0:-1]    # 移到开始迭代前--- 5
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
            # pij 计算 ((WS)i)j = (Wi)j dot S
            WSij = Wij.dot(S)
            # pij 获得(WS)i通过在行进程中Allgather( ((WS)i)j )
            comm_cart_rows.Allgatherv(WSij, [WSi, counts_WSi, displ_WSi, MPI.DOUBLE])
            # pij 在二维并行更新S时已在列进程中gather了Vj
            # pij 计算 (WSVT)ij = (WS)i dot Vj.T
            WSVTij = WSi.dot(Vj.T)
            # pij 计算 Aij = Dij - (WSVT)ij，计算 aij = norm(Aij)^2
            aij = np.linalg.norm(Dij - WSVTij)**2   # norm自带开平方，所以reduce前要平方取消根号
            # 0号进程计算 a = reduce(aij, root=0)
            a = comm.reduce(aij, op=MPI.SUM, root=0)

            break_flag = False
            if comm_rank == 0:
                loss = a
                loss_list.append(loss)
                f = open(loss_trend_file, "a")
                f.write(str(loss) + "\n")
                f.close()
                if len(loss_list) > 2 and abs(loss_list[-1] - loss_list[-2]) < 1e-4:    # 收敛条件可以在experiment.ini中设置
                    break_flag = True
            break_flag = comm.bcast(break_flag, root=0)
            if break_flag:
                break   # 所有进程一起break
        # compute loss ---End，并行计算loss


    if comm_rank == 0:
        total_iter_time = time.time() - time_start  # timing    # 总迭代时间

    '''Result Gathering'''
    ccounts_gather_Wij = np.empty(comm_size, dtype='i')
    comm.Allgather(np.array([W_ccount], dtype='i'), ccounts_gather_Wij)
    counts_gather_Wij = ccounts_gather_Wij * num_word_topic
    displ_gather_Wij = np.insert(np.cumsum(counts_gather_Wij), 0, 0)[0:-1]
    comm.Gatherv(Wij, [W, counts_gather_Wij, displ_gather_Wij, MPI.DOUBLE], root=0)
    
    # 二维gather方法，先gather到col首，再由col首gather到row首
    # 先gather到col首
    Vj = np.empty((rcount, num_doc_topic))
    displ_Vji = np.insert(np.cumsum(counts_Vji), 0, 0)[0:-1]
    comm_cart_cols.Gatherv(Vji, [Vj, counts_Vji, displ_Vji, MPI.DOUBLE], root=0)
    # 再由col首gather到row首
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
    