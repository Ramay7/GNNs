# -*- coding: utf-8 -*-

import numpy
import sys
import nmslib
import time
import math
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

def matrToStrArray(sparseMatr):
    res = []
    indptr = sparseMatr.indptr
    indices = sparseMatr.indices
    for row in range(sparseMatr.shape[0]):
        arr = [k for k in indices[indptr[row] : indptr[row + 1]]]
        arr.sort()
        res.append(' '.join([str(k) for k in arr]))
    return res

if __name__ == '__main__':
    x = [[0, 2, 0], [1, 2, 3], [0, 1, 2], [-1, 0, 7], [0, 0, 7]]
    data = sp.csr_matrix(x)

    # Set index parameters
    # These are the most important onese
    M = 30
    efC = 100

    num_threads = 4
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}

    # Number of neighbors
    K = 3

    # Intitialize the library, specify the space, the type of the data and add data points
    # Note that in the GENERIC case, data points are passed as strings!
    space_names = ['jaccard_sparse', 'cosinesimil_sparse', 'l2']
    space_name = space_names[1]

    if space_name == 'jaccard_sparse':
        data_input = matrToStrArray(data)
        query_input = data_input.copy()
        data_type = nmslib.DataType.OBJECT_AS_STRING
    elif space_name == 'cosinesimil_sparse':
        data_input = data.copy()
        query_input = data.copy()
        data_type = nmslib.DataType.SPARSE_VECTOR
    else:
        raise NotImplementedError

    index = nmslib.init(method='hnsw', space=space_name, data_type=data_type)
    index.addDataPointBatch(data_input)

    # Create an index
    start = time.time()
    index.createIndex(index_time_params)
    end = time.time()
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end - start))

    # Setting query-time parameters
    efS = 100
    query_time_params = {'efSearch': efS}
    print('Setting query-time parameters', query_time_params)
    index.setQueryTimeParams(query_time_params)

    # Querying
    if space_name == 'jaccard_sparse':
        query_qty = len(query_input)
    elif space_name == 'cosinesimil_sparse':
        query_qty = query_input.shape[0]
    else:
        raise NotImplementedError

    start = time.time()
    nbrs = index.knnQueryBatch(query_input, k=K, num_threads=num_threads)
    end = time.time()
    print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % (
    end - start, float(end - start) / query_qty, num_threads * float(end - start) / query_qty))

    tt = [i[0] for i in nbrs]
    for index, p in enumerate(tt):
        print(f"index={index} p = {p} type(p)={type(p)}")