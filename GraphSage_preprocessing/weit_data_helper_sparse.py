# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from networkx.readwrite import json_graph
import json
import networkx as nx
from collections import defaultdict
import argparse
from sklearn.cluster import KMeans

datasets = ['Eurlex-4K']
datasets = ['Wiki10-31K']
datasets = ['AmazonCat-13K']
suffix = ['X.trn.npz', 'X.tst.npz', 'X.val.npz', 'Y.trn.npz', 'Y.tst.npz', 'Y.val.npz']

# K = 10 # for kNN
kNN_type = 4
distance_type = 'L2'

'''
Solutions for kNN:
    * sklearn (NearestNeighbors, dense input)
    * scipy (KDTree, dense input)
    * n2 (HnswIndex, dense input): https://github.com/kakao/n2
    * PySparNN (sparse input): https://github.com/facebookresearch/pysparnn
    * nmslib (sparse input): https://github.com/nmslib/nmslib
'''

def find_edges(input, test, K, cluster_ids, query_ids):
    print(f"\tbuilding kNN classifier ... ", end=" ")
    st_time = time.time()

    if kNN_type in [1, 2]:
        input, test = input.todense(), test.todense()

    if kNN_type == 1:
        from sklearn.neighbors import NearestNeighbors
        tree = NearestNeighbors(n_neighbors=K + 1, algorithm='ball_tree').fit(input)
    elif kNN_type == 2:
        from scipy import spatial
        tree = spatial.KDTree(input)
    elif kNN_type == 3:
        from n2 import HnswIndex
        tree = HnswIndex(input.shape[1], distance_type) # distance_type in ['angular', 'L2']
        for index in tqdm(range(input.shape[0])):
            tree.add_data(input[index, :])
        tree.build(n_threads=20)
    elif kNN_type == 4:
        import pysparnn.cluster_index as ci
        input_num = input.shape[0]
        tree = ci.MultiClusterIndex(input, range(input_num))
    elif kNN_type == 5:
        import nmslib
        M, efC, num_threads = 30, 100, 10
        index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
        space_name = 'cosinesimil_sparse'
        data_type = nmslib.DataType.SPARSE_VECTOR
        tree = nmslib.init(method='hnsw', space=space_name, data_type=data_type)
        
        print(f"type(input) = {type(input)} type(test)={type(test)}", end=" ")
        
        tree.addDataPointBatch(input)

        tree.createIndex(index_time_params)
        # Setting query-time parameters
        efS = 100
        query_time_params = {'efSearch': efS}
        print('Setting query-time parameters', query_time_params)
        tree.setQueryTimeParams(query_time_params)

    else:
        raise NotImplementedError
    print(f"time={time.time()-st_time:.3f}s")


    print("\tfinding indices ... ", end=" ")
    if kNN_type == 1:
        _, indices = tree.kneighbors(test)
    elif kNN_type == 2:
        _, indices = tree.query(test, k=K + 1)
    elif kNN_type == 3:
        indices = []
        for i in tqdm(range(test.shape[0])):
            indices.append(tree.search_by_vector(test[i, :], k=K + 1))
    elif kNN_type == 4:
        indices = tree.search(test, k=K+1, k_clusters=100, return_distance=False)
    elif kNN_type == 5:
        indices_ = tree.knnQueryBatch(test, k=K, num_threads=num_threads)
        indices = [i[0] for i in indices_]
        del indices_
    else:
        raise NotImplementedError

    print(f"time={time.time()-st_time:.3f}s")


    edge_list = []
    for index1, per in enumerate(indices):
        for index2 in per:
            index2 = int(index2)
            if index1 != index2:
                edge_list.append((query_ids[index1], center_ids[index2]))
    print(f"\tdone! .... time={time.time()-st_time:.3f}s")
    return edge_list

def create_json_file(edge, fea, tra_id, val_id, tst_id, dataset_name, suffix=None, K=10):
    G = nx.Graph()
    G.add_nodes_from(tra_id)
    G.add_nodes_from(val_id)
    G.add_nodes_from(tst_id)
    G.add_edges_from(edge)

    for id in tra_id:
        G.node[id]["test"] = False
        G.node[id]["val"] = False
    for id in val_id:
        G.node[id]["test"] = False
        G.node[id]["val"] = True
    for id in tst_id:
        G.node[id]["test"] = True
        G.node[id]["val"] = False


    file_path = f"./data/{dataset_name}/{K}/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path += f"{dataset_name}-"
    if suffix is not None:
        file_path += suffix


    # edge list
    data = json_graph.node_link_data(G)
    with open(file_path + "G.json", mode="w") as f:
        f.write(json.dumps(data))

    # id_map
    id_map = {}
    all_ids = tra_id.copy()
    all_ids.extend(val_id)
    all_ids.extend(tst_id)
    for i in all_ids:
        assert i not in id_map.keys(), f"i={i} already in id_map"
        id_map[i] = i

    with open(file_path + "id_map.json", mode="w") as f:
        f.write(json.dumps(id_map))

    # feature
    sp.save_npz(file_path + 'feats', fea)
    # np.save(file_path + 'feats.npy', fea.todense()) # dense store

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_type", default=4, type=int, choices=[1, 2, 3, 4, 5], help="the algorithm of finding kNN")
    parser.add_argument("--distance_type", default="L2", type=str, choices=["L2", "angular"], help="the way to evaluate the smiliarity of two samples")
    parser.add_argument("--use_prepro_y", default=False, type=bool, help="whether or not use preprocessing features of labels")
    args = parser.parse_args()
    kNN_type = args.knn_type
    distance_type = args.distance_type

    for dataset in datasets:
        begin_time = time.time()
        print(f"start preprocessing {dataset} ...")
        file_path = './vanilla_data/' + dataset + '/'
        data, nums = [], []
        for j in range(len(suffix)):
            tmp = sp.load_npz(file_path + suffix[j])
            data.append(tmp)
            nums.append(data[-1].shape[0])
            print(f"{suffix[j]}={data[-1].shape}", end=' ')
        print("")
        tra_fea, tst_fea, val_fea, tra_lab, tst_lab, val_lab = data
        tra_num, tst_num, val_num = nums[0], nums[1], nums[2]

        tra_val_fea = sp.vstack([tra_fea, val_fea])
        all_fea = sp.vstack([tra_val_fea, tst_fea])

        tra_id = [i for i in range(tra_num)]
        val_id = [i for i in range(tra_num, tra_num + val_num)]
        tst_id = [i for i in range(tra_num + val_num, tra_num + val_num + tst_num)]

        #####################################
        # labels_reshape = labels.flatten()
        # n_labels = len(np.unique(labels_reshape))
        # labels_one_hot = np.eye(n_labels, dtype=int)[labels_reshape]
        # class_map = {k: list(labels_one_hot[i]) for i, k in enumerate(nodes.keys())}
        # with open("data-class_map.json", mode="w") as f:
        #     f.write(json.dumps(class_map, default=str))
        ######################################
        
        print(f"args.use_prepro_y = {args.use_prepro_y}")
        if (not args.use_prepro_y) or (not os.path.exists(file_path + "label_feat.npz")):
            print(f"preprocess label_features from scratch")
            # for each label, find the feature set
            label_num = tra_lab.shape[1]
            tra_val_lab = sp.vstack([tra_lab, val_lab])
            y_x_id = [[] for i in range(label_num)]
            indx, indy = tra_val_lab.nonzero()
            for i in range(len(indx)):
                y_x_id[indy[i]].append(indx[i])

            s = sum([len(i) for i in y_x_id])
            assert s == len(indx), f"check labels_num ERROR --- s={s} len(indx)={len(indx)}"

            fea_dim = tra_fea.shape[1]
            row_, col_, data_ = [], [], []
            label_fea = sp.csr_matrix((data_, (row_, col_)), shape=(label_num, fea_dim))#.tolil()
            tra_val_fea = tra_val_fea#.tolil()
            # tra_val_fea_dense = tra_val_fea.todense()
            # label_fea = np.zeros(shape=(label_num, fea_dim))
            error_label = []


            '''
            for i in tqdm(range(label_num)):
                if len(y_x_id[i]) == 0:
                    # print(f"label id = {i} has no corresponding examples !!!!")
                    error_label.append(i)
                    label_fea[i, 0] = 1e-10
                    continue
                # label_fea[i, :] = np.mean(tra_val_fea[y_x_id[i], :], axis=0)
                label_fea[i, :] = tra_val_fea[y_x_id[i], :].mean(axis=0)

            print(f"# error label = {len(error_label)}")
            label_fea = sp.csr_matrix(label_fea)
            '''

            def per_process(i):
                print (f'##################processing the {i}-th label########################')
                tmp = sp.csr_matrix((1, fea_dim))
                #tmp = sp.csr_matrix((data_, (row_, col_)), shape=(1, fea_dim))
                if len(y_x_id[i]) == 0:
                    error_label.append(i)
                    tmp[0,0] = 1e-10
                else:
                    tmp = tra_val_fea[y_x_id[i], :].mean(axis=0)
                tmp = sp.csr_matrix(tmp)
                assert len(tmp.nonzero()) > 0, f"Error i={i} len(tmp.nonzero())={len(tmp.nonzero())}"
                return tmp


            from multiprocessing.dummy import Pool as ThreadPool
            from multiprocessing import Pool
            from joblib import Parallel, delayed

            # pool = ThreadPool()
            # label_fea = pool.map(per_process, range(label_num))
            # pool.close()
            # pool.join()
            print ('parallel starting...')
            #label_fea = Parallel(n_jobs=25, prefer="threads")(delayed(per_process)(i) for i in range(label_num))
            #label_fea = sp.vstack(tuple(label_fea))

            pool = Pool(25)
            label_fea = pool.map(per_process, range(label_num))
            label_fea = sp.vstack(tuple(label_fea))
            sp.save_npz(file_path + 'label_feat.npz', label_fea)

            print ('parallel ending...')
            print (label_fea.shape)
            print (label_fea.getnnz())

        else:
            print(f"----load preprocessed label_features from {file_path + 'label_feat.npz'}")
            label_fea = sp.load_npz(file_path + 'label_feat.npz')


        y_tra_id = [i for i in range(label_fea.shape[0])]
        y_val_id, y_tst_id = [], []

        num_clusters = 3
        kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0, n_jobs=20, max_iter =10).fit(tra_val_fea)
        nearest_centers = kmeans.predict(all_fea)
        for K in [1, 5, 10, 20, 50, 100]:
            print(f"\nK={K} ...\npreprocessing features ...")
            t0 = time.time()

            x_edge_list = []
            for i in range(num_clusters):
                cluster_ids = np.where(kmeans.labels_ == i)[0]
                query_ids = np.where(nearest_centers == i)[0]
                x_edge_list += find_edges(tra_val_fea[cluster_ids, :], all_fea[query_ids, :], K, cluster_ids, query_ids)

            create_json_file(x_edge_list, all_fea, tra_id, val_id, tst_id, dataset, suffix='X-', K=K)
            
            print(f"preprocessing labels ...")		
            y_edge_list = find_edges(label_fea, label_fea, K)
            create_json_file(y_edge_list, label_fea, y_tra_id, y_val_id, y_tst_id, dataset, suffix='Y-', K=K)

        print(f"finish {dataset} time={time.time()-begin_time:.3f}s")
