# -*- coding: utf-8 -*-

import os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz
from tqdm import tqdm
from networkx.readwrite import json_graph
import json
import networkx as nx
from collections import defaultdict
import argparse

datasets = ['Eurlex-4K']
suffix = ['X.trn.npz', 'X.tst.npz', 'X.val.npz', 'Y.trn.npz', 'Y.tst.npz', 'Y.val.npz']

K = 10 # for kNN
kNN_type = 3
distance_type = 'L2'

def find_edges(input, test):
    print(f"building kNN classifier ... ", end=" ")
    st_time = time.time()
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
        tree.build(n_threads=10)
    else:
        raise NotImplementedError
    print(f"time={time.time()-st_time:.3f}s")


    print("finding indices ... ", end=" ")
    if kNN_type == 1:
        _, indices = tree.kneighbors(test)
    elif kNN_type == 2:
        _, indices = tree.query(test, k=K + 1)
    else:
        indices = []
        for i in tqdm(range(test.shape[0])):
            indices.append(tree.search_by_vector(test[i, :], k=K + 1))
    print(f"time={time.time()-st_time:.3f}s")


    edge_list = []
    for index1, per in enumerate(indices):
        for index2 in per:
            if index1 != index2:
                edge_list.append((index1, index2))
    print(f"done! .... time={time.time()-st_time:.3f}s")
    return edge_list

def create_json_file(edge, fea, tra_id, val_id, tst_id, dataset_name, suffix=None):
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


    file_path = f"./data/{dataset_name}/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path += "{dataset_name}-"
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
    sp_fea = sp.csr_matrix(fea).tolil()
    sp.save_npz(file_path, sp_fea)
    # np.save(file_path + 'feats.npy', fea) # dense store

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_type", default=3, type=int, choices=[1, 2, 3], help="the algorithm of finding kNN")
    parser.add_argument("--distance_type", default="L2", type=str, choices=["L2", "angular"], help="the way to evaluate the smiliarity of two samples")
    args = parser.parse_args()
    kNN_type = args.knn_type
    distance_type = args.distance_type

    for dataset in datasets:
        begin_time = time.time()
        print(f"start preprocessing {dataset} ...")
        file_path = './vanilla_data/' + dataset + '/'
        data = []
        nums = []
        for j in range(len(suffix)):
            tmp = np.array(load_npz(file_path + suffix[j]).todense())
            data.append(tmp)
            nums.append(data[-1].shape[0])
            print(f"{suffix[j]}.shape={data[-1].shape}", end=' ')
        print("")
        tra_fea, tst_fea, val_fea, tra_lab, tst_lab, val_lab = data
        tra_num, tst_num, val_num = nums[0], nums[1], nums[2]

        tra_val_fea = np.concatenate((tra_fea, val_fea), axis=0)
        all_fea = np.concatenate((tra_val_fea, tst_fea), axis=0)
        x_edge_list = find_edges(tra_val_fea, all_fea)

        tra_id = [i for i in range(tra_num)]
        val_id = [i for i in range(tra_num, tra_num + val_num)]
        tst_id = [i for i in range(tra_num + val_num, tra_num + val_num + tst_num)]

        create_json_file(x_edge_list, all_fea, tra_id, val_id, tst_id, dataset, suffix='X-')
        del tra_val_fea
        del all_fea

        print(f"finish features ... time={time.time()-begin_time:.3f}s")

        # for each label, find the feature set
        label_num = tra_lab.shape[0]
        tra_val_lab = np.concatenate((tra_lab, val_lab), axis=0)
        print(tra_val_lab.shape)
        y_x_id = [[] for i in range(label_num)]
        indx, indy = tra_val_lab.nonzero()
        for i in range(len(indx)):
            y_x_id[indy[i]].append(indx[i])

        s = sum([len(i) for i in y_x_id])
        assert s == len(indx), f"s={s} len(indx)={len(indx)}"

        fea_dim = tra_fea.shape[1]
        label_fea = np.zeros((label_num, fea_dim))
        error_label = []
        for i in tqdm(range(label_num)):
            if len(y_x_id[i]) == 0:
                # print(f"label id = {i} has no corresponding examples !!!!")
                error_label.append(i)
                continue
            label_fea[i, :] = np.mean(tra_val_lab[y_x_id[i], :])

        y_edge_list = find_edges(label_fea, label_fea)
        y_tra_id = [i for i in range(label_fea.shape[0])]
        y_val_id, y_tst_id = [], []
        create_json_file(y_edge_list, label_fea, y_tra_id, y_val_id, y_tst_id, dataset, suffix='Y-')

        print(f"# error label = {len(error_label)}")
        print(f"finish {dataset} ... time={time.time()-begin_time:.3f}s")