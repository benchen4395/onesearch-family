import os, pickle, csv, argparse
import faiss # pip install faiss-cpu
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

def merger_query_item(query_txt, item_txt, keyfile, embfile):
    keys, embs = [], []
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'processing {query_txt} ...')
    with open(query_txt, 'r') as f:
        for line in tqdm(f):
            try:
                query, ners, emb = line.rstrip('\n').split('\t')
            except:
                print(line.rstrip('\n'))
                continue

            emb = [float(i) for i in emb.replace('[', '').replace(']', '').replace(' ', '').split(',')]
            keys.append(f'{query}\t{ners}\n')
            embs.append(emb)
    
    querynum = len(keys)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'query len: {querynum}')

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'processing {item_txt} ...')
    with open(item_txt, 'r') as f:
        for line in tqdm(f):
            try:
                item_id, title, ners, emb = line.rstrip('\n').split('\t')
            except:
                print(line.rstrip('\n'))
                continue

            emb = [float(i) for i in emb.replace('[', '').replace(']', '').replace(' ', '').split(',')]
            keys.append(f'{item_id}\t{title}\t{ners}\n')
            embs.append(emb)

    itemnum = len(keys) - querynum
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'item len: {itemnum}')

    emb_array = np.array(embs)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'key len: {len(keys)}, emb shape: {emb_array.shape}, start writing...')

    with open(keyfile, 'w') as f:
        f.writelines(keys)
    with open(embfile, "wb") as f:
        pickle.dump(emb_array, f)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{keyfile} and {embfile} saved!")

    return querynum, itemnum

def quantitative_codebook(folder_path, refer_folder, M, k_list, start_L, L, is_norm):
    """
    Input:
    M : numpy.ndarray
        shape (N, dim).
    K : int
        Codebook size for each layer.
    L : int
        Codebook layers.

    Output:
    RQCodeList : list
        The list of residual codebooks.
    IdList : list
        The list of query semantic_ids.
    """
    if M.dtype != np.float32:
        M = M.astype('float32') 
    N = M.shape[0]  # Number of querys
    d = M.shape[1]  # Size of embedding
    RQCodeList = []
    IdList = []
    
    if refer_folder != '' and start_L != 0:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'Refer kmeans: {refer_folder}, refer layers:{start_L}')
        f = open(f"{refer_folder}/RQCodeList-{start_L}.pkl", 'rb')
        RQCodeList = pickle.load(f)
        for i in range(len(RQCodeList)):
            if RQCodeList[i].shape[0] != k_list[i]:
                print(f'Error: layer {i+1} cluster num conflicts! refer {RQCodeList[i].shape[0]} != target {k_list[i]}')
            else:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"RQCodeList: layer {i+1} cluster num {RQCodeList[i].shape[0]}")

        f = open(f'{refer_folder}/IdList-{start_L}.pkl', 'rb')
        IdList = pickle.load(f)
        if IdList[0].shape[0] != N:
            print(f'Error: data num conflicts! refer {IdList[0].shape[0]} != target {N}')
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"IdList: {len(IdList)}, {IdList[0].shape}")

        for i in tqdm(range(N), mininterval=60, desc="Residual Embedding"):
            for j in range(start_L):
                cluster_id = IdList[j][i]
                M[i] = M[i] - RQCodeList[j][cluster_id]
                M[i] = M[i] / (np.linalg.norm(M[i]) + 1e-8) if is_norm else M[i]
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Residual Embedding calculated!')

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'Kmeans start! d:{d}, K:{k_list}, start_L: {start_L}, L:{L}')
    for i in range(start_L, L):
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'Round {i+1} clustering start! K:{k_list[i]}')
        kmeans = faiss.Kmeans(d, k_list[i], niter=25, verbose=True, max_points_per_centroid=10000) # sample max_points_per_centroid per cluster, iterate niter times
        kmeans.train(M)
        D, I = kmeans.index.search(M, 1) # nearest neighbor search for each vector in M; I is the index of the nearest centroid, D is the distance
        I = I.reshape(-1) # flatten index array to 1D
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'Round {i+1} clustering complete! {I.shape}')
        IdList.append(I)
        O = kmeans.centroids[I] # retrieve centroid vectors corresponding to index I
        M = M - O # compute residuals: update M to be original vectors minus their assigned centroids
        if is_norm: # L2 normalization
            norms = np.linalg.norm(M, axis=1, keepdims=True)  # shape (N, 1)
            M = M / (norms + 1e-8)  # avoid division by zero
        RQCodeList.append(kmeans.centroids)

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'saving RQCodeList.pkl')
        with open(f'{folder_path}/RQCodeList-{i+1}.pkl', 'wb') as f:
            pickle.dump(RQCodeList, f)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'{folder_path}/RQCodeList-{i+1}.pkl saved!')
        
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'saving IdList.pkl')
        with open(f'{folder_path}/IdList-{i+1}.pkl', 'wb') as f:
            pickle.dump(IdList, f)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'{folder_path}/IdList-{i+1}.pkl saved!')

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'saving residual_emb.pkl')
        with open(f'{folder_path}/residual_emb_{i+1}.pkl', 'wb') as f:
            pickle.dump(M, f)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'{folder_path}/residual_emb_{i+1}.pkl saved! {M.shape}')
    
    return RQCodeList, IdList

def get_semantic_ids(folder_path, data, IdList, k_list, split_id, querytxtfile, itemtxtfile):
    L = len(k_list)
    assert L == len(IdList), f"Error: invalid IdList L {len(IdList)} not match k_list L {L}"
    print(f'IdList len:{len(IdList)} shape:{IdList[0].shape}')
    cluster = [{} for _ in range(L)]
    f1 = open(querytxtfile, 'w')
    f2 = open(itemtxtfile, 'w')
    for i, row in tqdm(enumerate(data), mininterval=300):
        ids = []
        for j in range(L):
            ids.append(str(IdList[j][i]))
        semantic_id = "_".join(ids)
        if i < split_id:
            f1.write(f"{row}\t{semantic_id}\n")
        else:
            f2.write(f"{row}\t{semantic_id}\n")
            for k in range(L):
                current_sid = semantic_id if k == L-1 else '_'.join(ids[:k+1])
                if current_sid in cluster[k]:
                    cluster[k][current_sid].append(row)
                else:
                    cluster[k][current_sid] = [row]

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'sid saved in: {querytxtfile} & {itemtxtfile}')

    file = f"{folder_path}/cluster.pkl"
    f = open(file, 'wb')
    pickle.dump(cluster, f)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{file} saved! len: {len(cluster)}")
    
    size = 1
    tokens = 0
    for i in range(L):
        tokens += k_list[i]
        size = size * k_list[i]
        print(f"cluster layer {i+1} count: {len(cluster[i])}\tcodebook size: {size}\tutilization: {round(len(cluster[i])/size*100, 2)}%")
    print(f"total tokens: {tokens}")

    return cluster


if __name__ == "__main__":
    # 0. load args
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='folder path')
    parser.add_argument('--referfolder', type=str, default='', help='folder path for continue rq')
    parser.add_argument('--referlayer', type=int, default=0, help='refer layers for continue rq')
    parser.add_argument('--querynum', type=int, default=0, help='for split')
    parser.add_argument('--querytxt', type=str, default='', help='for merge')
    parser.add_argument('--itemtxt', type=str, default='', help='for merge')
    parser.add_argument('--keyfile', type=str, help='query and item_id merged txt')
    parser.add_argument('--embfile', type=str, help='query_emb and item_emb merged pkl')
    parser.add_argument('--k', type=str, default='1024-1024-1024', help='cluster num')
    parser.add_argument('--isnorm', type=int, default=0, help='cluster use norm or not')
    
    args = parser.parse_args()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'args:', args)

    # 1. merge query & item data
    if args.querytxt and args.itemtxt:
        querynum, itemnum = merger_query_item(args.querytxt, args.itemtxt, args.keyfile, args.embfile)
    elif args.querynum > 0:
        querynum = args.querynum
    else:
        print('must have param [querynum]!!!')
        exit()

    # 2. RQ clustering
    k_list = [int(row) for row in args.k.split('-')]
    print(k_list)
    # main(args.embfile, args.keyfile, querynum, args.folder, args.referfolder, k_list, args.referlayer, len(k_list), args.isnorm, args.cnt, args.dictname)

    # 3. balance last layer
    # The relevant code is integrated into the corporate system.

    
    

