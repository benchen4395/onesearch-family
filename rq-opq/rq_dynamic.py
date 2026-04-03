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

def balanced_kmeans_last_layer_with_l2(M, l2_ids, k2, k3, niter: int = 25, max_points_per_centroid: int = 10000, verbose: bool = False):
    N, d = M.shape
    l3_ids = np.full(N, -1, dtype=np.int64)
    all_centroids = np.zeros((k2 * k3, d), dtype=np.float32)
    residuals = M.copy()

    for p in tqdm(range(k2), desc="L3 balanced kmeans (per L2 parent)"):
        mask = (l2_ids == p)
        idx  = np.where(mask)[0]
        n_p  = len(idx)

        if n_p == 0:
            # Empty parent – leave centroids as zeros, assign first code
            continue

        sub_M = M[idx]  # (n_p, d)

        if n_p <= k3:
            # Fewer points than centroids: assign each point its own centroid, and pad remaining centroids with the group mean.
            local_ids = np.arange(n_p, dtype=np.int64)
            local_centroids = sub_M.copy()
            if n_p < k3:
                mean_vec = sub_M.mean(axis=0, keepdims=True)  # (1, d)
                pad = np.tile(mean_vec, (k3 - n_p, 1))
                local_centroids = np.vstack([local_centroids, pad])
        else:
            # balance kmeans
            kmeans = faiss.Kmeans(d, k3, niter=niter, verbose=verbose, max_points_per_centroid=max_points_per_centroid)
            kmeans.train(sub_M)
            _, I = kmeans.index.search(sub_M, 1)
            local_ids = I.reshape(-1)
            local_centroids = kmeans.centroids  # (k3, d)

        # Map local codes → global L3 codes
        # global_l3 = p * k3 + local_l3, range [0, k2*k3)
        global_ids = (p * k3 + local_ids).astype(np.int64)
        l3_ids[idx] = global_ids

        # Store centroids at the correct slice
        all_centroids[p * k3 : (p + 1) * k3] = local_centroids

        # Compute residuals for these points
        assigned_centroids = local_centroids[local_ids]  # (n_p, d)
        residuals[idx] = sub_M - assigned_centroids

    return l3_ids, all_centroids, residuals


def balanced_kmeans_last_layer_with_l1_l2(M, l1_ids, l2_ids, k1, k2, k3, niter: int = 25, max_points_per_centroid: int = 10000, verbose: bool = False):
    N, d = M.shape
    n_parents  = k1 * k2
    total_centroids = n_parents * k3
    l3_ids = np.full(N, -1, dtype=np.int64)
    all_centroids = np.zeros((total_centroids, d), dtype=np.float32)
    residuals = M.copy()

    # Precompute composite parent id: p = l1 * k2 + l2, range [0, k1*k2)
    parent_ids = l1_ids.astype(np.int64) * k2 + l2_ids.astype(np.int64)

    for p in tqdm(range(n_parents), desc="L3 balanced kmeans (per L1×L2 parent)"):
        mask = (parent_ids == p)
        idx  = np.where(mask)[0]
        n_p  = len(idx)

        if n_p == 0:
            # Empty parent – leave centroids as zeros, assign first code
            continue

        sub_M = M[idx]  # (n_p, d)

        if n_p <= k3:
            # Fewer points than centroids: assign each point its own centroid, and pad remaining centroids with the group mean.
            local_ids = np.arange(n_p, dtype=np.int64)
            local_centroids = sub_M.copy()
            if n_p < k3:
                mean_vec = sub_M.mean(axis=0, keepdims=True)  # (1, d)
                pad = np.tile(mean_vec, (k3 - n_p, 1))
                local_centroids = np.vstack([local_centroids, pad])
        else:
            # balance kmeans
            kmeans = faiss.Kmeans(d, k3, niter=niter, verbose=verbose, max_points_per_centroid=max_points_per_centroid)
            kmeans.train(sub_M)
            _, I = kmeans.index.search(sub_M, 1)
            local_ids = I.reshape(-1)
            local_centroids = kmeans.centroids  # (k3, d)

        # Map local codes → global L3 codes
        # global_l3 = (l1 * k2 + l2) * k3 + local_l3  =  p * k3 + local_l3, range [0, k1*k2*k3)
        global_ids = (p * k3 + local_ids).astype(np.int64)
        l3_ids[idx] = global_ids

        # Store centroids at the correct slice
        all_centroids[p * k3 : (p + 1) * k3] = local_centroids

        # Compute residuals for these points
        assigned_centroids = local_centroids[local_ids]  # (n_p, d)
        residuals[idx] = sub_M - assigned_centroids

    return l3_ids, all_centroids, residuals



def quantitative_codebook(folder_path, refer_folder, M, k_list, start_L, L, is_norm, balanced_type):
    if M.dtype != np.float32:
        M = M.astype('float32') 
    N = M.shape[0]  # Number of keys
    d = M.shape[1]  # Size of embedding
    RQCodeList = []
    IdList = []
    
    # For continue training
    if refer_folder != '' and start_L != 0:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'Refer kmeans: {refer_folder}, refer layers:{start_L}')
        f = open(f"{refer_folder}/RQCodeList-{start_L}.pkl", 'rb')
        RQCodeList = pickle.load(f)
        for i in range(len(RQCodeList)):
            if i < L - 1 or balanced_type == 0:
                expected_k = k_list[i]
            elif balanced_type == 1: 
                expected_k = k_list[i - 2] * k_list[i - 1] * k_list[i]
            else:
                expected_k = k_list[i - 1] * k_list[i]
            if RQCodeList[i].shape[0] != expected_k:
                print(f'Error: layer {i+1} cluster num conflicts! refer {RQCodeList[i].shape[0]} != target {expected_k}')
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

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'Kmeans start! d:{d}, K:{k_list}, start_L:{start_L}, L:{L}, balanced_type:{balanced_type}')

    for i in range(start_L, L):
        is_balanced   = (i == L - 1) and (balanced_type != 0) and (i >= 1)

        if is_balanced and balanced_type == 1:
            # ---------------------------------------------------------------
            # L3 Balanced K-means (type 1: per-(L1,L2) parent):
            #   For each (L1, L2) pair, independently run K-means with size k3.
            #   l1_ids from IdList[i-2], l2_ids from IdList[i-1].
            #   Global id = (l1 * k2 + l2) * k3 + l3
            #   L3 codebook size = k1 * k2 * k3
            # ---------------------------------------------------------------
            assert i >= 2, f"balanced_type=1 requires 3 layers, but i={i}"
            k1 = k_list[i - 2]  # number of L1 clusters
            k2 = k_list[i - 1]  # number of L2 clusters
            k3 = k_list[i]      # sub-codebook size per (L1,L2) parent
            l1_ids = IdList[i - 2]  # shape (N,)
            l2_ids = IdList[i - 1]  # shape (N,)

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'L3 BALANCED(type=1, L1×L2) clustering start! \
                  k1={k1}, k2={k2}, k3(per-parent codebook)={k3}, L3 centroids={k1 * k2 * k3}')

            l3_ids, all_centroids, M = balanced_kmeans_last_layer_with_l1_l2(M, l1_ids, l2_ids, k1, k2, k3, niter=25, max_points_per_centroid=10000)

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'L3 BALANCED(type=1, L1×L2) clustering complete! \
                  id range [{l3_ids.min()}, {l3_ids.max()}], unique ids used: {len(np.unique(l3_ids))} / {k1 * k2 * k3}')

            IdList.append(l3_ids)
            RQCodeList.append(all_centroids)  # (k1*k2*k3, d)

        elif is_balanced and balanced_type == 2:
            # ---------------------------------------------------------------
            # L3 Balanced K-means (type 2: per-L2 parent):
            #   For each L2 parent, independently run K-means with codebook size k3.
            #   Global id = l2 * k3 + l3
            #   L3 codebook size = k2 * k3
            # ---------------------------------------------------------------
            k2 = k_list[i - 1]  # number of L2 clusters
            k3 = k_list[i]      # sub-codebook size per parent
            l2_ids = IdList[i - 1]  # shape (N,)

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'L3 BALANCED(type=2, L2) clustering start! \
                  k2={k2}, k3(per-parent codebook)={k3}, L3 centroids={k2 * k3}')

            l3_ids, all_centroids, M = balanced_kmeans_last_layer_with_l2(M, l2_ids, k2, k3, niter=25, max_points_per_centroid=10000)

            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'L3 BALANCED(type=2) clustering complete! \
                  id range [{l3_ids.min()}, {l3_ids.max()}], unique ids used: {len(np.unique(l3_ids))} / {k2 * k3}')

            IdList.append(l3_ids)
            RQCodeList.append(all_centroids)  # (k2*k3, d)

        else:
            # Standard K-means
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'L{i+1} clustering start! K:{k_list[i]}')
            kmeans = faiss.Kmeans(d, k_list[i], niter=25, verbose=True, max_points_per_centroid=10000)
            kmeans.train(M)
            D, I = kmeans.index.search(M, 1)
            I = I.reshape(-1)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'L{i+1} clustering complete! {I.shape}')
            IdList.append(I)
            O = kmeans.centroids[I]
            M = M - O
            RQCodeList.append(kmeans.centroids)

        # L2-normalise residuals if requested (standard mode only)
        if is_norm and not is_balanced:
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            M = M / (norms + 1e-8)

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

def get_semantic_ids(folder_path, data, IdList, k_list, split_id, querytxtfile, itemtxtfile, balanced_type):
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
    centroid_cnt = 0
    for i in range(L):
        is_last = (i == L - 1) and (i >= 2)
        if is_last and balanced_type == 1 and i >= 2:
            centroid_cnt += k_list[i - 2] * k_list[i - 1] * k_list[i]
        elif is_last and balanced_type == 2:
            centroid_cnt += k_list[i - 1] * k_list[i]
        else:
            centroid_cnt += k_list[i]
        size = size * k_list[i]
        print(f"cluster layer {i+1} count: {len(cluster[i])}\tcodebook size: {size}\tutilization: {round(len(cluster[i])/size*100, 2)}%")
    print(f"total centroids: {centroid_cnt}")

    return cluster

def main(embfile, keyfile, querynum, folder_path, refer_folder, k_list, start_L, L, is_norm, balanced_type):
    try:
        os.makedirs(folder_path)
        print(f"{folder_path} created.")
    except FileExistsError:
        print(f"{folder_path} exists!")

    """step1: clustering"""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'loading {embfile}') 
    with open(embfile, 'rb') as ef:
        emb_arrays = pickle.load(ef)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'emb_arrays shape:', emb_arrays.shape) 

    RQCodeList, IdList = quantitative_codebook(folder_path, refer_folder, emb_arrays, k_list, start_L, L, is_norm, balanced_type)

    """step2: assigning semantic_id"""
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "assigning semantic_id ...")
    with open(keyfile, 'r', encoding='utf-8') as f:
        keywords = [line.rstrip('\n').split('\t', 1)[0] for line in f]
    with open(f'{folder_path}/IdList-{L}.pkl', 'rb') as pf:
        IdList = pickle.load(pf)
    if len(keywords) != IdList[0].shape[0]:
        print(f"Invalid:{keyfile} len: {len(keywords)}, not equal to IdList shape: {IdList[0].shape[0]}")
    cluster = get_semantic_ids(folder_path, keywords, IdList, k_list, querynum, f'{folder_path}/results-l{L}-query.txt', f'{folder_path}/results-l{L}-item.txt', balanced_type)

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
    parser.add_argument('--k', type=str, default='1024-1024-1024', help='cluster num, only support 3 layers when L3 balanced')
    parser.add_argument('--isnorm', type=int, default=0, help='cluster use norm or not')
    parser.add_argument('--balanced', type=int, default=0,
                        help='0 = standard global k-means (default); \
                            1 = use L1-L2-parent balanced k-means on the last layer (L3 balanced); \
                            2 = use L2-parent balanced k-means on the last layer (L3 balanced).')
    
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
    print(f"balanced: {args.balanced}")
    # ---------------------------------------------------------------------------
    # Balanced K-means for the last RQ layer (L3 balanced)
    # ---------------------------------------------------------------------------
    # Core idea (from OneSearch §3.1.3):
    #   Standard RQ-Kmeans uses a single global codebook per layer, so the last
    #   layer codebook size is k3 (shared across ALL parent clusters).  This leads
    #   to very low CUR/ICR because many leaf clusters are never used.
    #
    #   L3-balanced assigns each L1-L2 / L2 parent its OWN sub-codebook of size k3.
    #   Items under the same parent compete only within that group, which forces 
    #   uniform utilisation.  The total last-layer parameter count becomes
    #   k1 * k2 * k3 (for L1-L2-parent) / k2 * k3 (for L2-parent) instead of k3.
    # ---------------------------------------------------------------------------
    if args.balanced > 0 and len(k_list) != 3:
        print("ERROR: L3 balancing requires 3 RQ layers!")
        exit()
    
    main(args.embfile, args.keyfile, querynum, args.folder, args.referfolder, k_list, args.referlayer, len(k_list), args.isnorm, args.balanced)


    
    

