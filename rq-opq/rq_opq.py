import os
import json
import numpy as np
import faiss
import pickle
import logging
from tqdm import tqdm
import gc
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FullDataOPQEncoder:
    def __init__(self, n_codebook=2, codebook_size=256):
        self.n_codebook = n_codebook       # number of codebooks (sub-spaces)
        self.codebook_size = codebook_size # size of each codebook
        self.n_codebook_bits = int(np.log2(codebook_size)) # bits per codebook, 256 = 2^8, so 8 bits per codebook index
        self.index_factory = f'OPQ{n_codebook},IVF1,PQ{n_codebook}x{self.n_codebook_bits}'
        self.index = None
        
        logger.info(f"OPQ config: n_codebook={n_codebook}, codebook_size={codebook_size}")
    
    def load_metadata(self, txt_path):
        """Load metadata file"""
        logger.info(f"Loading metadata: {txt_path}")
        
        data_list = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading metadata", mininterval=60):
                line = line.rstrip('\n')
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        key, sid = parts
                        data_list.append([key, sid])
                    else:
                        logger.warning(f"Malformed line: {line}")
        
        logger.info(f"Metadata loaded: {len(data_list)} entries")
        return data_list
    
    def load_embeddings(self, pkl_path):
        """Load embedding file"""
        logger.info(f"Loading embeddings: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            embeddings = pickle.load(f)

        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"Embeddings loaded: {embeddings.shape}")
        
        return embeddings
    
    def check_data_consistency(self, data_list, embeddings):
        """Check data consistency"""
        logger.info("Checking data consistency...")
        
        meta_count = len(data_list)
        emb_count = embeddings.shape[0]
        
        logger.info(f"Metadata rows: {meta_count}")
        logger.info(f"Embeddings rows: {emb_count}")
        
        if meta_count != emb_count:
            logger.error(f"Data mismatch! Metadata: {meta_count}, Embeddings: {emb_count}")
            raise ValueError(f"Row count mismatch: metadata={meta_count}, embeddings={emb_count}")
        
        logger.info("Data consistency check passed")
        return True
    
    def train(self, embeddings):
        """Train OPQ model"""
        # OPQ training: learn the optimal rotation matrix so that rotated vectors are better suited for product quantization
        # PQ training: run K-means clustering in each sub-space to build the codebook
        logger.info("Starting OPQ model training...")
        
        # Build index
        self.index = faiss.index_factory(
            embeddings.shape[1],         # vector dimension
            self.index_factory,          # index config string
            faiss.METRIC_INNER_PRODUCT   # use inner product as similarity metric
        )
        
        # Train
        self.index.train(embeddings)
        logger.info("OPQ model training complete")
    
    def encode(self, embeddings):
        """Encode vectors"""
        logger.info("Starting encoding...")
        # Reset index
        self.index.reset()  
        # Add to index
        self.index.add(embeddings)
        
        # Extract codes
        ivf_index = faiss.downcast_index(self.index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        
        # Decode
        results = []
        n_bytes = pq_codes.shape[1]
        
        for i in tqdm(range(len(pq_codes)), desc="Decoding", mininterval=60):
            u8code = pq_codes[i]
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for j in range(self.n_codebook):
                code.append(bs.read(self.n_codebook_bits))
            results.append(code)
        
        logger.info(f"Encoding complete: {len(results)} entries")
        return results
    
    def save_results(self, data_list, opq_codes, output_path, batch_size=100000):
        """Save results in streaming"""
        logger.info(f"Saving results: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in tqdm(range(0, len(data_list)), desc="Saving", mininterval=60):
                key = data_list[i][0]
                sid =  f"{data_list[i][1]}_{opq_codes[i][0]}_{opq_codes[i][1]+256}"
                f.write(f"{key}\t{sid}\n")
        
        logger.info("Save complete")

def get_opq_ids(model_path, txt_path, emb_path, output_path, n_codebook=2, codebook_size=256):
    n_codebook_bits = int(np.log2(codebook_size))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    index = faiss.read_index(model_path)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"{model_path} loaded")

    data_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading metadata", mininterval=60):
            line = line.rstrip('\n')
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    key, sid = parts
                    data_list.append([key, sid])
                else:
                    print(f"Malformed line: {line}")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"Metadata loaded: {len(data_list)} entries")
    
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"Loading embeddings: {emb_path}")
    with open(emb_path, 'rb') as f:
        embeddings = pickle.load(f)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"Embeddings loaded: {embeddings.shape}")

    assert len(data_list) == embeddings.shape[0], f'len error: item_len{len(data_list)} emb_len{embeddings.shape[0]}'
    index.reset()
    index.add(embeddings)
    # Extract codes
    ivf_index = faiss.downcast_index(index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    pq_codes = pq_codes.reshape(-1, invlists.code_size)
    # Decode
    results = []
    n_bytes = pq_codes.shape[1]
    f = open(output_path, 'w')
    for i in tqdm(range(len(pq_codes)), desc="opq decode", mininterval=60):
        u8code = pq_codes[i]
        bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
        key, rq = data_list[i]
        rq_opq = f"{rq}_{bs.read(n_codebook_bits)}_{bs.read(n_codebook_bits) + 256}"
        f.write(f"{key}\t{rq_opq}\n")
    
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"OPQ calculated, results saved into: {output_path}")


if __name__ == "__main__":
    '''1. train '''
    # main(query_sid_file, query_residual_emb_file, item_sid_file, item_residual_emb_file, model_path)

    '''2. infer '''
    # get_opq_ids(model_path, txt_path, emb_path, output_path)


