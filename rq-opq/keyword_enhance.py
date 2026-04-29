from tqdm import tqdm
import numpy as np
from datetime import datetime
from typing import List
import ahocorasick # pip install pyahocorasick


class AhoCorasickMatcher:
    def __init__(self, keywords: List[str]):
        self.automaton = ahocorasick.Automaton()
        for keyword in sorted(set(keywords), key=lambda x: -len(x)):
            self.automaton.add_word(keyword.lower(), keyword)
        self.automaton.make_automaton()

    def match(self, text: str) -> List[str]:
        matched = set()
        for _, keyword in self.automaton.iter(text.lower()):
            matched.add(keyword)
        return list(matched)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def keyword_enhance_emb(infile, dict_path, dim):
    cpv_library = []
    dic_ner_tag, dic_ner_emb = {}, {}
    with open(dict_path, 'r') as nf:
        for line in tqdm(nf, desc='loading ner_dict', mininterval=60):
            # 'red', 'color', pv, emb
            ner, tag, pv, ner_emb = line.strip().split('\t') 
            ner_emb = [float(x) for x in ner_emb.split('\x02')]
            if len(ner_emb) != dim:
                continue
            cpv_library.append(ner)
            if dic_ner_tag.get(ner):
                dic_ner_tag[ner].append(tag)
            else:
                dic_ner_tag[ner] = [tag]
            dic_ner_emb[ner] = ner_emb
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'keywords len: {len(cpv_library)}')

    matcher = AhoCorasickMatcher(cpv_library)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'AhoCorasickMatcher prepared!')
    total_embs = []
    source_embs = []
    keys = []
    ners = []

    with open(infile, 'r') as f:
        for line in tqdm(f, mininterval=60):
            line = line.rstrip('\n')
            query, raw_emb = line.split('\t')
            if query == '\\N' or len(query) == 0: continue

            if '\x02' in raw_emb:
                raw_emb_list = [float(x) for x in raw_emb.split('\x02')]
            else:
                raw_emb_list = [float(x) for x in raw_emb.strip('[]').split(',')]
            if len(raw_emb_list) != dim:
                print(f'invalid embedding dim ({len(raw_emb_list)} != {dim}): {query}')
                continue

            raw_emb = np.array(raw_emb_list, dtype=np.float32)
            matched_list = matcher.match(query)

            if len(matched_list) == 0:
                keys.append(query)
                ners.append([])
                source_embs.append(raw_emb)
                total_embs.append(raw_emb)
                continue

            matched_list = sorted(matched_list, key=lambda x: -len(x))
            res = [matched_list[0]]
            for subterm in matched_list[1:]:
                if not any(subterm in resterm for resterm in res):
                    res.append(subterm)

            new_attrs = res
            keys.append(query)
            ners.append(new_attrs)
            source_embs.append(raw_emb)

            new_emb_list = [dic_ner_emb[p] for p in new_attrs if p in dic_ner_emb]
            if len(new_emb_list) == 0:
                total_embs.append(raw_emb)
                continue

            keyword_embs = np.array(new_emb_list, dtype=np.float32)
            fused_emb = 0.5 * raw_emb + 0.5 * keyword_embs.mean(axis=0)
            fused_emb = l2_normalize(fused_emb)
            if fused_emb.shape[0] != dim:
                total_embs.append(raw_emb)
                continue
            total_embs.append(fused_emb)

    emb_arrays = np.vstack(total_embs)
    del total_embs
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"key_len: {len(keys)}, emb_shape: {emb_arrays.shape}")

    return keys, emb_arrays, source_embs, ners
