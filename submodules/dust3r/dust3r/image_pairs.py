# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed to load image pairs
# --------------------------------------------------------
import numpy as np
import torch

import os
from PIL import Image
import numpy as np
from numpy import linalg

# 计算图片的余弦距离
def calculate_cosine_similarity(image1, image2):
    image1 = image1.cuda().float().flatten()
    image2 = image2.cuda().float().flatten()
    norm1 = torch.norm(image1)
    norm2 = torch.norm(image2)
    similarity = torch.dot(image1 / norm1, image2 / norm2).cpu().item()
    return similarity

# function：生成从start到end，以start为参考的匹配对。如(0, 1) (0, 2) ....
def MakePair(start, end):
    pairs = []
    for i in range(start, end):
        if i != start:
            tmp_pair = [start, i]
            pairs.append(tmp_pair)
    return pairs

def MakePairPolicy(indices, ratio):
    result = []
    prev_end = 0
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i+1]
        length = end - prev_end
        start = 0 if start == 0 else int(start - length * ratio)
        pairs = MakePair(start, end)
        result.extend(pairs)
        prev_end = end

    return result

def make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True):
    pairs = []
    if scene_graph == 'complete':  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('swin'):
        winsize = int(scene_graph.split('-')[1]) if '-' in scene_graph else 3
        pairsid = set()
        for i in range(len(imgs)):
            for j in range(1, winsize+1):
                idx = (i + j) % len(imgs)  # explicit loop closure
                pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('oneref'):
        refid = int(scene_graph.split('-')[1]) if '-' in scene_graph else 0
        for j in range(len(imgs)):
            if j != refid:
                pairs.append((imgs[refid], imgs[j]))
    elif scene_graph.startswith('consine'):

        if len(imgs) < 2:
            print("Not enough images to compare.")
            return []

        similarities = []
        for i in range(len(imgs) - 1):
            similarity = calculate_cosine_similarity(imgs[i]['img'][0], imgs[i + 1]['img'][0])
            similarities.append(similarity)

        k = 2
        similarity_array = np.array(similarities)
        differences = np.abs(np.diff(similarity_array))
        sorted_indices = np.argsort(differences)[::-1]
        indices = sorted(list(map(int, sorted_indices[:k])) + [0, len(imgs) - 1])
        result = MakePairPolicy(indices, 0.6)
        for pair in result:
            pairs.append((imgs[pair[0]], imgs[pair[1]]))
                
    if symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]

    # now, remove edges
    if isinstance(prefilter, str) and prefilter.startswith('seq'):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]))

    if isinstance(prefilter, str) and prefilter.startswith('cyc'):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]), cyclic=True)

    return pairs


def sel(x, kept):
    if isinstance(x, dict):
        return {k: sel(v, kept) for k, v in x.items()}
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x[kept]
    if isinstance(x, (tuple, list)):
        return type(x)([x[k] for k in kept])


def _filter_edges_seq(edges, seq_dis_thr, cyclic=False):
    # number of images
    n = max(max(e) for e in edges)+1

    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i-j)
        if cyclic:
            dis = min(dis, abs(i+n-j), abs(i-n-j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def filter_pairs_seq(pairs, seq_dis_thr, cyclic=False):
    edges = [(img1['idx'], img2['idx']) for img1, img2 in pairs]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def filter_edges_seq(view1, view2, pred1, pred2, seq_dis_thr, cyclic=False):
    edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    print(f'>> Filtering edges more than {seq_dis_thr} frames apart: kept {len(kept)}/{len(edges)} edges')
    return sel(view1, kept), sel(view2, kept), sel(pred1, kept), sel(pred2, kept)
