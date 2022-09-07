import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def drag_topK(ts, length, r, exclusionIndices):
    # 初步选出候选下标，可能含有假阳性
    ts = np.array(ts)
    min_separation = length
    test = ts[0:length]
    C = dict({
        0: {
            'series': (ts[0:length] - ts[0:length].mean()) / ts[0:length].std()
        }
    })

    subsequence_cnt = ts.shape[0] - length + 1
    for i in range(1, subsequence_cnt):
        processing_ts = np.array(ts[i:i + length])
        processing_ts_nor = (processing_ts - processing_ts.mean()) / processing_ts.std()
        is_candidate = True
        keys = list(C.keys())
        for idx in keys:
            c = C[idx]
            if abs(idx - i) < min_separation:
                continue
            d = np.linalg.norm(processing_ts_nor - c['series'])
            if d < r:
                C.pop(idx)
                is_candidate = False
                break
        if is_candidate:
            C[i] = {
                'series': processing_ts_nor
            }

    # 候选集中去除已经选中的下标
    for [idx, c] in C.items():
        # print(idx)
        for exclusionIdx in exclusionIndices:
            if abs(idx - exclusionIdx) <= length:
                C.pop(idx)
                break

    # 候选集为空
    if not C:
        print('First candidate set is empty.')
        print('Failure: The r parameter of ', r, 'was too large for the algorithm to work')
        disc_dist = -float('inf')
        disc_loc = -1
        return disc_dist, disc_loc

    # 优化候选集
    # 找到所有候选集距离自己最近邻居的距离
    r2 = r ** 2
    D = dict()
    D = D.fromkeys(C.keys(), float('inf'))
    for i in range(subsequence_cnt):
        if not C:
            break
        processing_ts = np.array(ts[i:i + length])
        processing_ts_nor = (processing_ts - processing_ts.mean()) / processing_ts.std()
        keys = list(C.keys())
        for idx in keys:
            c = C[idx]
            if abs(idx - i) < min_separation:
                continue
            d = np.square(processing_ts_nor - c['series'])
            d = d.sum()
            distance = D[idx]
            if r2 > d:
                C.pop(idx)
                D.pop(idx)
                continue
            elif d < distance:
                D[idx] = d
    # 候选集为空
    if not C:
        print('Final candidate set is empty.')
        print('Failure: The r parameter of ', r, 'was too large for the algorithm to work')
        disc_dist = -float('inf')
        disc_loc = -1
        return disc_dist, disc_loc
    else:
        # 距离最大的候选方案
        max_dist = 0
        max_idx = -1
        for [idx, d] in D.items():
            if d > max_dist:
                max_dist = d
                max_idx = idx
        max_dist = math.sqrt(max_dist)
        print('The top discord of the series is at ', max_idx, ' with a discord distance of max_dist.')
    return max_dist, idx


def MERLIN_topK(T, MinL, MaxL, K):
    numLengths = MaxL - MinL + 1
    distances = -np.ones((numLengths, K))
    indices = np.zeros((numLengths, K))
    exclusionIndices = []
    kMultipler = 1
    lengths = range(MinL, MaxL + 1)
    r = 2 * math.sqrt(MinL)
    for ki in range(K):
        while distances[0][ki] < 0:
            [distances[0][ki], indices[0][ki]] = drag_topK(T, MinL, r * kMultipler, exclusionIndices)
            if ki == 0:
                r *= 0.5
            else:
                kMultipler *= 0.95
            exclusionIndices.append(indices[0][ki])
    for i in range(1, 5):
        if i > numLengths:
            return
        exclusionIndices = []
        kMultipler = 1
        r = distances[i - 1][0] * 0.99
        for ki in range(K):
            while distances[i, ki] < 0:
                distances[i][ki], indices[i][ki] = drag_topK(T, lengths[i], r * kMultipler, exclusionIndices)
                if ki == 1:
                    r *= 0.99
                else:
                    kMultipler *= 0.95
            exclusionIndices.append(indices[i][ki])
    if numLengths < 5:
        return
    for i in range(5, numLengths):
        exclusionIndices = []
        kMultipler = 1
        m = distances[i - 5:i - 1, 0].mean()
        s = distances[i - 5:i - 1, 0].std()
        r = m - 2 * s
        for ki in range(K):
            while distances[i, ki] < 0:
                print(i, '------:', ki)
                distances[i][ki], indices[i][ki] = drag_topK(T, lengths[i], r * kMultipler, exclusionIndices)
                if ki == 1:
                    r *= 0.99
                else:
                    kMultipler *= 0.95
            exclusionIndices.append(indices[i][ki])

    plt.figure(figsize=(100, 6.0))
    plt.subplot(2, 1, 1)
    plt.plot(range(data_values.shape[0]), data_values)
    plt.subplot(2, 1, 2)
    # plt.Rectangle((1, MinL), T.shape[0], MaxL)
    plt.xlim([0, T.shape[0]])
    plt.ylim([MaxL, MinL])
    markerSize = 4
    for ki in range(K):
        if ki == 0:
            plt.scatter(indices[:, ki], lengths, markerSize, c='r', marker='.')
        else:
            plt.scatter(indices[:, ki], lengths, markerSize, marker='.')
    plt.show()


if __name__ == '__main__':
    data_path = './data/artificialWithAnomaly/art_increase_spike_density.csv'
    data = pd.read_csv(data_path)
    data_values = np.array(data['value'])
    MERLIN_topK(data_values, 100, 200, 1)
