import numpy as np
import torch
from bisect import bisect

pos_rate = 0.2403846153846154


def empirical_p_values(query):
    distribution = np.load('statisticalModule/pos_train_dist.npy')
    distribution = np.sort(distribution)
    dist_len = len(distribution)
    query_len = len(query)
    p_values = np.zeros([query_len, ])

    print(bisect(distribution, query[0]))
    for i, score in enumerate(query):
        p_values[i] = (bisect(distribution, score)) / dist_len
    return np.sort(p_values)


def calculate_qvalues_from_pvalues(query, pi_0=1 - pos_rate):
    p_values = empirical_p_values(query)
    q_values = p_values * len(p_values) * pi_0
    q_values = q_values / np.arange(1, len(p_values) + 1)
    for i in range(len(p_values) - 1, 0, -1):
        q_values[i - 1] = min(q_values[i - 1], q_values[i])

    return q_values


def calculate_fdr(scores, labels):
    sort_data = torch.sort(scores, descending=False)
    sorted_test_labels = labels[sort_data[1].data.cpu().numpy()]

    negative = 0
    positive = 0
    fdr = []
    for label in sorted_test_labels:
        negative += label.item() == 0
        positive += label.item() == 1
        fdr.append(positive / (negative + positive))
    return np.array(fdr)


def calculate_qvalues_from_labels(scores, labels):
    qvalue = calculate_fdr(scores, labels)
    for i in range(len(qvalue) - 1, 0, -1):
        qvalue[i - 1] = min(qvalue[i], qvalue[i - 1])
    return qvalue
