import codecs
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve, precision_recall_curve


def pr_curve(y_label, y_score):
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_label = y_label[desc_score_indices]
    tp, fp, tps, fps = 0., 0., [], []
    for i in y_label:
        if i == 1:
            tp += 1
        else:
            fp += 1
        fps.append(fp)
        tps.append(tp)

    precision, recall = [1], [0]
    for f, t in zip(fps, tps):
        precision.append(t / (t + f))
        recall.append(t / tps[-1])

    precision.append(0)
    recall.append(1)

    return precision, recall


def evalution(adj_rec, train_pos, test_pos):
    num = adj_rec.shape[0]
    x, y = np.triu_indices(num, k=1)

    c_set = set(zip(x, y)) - \
            set(zip(train_pos[:, 0], train_pos[:, 1])) - set(zip(train_pos[:, 1], train_pos[:, 0]))

    inx = np.array(list(c_set))
    Y = np.zeros((num, num))
    Y[test_pos[:, 0], test_pos[:, 1]] = 1
    Y[test_pos[:, 1], test_pos[:, 0]] = 1
    labels = Y[inx[:, 0], inx[:, 1]]
    val = adj_rec[inx[:, 0], inx[:, 1]]

    fpr, tpr, throc = roc_curve(labels, val)
    auc_val = auc(fpr, tpr)
    # prec, rec, thpr = precision_recall_curve(labels, val)
    prec, rec = pr_curve(labels, val)
    aupr_val = auc(rec, prec)

    f1_val = 0
    for i in range(len(prec)):
        if (prec[i] + rec[i]) == 0:
            continue
        f = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
        if f > f1_val:
            f1_val = f

    return auc_val, aupr_val, f1_val


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def ndcg_score(adj_rec, known_pairs, adj_lab, k=10):
    rank_preds = []

    num = adj_rec.shape[0]
    x, y = np.triu_indices(num, k=1)
    c_set = set(zip(x, y)) - set(zip(known_pairs[:, 0], known_pairs[:, 1])) - set(
        zip(known_pairs[:, 1], known_pairs[:, 0]))
    inx = np.array(list(c_set))
    Y = np.zeros((num, num))
    Y[inx[:, 0], inx[:, 1]] = adj_rec[inx[:, 0], inx[:, 1]]

    with tf.Session() as sess:
        _, pairs_id = sess.run(tf.nn.top_k(tf.reshape(Y, [-1]), k=k, name='topk'))

    for id in pairs_id:
        rank_preds.append(adj_lab[id // num, id % num])

    return ndcg_at_k(rank_preds, k)


def evalution_bal(adj_rec, edges_pos, edges_neg):
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []

    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
        if len(preds_neg) == len(preds):
            break

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    fpr, tpr, th = roc_curve(labels_all, preds_all)
    roc_score = auc(fpr, tpr)
    prec, rec, thr = precision_recall_curve(labels_all, preds_all)
    aupr_score = auc(rec, prec)

    for x, y in zip(fpr, tpr):
        roc = str(x) + '\t' + str(y)
        codecs.open('roc_bal.txt', mode='a', encoding='utf-8').write(roc + "\n")

    for x, y in zip(prec, rec):
        prc = str(x) + '\t' + str(y)
        codecs.open('prc_bal.txt', mode='a', encoding='utf-8').write(prc + "\n")

    return roc_score, aupr_score
