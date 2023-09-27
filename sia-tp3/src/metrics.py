from typing import Dict

"""
Metrics seria un diccionario de la forma
Metrics ={
    true_p:0,
    true_n:0,
    false_p:0,
    false_n:0
}
"""


def accuracy(metrics: Dict) -> float:
    true_p = metrics['tp']
    true_n = metrics['tn']
    false_n = metrics['fn']
    false_p = metrics['fp']
    return (true_n + true_p) / (true_n + true_p + false_p + false_n)


def precision(metrics: Dict) -> float:
    true_p = metrics['tp']
    false_p = metrics['fp']
    if true_p + false_p == 0:
        return 0
    return true_p / (true_p + false_p)


def recall(metrics: Dict) -> float:
    true_p = metrics['tp']
    false_n = metrics['fn']
    if true_p + false_n == 0:
        return 0
    return true_p / (true_p + false_n)


def f1_score(metrics: Dict) -> float:
    prec = precision(metrics)
    rec = recall(metrics)
    if prec + rec == 0:
        return 0
    return (2 * prec * rec) / (prec + rec)


def true_p_rate(metrics: Dict) -> float:
    return recall(metrics)


def false_p_rate(metrics: Dict) -> float:
    false_p = metrics['fp']
    true_n = metrics['tn']
    if false_p + true_n == 0:
        return 0
    return false_p / (false_p + true_n)
