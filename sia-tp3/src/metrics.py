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
    true_p = metrics['true_p']
    true_n = metrics['true_n']
    false_n = metrics['false_n']
    false_p = metrics['false_p']
    return true_n + true_p / (true_n + true_p + false_p + false_n)


def precision(metrics: Dict) -> float:
    true_p = metrics['true_p']
    false_p = metrics['false_p']
    return true_p / (true_p + false_p)


def recall(metrics: Dict) -> float:
    true_p = metrics['true_p']
    false_n = metrics['false_n']
    return true_p / (true_p + false_n)


def f1_score(metrics: Dict) -> float:
    prec = precision(metrics)
    rec = recall(metrics)
    return (2 * prec * rec) / (prec + rec)


def true_p_rate(metrics: Dict) -> float:
    return recall(metrics)


def false_p_rate(metrics: Dict) -> float:
    false_p = metrics['false_p']
    true_n = metrics['true_n']
    return false_p / (false_p + true_n)
