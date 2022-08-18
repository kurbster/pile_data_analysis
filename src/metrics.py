"""
Code for hotpotQA evaluation modified from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
"""
import re
import string
import logging

from typing import Any, List, Dict
from collections import Counter

logger = logging.getLogger("apiLogger")

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # If the ground truth and prediction only contained stop words this is empty
    if normalized_prediction == "" and normalized_ground_truth == "":
        return (1.0, 1.0, 1.0)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def hotpot_qa_eval(
    predictions: Dict[str, str],
    ground_truths: Dict[str, str]
) -> Dict[str, Any]:
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}

    for label in ground_truths:
        cur_id = label['id']
        if cur_id not in predictions:
            logger.info('missing answer {}'.format(cur_id))
        else:
            update_answer(
                metrics, predictions[cur_id], label['answer']
            )

    N = len(ground_truths)
    for k in metrics.keys():
        metrics[k] /= N

    logger.info(metrics)

    return metrics
