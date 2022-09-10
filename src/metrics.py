import re
import string
import logging
import unicodedata

from typing import Any, Dict, Tuple
from collections import Counter

logger = logging.getLogger("apiLogger")

def normalize_answer(s):
    s = unicodedata.normalize("NFD", s)

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

def compute_metrics(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall

def update_metrics(metrics, em, f1, prec, recall):
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, f1, prec, recall

"""
Code for hotpotQA evaluation modified from https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
"""
def hotpot_qa_eval(
    predictions: Dict[str, str],
    ground_truths: Dict[str, str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}

    per_question_metrics = {}

    for label in ground_truths:
        cur_id = label['id']
        if cur_id not in predictions:
            logger.info('missing answer {}'.format(cur_id))
        else:
            em, f1, prec, recall = compute_metrics(
                predictions[cur_id], label['answer']
            )

            update_metrics(metrics, em, f1, prec, recall)
            per_question_metrics[cur_id] = {
                'em': em, 'f1': f1, 'prec': prec, 'recall': recall
            }

    N = len(ground_truths)
    for k in metrics.keys():
        metrics[k] /= N

    logger.info(metrics)

    return metrics, per_question_metrics

def nq_open_eval(
    predictions: Dict[str, str],
    ground_truths: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}

    per_question_metrics = {}

    for id, label in enumerate(ground_truths):
        prediction = predictions[id]
        # Get the max value over ever answer
        em_scores, f1_scores, prec_scores, recall_scores = [], [], [], []
        # TODO: Can we put this in a nice list comprehension?
        for answer in label['answer']:
            em, f1, prec, recall = compute_metrics(prediction, answer)
            em_scores.append(em)
            f1_scores.append(f1)
            prec_scores.append(prec)
            recall_scores.append(recall)

        em = max(em_scores)
        f1 = max(f1_scores)
        prec = max(prec_scores)
        recall = max(recall_scores)

        update_metrics(metrics, em, f1, prec, recall)
        per_question_metrics[id] = {
            'em': em, 'f1': f1, 'prec': prec, 'recall': recall
        }

    N = len(ground_truths)
    for k in metrics.keys():
        metrics[k] /= N

    logger.info(metrics)

    return metrics, per_question_metrics