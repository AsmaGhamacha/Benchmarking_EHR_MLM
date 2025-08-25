# qa_metrics.py

import re
import string
import collections

def normalize_answer(s):
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gt_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return int(pred_tokens == gt_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_exact(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)

def compute_em_f1(predictions, references):
    total = len(references)
    exact_scores = []
    f1_scores = []

    for ref in references:
        qid = ref["id"]
        gold_answers = ref["answers"]["text"]
        if not gold_answers:
            gold_answers = [""]
        pred = predictions.get(qid, "")
        exact = metric_max_over_ground_truths(compute_exact, pred, gold_answers)
        f1 = metric_max_over_ground_truths(compute_f1, pred, gold_answers)
        exact_scores.append(exact)
        f1_scores.append(f1)

    return {
        "eval_exact_match": 100.0 * sum(exact_scores) / total,
        "eval_f1": 100.0 * sum(f1_scores) / total
    }
