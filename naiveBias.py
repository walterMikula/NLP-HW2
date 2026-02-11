import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

TOKEN_RE = re.compile(r"[a-z']+")

def tokenize(text: str) -> List[str]:
    # super simple tokenizer: lowercase + words/apostrophes
    return TOKEN_RE.findall(text.lower())

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="latin-1") as f:
        return [line.strip() for line in f if line.strip()]

def train_dev_test_split(
    pos_lines: List[str],
    neg_lines: List[str],
    seed: int = 0,
) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]], List[Tuple[str,int]]]:
    data = [(x, 1) for x in pos_lines] + [(x, 0) for x in neg_lines]
    rnd = random.Random(seed)
    rnd.shuffle(data)

    n = len(data)
    n_train = int(0.70 * n)
    n_dev = int(0.15 * n)
    train = data[:n_train]
    dev = data[n_train:n_train + n_dev]
    test = data[n_train + n_dev:]
    return train, dev, test

def build_vocab(
    train_data: List[Tuple[str,int]],
    max_vocab: int = 5000,
    min_count: int = 2,
) -> Dict[str,int]:
    # count tokens in training only
    freq = Counter()
    for text, _ in train_data:
        freq.update(tokenize(text))

    # apply min_count then take top max_vocab
    items = [(w, c) for w, c in freq.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_vocab]

    vocab = {w:i for i,(w,_) in enumerate(items)}
    return vocab

def vectorize(text: str, vocab: Dict[str,int]) -> Counter[int]:
    counts = Counter()
    for w in tokenize(text):
        if w in vocab:
            counts[vocab[w]] += 1
    return counts

@dataclass
class NBModel:
    vocab: Dict[str,int]
    alpha: float
    log_prior_pos: float
    log_prior_neg: float
    log_p_w_pos: List[float]  # log P(word|pos)
    log_p_w_neg: List[float]  # log P(word|neg)

def train_multinomial_nb(
    train_data: List[Tuple[str,int]],
    vocab: Dict[str,int],
    alpha: float = 1.0,
) -> NBModel:
    V = len(vocab)

    # class priors
    n_pos = sum(y for _, y in train_data)
    n_total = len(train_data)
    n_neg = n_total - n_pos
    # avoid log(0)
    log_prior_pos = math.log((n_pos + 1e-12) / n_total)
    log_prior_neg = math.log((n_neg + 1e-12) / n_total)

    # token counts per class
    pos_counts = [0] * V
    neg_counts = [0] * V
    total_pos_tokens = 0
    total_neg_tokens = 0

    for text, y in train_data:
        vec = vectorize(text, vocab)
        if y == 1:
            for idx, c in vec.items():
                pos_counts[idx] += c
                total_pos_tokens += c
        else:
            for idx, c in vec.items():
                neg_counts[idx] += c
                total_neg_tokens += c

    # Laplace smoothing
    denom_pos = total_pos_tokens + alpha * V
    denom_neg = total_neg_tokens + alpha * V

    log_p_w_pos = [math.log((pos_counts[i] + alpha) / denom_pos) for i in range(V)]
    log_p_w_neg = [math.log((neg_counts[i] + alpha) / denom_neg) for i in range(V)]

    return NBModel(
        vocab=vocab,
        alpha=alpha,
        log_prior_pos=log_prior_pos,
        log_prior_neg=log_prior_neg,
        log_p_w_pos=log_p_w_pos,
        log_p_w_neg=log_p_w_neg,
    )

def predict_logprobs(model: NBModel, text: str) -> Tuple[float,float]:
    vec = vectorize(text, model.vocab)
    lp_pos = model.log_prior_pos
    lp_neg = model.log_prior_neg
    for idx, c in vec.items():
        lp_pos += c * model.log_p_w_pos[idx]
        lp_neg += c * model.log_p_w_neg[idx]
    return lp_pos, lp_neg

def predict(model: NBModel, text: str) -> Tuple[int, float]:
    lp_pos, lp_neg = predict_logprobs(model, text)
    # confidence via probability from log-odds
    # p(pos) = 1/(1+exp(lp_neg-lp_pos))
    diff = lp_pos - lp_neg
    # clamp for safety
    diff = max(min(diff, 60.0), -60.0)
    p_pos = 1.0 / (1.0 + math.exp(-diff))
    yhat = 1 if p_pos >= 0.5 else 0
    conf = max(p_pos, 1 - p_pos)  # confidence in chosen label
    return yhat, conf

def evaluate(model: NBModel, data: List[Tuple[str,int]]) -> Dict[str,float]:
    tp = tn = fp = fn = 0
    for text, y in data:
        yhat, _ = predict(model, text)
        if y == 1 and yhat == 1: tp += 1
        elif y == 0 and yhat == 0: tn += 1
        elif y == 0 and yhat == 1: fp += 1
        else: fn += 1
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def most_useful_features(model: NBModel, top_k: int = 20) -> Tuple[List[Tuple[str,float]], List[Tuple[str,float]]]:
    # log-odds: log P(w|pos) - log P(w|neg)
    inv_vocab = {i:w for w,i in model.vocab.items()}
    scores = []
    for i in range(len(model.vocab)):
        s = model.log_p_w_pos[i] - model.log_p_w_neg[i]
        scores.append((inv_vocab[i], s))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_pos = scores[:top_k]
    top_neg = list(reversed(scores[-top_k:]))
    return top_pos, top_neg

def confident_and_uncertain_examples(model: NBModel, test: List[Tuple[str,int]], k: int = 5):
    rows = []
    for text, y in test:
        yhat, conf = predict(model, text)
        lp_pos, lp_neg = predict_logprobs(model, text)
        diff = lp_pos - lp_neg
        rows.append((conf, abs(diff), y, yhat, text))
    rows.sort(key=lambda r: r[0], reverse=True)
    most_conf = rows[:k]
    rows.sort(key=lambda r: r[0])  # least confident
    most_unc = rows[:k]
    return most_conf, most_unc

def main():
    pos = load_lines("rt-polarity.pos")
    neg = load_lines("rt-polarity.neg")
    train, dev, test = train_dev_test_split(pos, neg, seed=0)

    # ---- Step 2: tune on dev (keep this small/simple) ----
    alphas = [0.5, 1.0, 2.0]
    vocab_sizes = [2000, 5000, 10000]
    min_counts = [1, 2]

    best = None
    best_score = -1.0
    best_cfg = None

    for min_count in min_counts:
        for max_vocab in vocab_sizes:
            vocab = build_vocab(train, max_vocab=max_vocab, min_count=min_count)
            for alpha in alphas:
                model = train_multinomial_nb(train, vocab, alpha=alpha)
                m = evaluate(model, dev)
                if m["accuracy"] > best_score:
                    best_score = m["accuracy"]
                    best = model
                    best_cfg = (alpha, max_vocab, min_count, m)

    print("Best dev config:", best_cfg)

    # ---- Step 3: retrain on train+dev with best hyperparams ----
    alpha, max_vocab, min_count, _ = best_cfg
    train_plus = train + dev
    vocab = build_vocab(train_plus, max_vocab=max_vocab, min_count=min_count)
    final_model = train_multinomial_nb(train_plus, vocab, alpha=alpha)

    test_metrics = evaluate(final_model, test)
    print("Test metrics:", test_metrics)

    # ---- Step 4: confident vs uncertain examples ----
    most_conf, most_unc = confident_and_uncertain_examples(final_model, test, k=5)
    print("\nMOST CONFIDENT (conf, |logodds|, gold, pred, text):")
    for row in most_conf:
        conf, logodds_abs, y, yhat, text = row
        print(f"{conf:.3f}\t{logodds_abs:.2f}\t{y}\t{yhat}\t{text[:120]}...")

    print("\nMOST UNCERTAIN (conf, |logodds|, gold, pred, text):")
    for row in most_unc:
        conf, logodds_abs, y, yhat, text = row
        print(f"{conf:.3f}\t{logodds_abs:.2f}\t{y}\t{yhat}\t{text[:120]}...")

    # ---- Step 5: most useful features ----
    top_pos, top_neg = most_useful_features(final_model, top_k=20)
    print("\nTop POSITIVE words (word, log-odds):")
    for w,s in top_pos:
        print(w, f"{s:.3f}")

    print("\nTop NEGATIVE words (word, log-odds):")
    for w,s in top_neg:
        print(w, f"{s:.3f}")

if __name__ == "__main__":
    main()
