import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict

TOKEN_RE = re.compile(r"[a-z']+")

def tokenize(text: str) -> List[str]:
    #tokenizer for lowercase words
    return TOKEN_RE.findall(text.lower())

def load_data(path: str) -> List[str]:
    with open(path, "r", encoding="latin-1") as f:
        return [line.strip() for line in f if line.strip()]

def train_dev_test_split(
    pos_lines: List[str],
    neg_lines: List[str],
    seed: int = 0,
) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]], List[Tuple[str,int]]]:
    data = [(x, 1) for x in pos_lines] + [(x, 0) for x in neg_lines] #setting positive reviews to the label 1 and the negative reviews to label 0
    rnd = random.Random(seed)
    rnd.shuffle(data)

    n = len(data)
    n_train = int(0.70 * n) # 70% train
    n_dev = int(0.15 * n)  #15% dev
    n_test = n - n_train - n_dev  #15% remaininf to set to teh test set
    train = data[:n_train]
    dev = data[n_train:n_train + n_dev]
    test = data[n_train + n_dev:]
    return train, dev, test

def build_vocabulary(
    train_data: List[Tuple[str,int]],
    max_vocab: int = 5000,
    min_count: int = 2,
) -> Dict[str,int]:
    freq = Counter() #count the tokens in the training dataa
    for text, _ in train_data:
        freq.update(tokenize(text))

    items = [(w, c) for w, c in freq.items() if c >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_vocab]

    vocab = {w:i for i,(w,_) in enumerate(items)} #give each word in teh dictionary an integer representation
    return vocab

def create_vectors(text: str, vocab: Dict[str,int]) -> Counter[int]:
    counts = Counter()
    for w in tokenize(text):
        if w in vocab: #if the token is in the vocab, add 1 to the count for that token's index in the vocab
            counts[vocab[w]] += 1
    return counts


@dataclass
class naive_bias:
    vocab: Dict[str,int]
    alpha: float
    log_prior_pos: float
    log_prior_neg: float
    log_p_w_pos: List[float]  # log P  (word | pos)
    log_p_w_neg: List[float]  # log P ( word | neg)

def train_naive_bias(
    train_data: List[Tuple[str,int]],
    vocab: Dict[str,int],
    alpha: float = 1.0,
) -> naive_bias:
    V = len(vocab)

    n_pos = sum(y for _, y in train_data)
    n_total = len(train_data)
    n_neg = n_total - n_pos
    log_prior_pos = math.log((n_pos + 1e-12) / n_total) # avoid log(0) if there are no positive examples
    log_prior_neg = math.log((n_neg + 1e-12) / n_total)# avoid log(0) if there are no negative examples
    pos_counts = [0] * V
    neg_counts = [0] * V
    total_pos_tokens = 0
    total_neg_tokens = 0

    for text, y in train_data:
        vec = create_vectors(text, vocab)
        if y == 1: #positive labels
            for idx, c in vec.items():
                pos_counts[idx] += c
                total_pos_tokens += c
        else:
            for idx, c in vec.items():
                neg_counts[idx] += c
                total_neg_tokens += c

    # smoothing
    denom_pos = total_pos_tokens + alpha * V
    denom_neg = total_neg_tokens + alpha * V

    log_p_w_pos = [math.log((pos_counts[i] + alpha) / denom_pos) for i in range(V)]
    log_p_w_neg = [math.log((neg_counts[i] + alpha) / denom_neg) for i in range(V)]

    return naive_bias(
        vocab=vocab, 
        alpha=alpha,
        log_prior_pos=log_prior_pos,
        log_prior_neg=log_prior_neg,
        log_p_w_pos=log_p_w_pos,
        log_p_w_neg=log_p_w_neg,
    )

def predict_logprobs(model: naive_bias, text: str) -> Tuple[float,float]:
    vec = create_vectors(text, model.vocab)
    lp_pos = model.log_prior_pos
    lp_neg = model.log_prior_neg
    for idx, c in vec.items():
        lp_pos += c * model.log_p_w_pos[idx]
        lp_neg += c * model.log_p_w_neg[idx]
    return lp_pos, lp_neg

#model will predict the label for a new review by calculating the log-probabilities of 
# the review being positive or negative 
# The function will return the predicted label 1 for positive, 0 for negative and a 
# confidence score based on the log-odds of the two classes.
def predict(model: naive_bias, text: str) -> Tuple[int, float]:
    lp_pos, lp_neg = predict_logprobs(model, text)
    diff = lp_pos - lp_neg
    diff = max(min(diff, 60.0), -60.0)
    p_pos = 1.0 / (1.0 + math.exp(-diff))
    yhat = 1 if p_pos >= 0.5 else 0
    conf = max(p_pos, 1 - p_pos)  # confidence in label
    return yhat, conf

def evaluate(model: naive_bias, data: List[Tuple[str,int]]) -> Dict[str,float]:
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

def most_useful_features(model: naive_bias, top_k: int = 20) -> Tuple[List[Tuple[str,float]], List[Tuple[str,float]]]:
    inv_vocab = {i:w for w,i in model.vocab.items()}
    scores = []
    for i in range(len(model.vocab)):
        s = model.log_p_w_pos[i] - model.log_p_w_neg[i]
        scores.append((inv_vocab[i], s))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_pos = scores[:top_k]
    top_neg = list(reversed(scores[-top_k:]))
    return top_pos, top_neg

def confident_and_uncertain_examples(model: naive_bias, test: List[Tuple[str,int]], k: int = 5):
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
    pos = load_data("rt-polarity.pos")
    neg = load_data("rt-polarity.neg")
    train, dev, test = train_dev_test_split(pos, neg, seed=0)

    # 2
    alphas = [0.5, 1.0, 2.0]
    vocab_sizes = [2000, 5000, 10000]
    min_counts = [1, 2]

    best_score = -1.0
    best_cfg = None 
    for min_count in min_counts:
        for max_vocab in vocab_sizes:
            vocab = build_vocabulary(train, max_vocab=max_vocab, min_count=min_count)
            for alpha in alphas:
                model = train_naive_bias(train, vocab, alpha=alpha)
                metrics = evaluate(model, dev)
                if metrics["accuracy"] > best_score:
                    best_score = metrics["accuracy"]
                    best_cfg = (alpha, max_vocab, min_count, metrics)

    alpha, max_vocab, min_count, dev_metrics = best_cfg
    print("\n=== Best Dev Config ===")
    print(f"alpha={alpha}, max_vocab={max_vocab}, min_count={min_count}")
    print(f"Development metrics: {dev_metrics}")

    # 3
    train_plus = train + dev
    vocab = build_vocabulary(train_plus, max_vocab=max_vocab, min_count=min_count)
    final_model = train_naive_bias(train_plus, vocab, alpha=alpha)

    test_metrics = evaluate(final_model, test)
    print("\nTEST METRICS")
    print(f"Accuracy : {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall   : {test_metrics['recall']*100:.2f}%")
    print(f"F1 Score : {test_metrics['f1']*100:.2f}%")

    # 4
    most_conf, most_unc = confident_and_uncertain_examples(final_model, test, k=5)

    def fmt_label(y):
        return "POS" if y == 1 else "NEG"

    print("\n=== Most Confident Predictions ===")
    for item in most_conf:
        
        conf, logodds_abs, y, yhat, text = item
        print(f"conf={conf:.3f} | logodds={logodds_abs:.2f} | true_label={fmt_label(y)} | predicted={fmt_label(yhat)}")
        print(f"  {text}\n")

    print("\n=== Most Uncertain Predictions ===")
    for item in most_unc:
        conf, logodds_abs, y, yhat, text = item
        print(f"conf={conf:.3f} | logodds={logodds_abs:.2f} | true_label={fmt_label(y)} | predicted={fmt_label(yhat)}")
        print(f"  {text}\n")

    #5
    top_pos, top_neg = most_useful_features(final_model, top_k=7)

    print("\n\nTop Positive Words ")
    for w, s in top_pos:
        print(f"{w}\t{s:.3f}")

    print("\n\nTop Negative Words ")
    for w, s in top_neg:
        print(f"{w}\t{s:.3f}")


if __name__ == "__main__":
    main()
