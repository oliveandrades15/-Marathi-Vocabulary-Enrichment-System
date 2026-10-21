"""
Step-by-step accuracy improvement for the synonym classifier.

Run from the project folder (needs cc.mr.300.vec and marathi_dictionary.csv):

    python improve_accuracy.py

Everything is tuned on the TRAINING pairs only (cross-validation). The test
pairs are used once per row, only to report the result. Uses the same 80/20
split by word as app.py.
"""
import ast
import difflib
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


# ---------------------------------------------------------------- features
def common_prefix_ratio(a, b):
    prefix = os.path.commonprefix([str(a), str(b)])
    m = min(len(a), len(b))
    return len(prefix) / m if m > 0 else 0


def common_suffix_ratio(a, b):
    return common_prefix_ratio(a[::-1], b[::-1])


def bigram_jaccard(a, b):
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / len(A | B) if (A | B) else 0.0


def basic_features(wv, a, b):
    return [float(wv.similarity(a, b)), common_prefix_ratio(a, b), abs(len(a) - len(b))]


def extended_features(wv, a, b):
    ra = np.log1p(wv.key_to_index[a])   # fastText vectors are sorted by frequency
    rb = np.log1p(wv.key_to_index[b])
    return basic_features(wv, a, b) + [
        common_suffix_ratio(a, b),
        bigram_jaccard(a, b),
        difflib.SequenceMatcher(None, a, b).ratio(),
        abs(ra - rb),
        min(ra, rb),
    ]


# ------------------------------------------------------------------- data
def build_pairs_and_features():
    from gensim.models import KeyedVectors

    df = pd.read_csv("marathi_dictionary.csv", encoding="utf-8-sig")
    df["synonyms_list"] = df["synonyms_list"].apply(ast.literal_eval)
    dict_lookup = dict(zip(df["word"], df["synonyms_list"]))
    wv = KeyedVectors.load_word2vec_format("cc.mr.300.vec", limit=500000)

    rng = random.Random(SEED)
    words = [w for w in dict_lookup if w in wv]
    rng.shuffle(words)
    split = int(0.8 * len(words))
    train_words, test_words = words[:split], words[split:]

    def pairs(word_list, neg_pool):
        out = []
        for w in word_list:
            true_syns = set(dict_lookup[w])
            for s in dict_lookup[w]:
                if s in wv and s != w:
                    out.append((w, s, 1))
            neg = rng.choice(neg_pool)
            tries = 0
            while (neg == w or neg in true_syns) and tries < 10:
                neg = rng.choice(neg_pool)
                tries += 1
            if neg != w and neg not in true_syns:
                out.append((w, neg, 0))
        return out

    tr, te = pairs(train_words, train_words), pairs(test_words, test_words)

    def mat(ps, fn):
        return np.array([fn(wv, a, b) for a, b, _ in ps])

    y_tr = np.array([p[2] for p in tr])
    y_te = np.array([p[2] for p in te])
    return (mat(tr, basic_features), mat(tr, extended_features), y_tr,
            mat(te, basic_features), mat(te, extended_features), y_te)


# ------------------------------------------------------------ experiments
def metrics(name, y_true, pred):
    return {
        "Step": name,
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced acc": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


def run_experiments(X3_tr, Xe_tr, y_tr, X3_te, Xe_te, y_te):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []

    # 1. current model
    m1 = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    m1.fit(X3_tr, y_tr)
    rows.append(metrics("1. Current (3 features, default SVM)", y_te, m1.predict(X3_te)))

    # 2. more features
    m2 = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    m2.fit(Xe_tr, y_tr)
    rows.append(metrics("2. + extended features", y_te, m2.predict(Xe_te)))

    # 3. tuned hyper-parameters (grid search on training data only)
    grid = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        {"svc__C": [0.5, 1, 3, 10, 30],
         "svc__gamma": ["scale", 0.03, 0.1, 0.3],
         "svc__class_weight": [None, "balanced"]},
        cv=cv, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(Xe_tr, y_tr)
    rows.append(metrics(f"3. + tuned SVM {grid.best_params_}", y_te, grid.predict(Xe_te)))

    # 4. tuned decision threshold (chosen from out-of-fold TRAIN scores)
    best = grid.best_estimator_
    oof = cross_val_predict(best, Xe_tr, y_tr, cv=cv, method="decision_function")
    cands = np.linspace(np.percentile(oof, 5), np.percentile(oof, 95), 80)
    accs = [accuracy_score(y_tr, oof >= t) for t in cands]
    thr = float(cands[int(np.argmax(accs))])
    best.fit(Xe_tr, y_tr)
    pred = best.decision_function(Xe_te) >= thr
    rows.append(metrics(f"4. + tuned threshold ({thr:.2f})", y_te, pred.astype(int)))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    X3_tr, Xe_tr, y_tr, X3_te, Xe_te, y_te = build_pairs_and_features()
    print(f"Train pairs: {len(y_tr)}  Test pairs: {len(y_te)}  "
          f"Positive share (test): {y_te.mean():.1%}\n")
    out = run_experiments(X3_tr, Xe_tr, y_tr, X3_te, Xe_te, y_te)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(out.round(3).to_string(index=False))
    out.round(4).to_csv("accuracy_improvement.csv", index=False)
    print("\nSaved accuracy_improvement.csv")
    