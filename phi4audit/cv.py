"""One evaluation protocol for every stage: 5-fold cross-validated ROC AUC, repeated over
20 shuffles of the fold assignment, reported as mean +/- std of the per-repeat means.
`groups` switches to StratifiedGroupKFold, so no group appears in both train and test."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPEATS, FOLDS = 20, 5


def hgb():
    return HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.1,
                                          random_state=0)


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


MODELS = {"gbt": hgb, "logistic": logit}


def cv_auc(make_clf, X, y, groups=None, shuffle_labels=False):
    means, skipped = [], 0
    for seed in range(REPEATS):
        yy = y.copy()
        if shuffle_labels:
            np.random.RandomState(1000 + seed).shuffle(yy)
        if groups is None:
            splits = StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, yy)
        else:
            splits = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(X, yy, groups)
        aucs = []
        for tr, te in splits:
            if len(set(yy[te])) < 2:      # a test fold with no positive family cannot be scored
                skipped += 1
                continue
            clf = make_clf().fit(X[tr], yy[tr])
            aucs.append(roc_auc_score(yy[te], clf.predict_proba(X[te])[:, 1]))
        if aucs:
            means.append(np.mean(aucs))
    m = np.array(means)
    return dict(mean=round(float(m.mean()), 3), std=round(float(m.std()), 3),
                folds_unscorable=skipped, folds_total=REPEATS * FOLDS)


def evaluate(X, y, groups=None):
    out = {name: cv_auc(mk, X, y, groups) for name, mk in MODELS.items()}
    out["shuffled_labels"] = {name: cv_auc(mk, X, y, groups, shuffle_labels=True)["mean"]
                              for name, mk in MODELS.items()}
    return out
