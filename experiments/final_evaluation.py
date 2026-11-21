"""Final comparison on VALIDATION (labels available) + test predictions (no test labels exist).
Models: Submission (quantum kernel), matched RBF control (same preprocessing), supplied reference. Seeds 0,1,2.
Run from the repository root:  python experiments/final_evaluation.py"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from challenge.data import load_data
from challenge.classical import reference
from challenge.metrics import score
from challenge.submission import Submission
from kernel_vs_control import paired_bootstrap

LABELS = ['alfalfa', 'maize', 'wheat']
OUT = ROOT / 'outputs' / 'final'; OUT.mkdir(parents=True, exist_ok=True)
tables, _, manifest, _ = load_data()
cfg = json.loads((ROOT / 'configs/final_model.json').read_text())
feats = manifest['features'] if isinstance(manifest, dict) and 'features' in manifest else \
    ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'NDVI']
tr, va, te = (tables[k] for k in ('train', 'validation', 'test'))
Xtr, ytr, Xva, yva, Xte = tr[feats].to_numpy(), tr.target.to_numpy(), va[feats].to_numpy(), va.target.to_numpy(), te[feats].to_numpy()
groups = json.loads((ROOT / 'data/validation_groups.json').read_text()); g = [groups[i] for i in va.sample_id]


class RBFControl:  # same preprocessing as the quantum model; tuned with the same trial budget
    def fit(self, X, y, seed=0):
        t0 = time.perf_counter(); hp = cfg['matched_rbf_control_by_input_dims']['8']
        self.prep = make_pipeline(StandardScaler(), PCA(8, svd_solver='full'), StandardScaler()).fit(X)
        t1 = time.perf_counter(); self.clf = SVC(kernel='rbf', gamma=hp['gamma'], C=hp['C']).fit(self.prep.transform(X), y)
        self.timing = dict(preprocessing_seconds=t1 - t0, fit_seconds=time.perf_counter() - t1); return self
    def predict(self, X):
        t0 = time.perf_counter(); p = self.clf.predict(self.prep.transform(X)); self.timing['predict_seconds'] = time.perf_counter() - t0; return p
    def resources(self):
        return dict(qubits=0, trainable_quantum_params=0, readout_trainable_params=int(self.clf.dual_coef_.size + self.clf.intercept_.size),
                    support_vectors=int(self.clf.n_support_.sum()))


class Reference:  # supplied RBF-SVM on all 13 raw features
    def fit(self, X, y, seed=0):
        t0 = time.perf_counter(); self.clf = reference().fit(X, y); self.timing = dict(preprocessing_seconds=0.0, fit_seconds=time.perf_counter() - t0); return self
    def predict(self, X):
        t0 = time.perf_counter(); p = self.clf.predict(X); self.timing['predict_seconds'] = time.perf_counter() - t0; return p
    def resources(self): return dict(qubits=0, trainable_quantum_params=0)


makers = {'quantum_kernel (final)': Submission, 'matched_rbf_control': RBFControl, 'reference_rbf_svm': Reference}
runs, preds_val = [], {}
for name, make in makers.items():
    for seed in (0, 1, 2):
        m = make().fit(Xtr, ytr, seed=seed); pv = m.predict(Xva); pt = m.predict(Xte)
        s = score(yva, pv, LABELS)
        runs.append(dict(model=name, seed=seed, **s, **m.timing, resources=m.resources()))
        if seed == 0:
            preds_val[name] = pv
            pd.DataFrame({'sample_id': te.sample_id, 'prediction': pt}).to_csv(OUT / f'test_predictions_{name.split()[0]}.csv', index=False)
            pd.DataFrame({'sample_id': va.sample_id, 'target': yva, 'prediction': pv}).to_csv(OUT / f'validation_predictions_{name.split()[0]}.csv', index=False)
        print(f'{name:24s} seed {seed}: F1={s["macro_f1"]:.4f} acc={s["accuracy"]:.4f} fit={m.timing["fit_seconds"]:.2f}s predict={m.timing["predict_seconds"]:.2f}s', flush=True)

boot = {k: paired_bootstrap(yva, preds_val['quantum_kernel (final)'], preds_val[k], g) for k in ('matched_rbf_control', 'reference_rbf_svm')}
(OUT / 'final_results.json').write_text(json.dumps(dict(runs=runs, bootstrap_quantum_minus=boot), indent=1, default=float))
L = ['| model | val macro-F1 (mean ± std, seeds 0/1/2) | val accuracy | fit s | predict s |', '|---|---|---|---|---|']
for name in makers:
    r = [x for x in runs if x['model'] == name]; f = [x['macro_f1'] for x in r]
    L.append(f'| {name} | {np.mean(f):.4f} ± {np.std(f):.4f} | {np.mean([x["accuracy"] for x in r]):.4f} | '
             f'{np.mean([x["fit_seconds"] for x in r]):.2f} | {np.mean([x["predict_seconds"] for x in r]):.2f} |')
L += ['', 'Per-class precision / recall (seed 0), classes = alfalfa, maize, wheat:']
for name in makers:
    x = [r for r in runs if r['model'] == name][0]
    L.append(f'- {name}: precision {np.round(x["precision"], 3).tolist()}, recall {np.round(x["recall"], 3).tolist()}, confusion {x["confusion"]}')
L += ['', 'Resources (seed 0):'] + [f'- {n}: {[r for r in runs if r["model"] == n][0]["resources"]}' for n in makers]
L += ['', 'Paired spatial-block bootstrap (quantum minus other), validation macro-F1:'] + \
     [f'- vs {k}: mean {b["mean"]:+.4f}, 95% CI [{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}]' for k, b in boot.items()]
(OUT / 'final_summary.md').write_text('\n'.join(L)); print('\n' + '\n'.join(L))
