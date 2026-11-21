"""Quantum fidelity-kernel SVM vs matched classical controls (validation-only).

Question: does the quantum component add anything beyond a matched classical component?
  quantum_zz   : entangling ZZ feature-map fidelity kernel -> SVM        (tuned: layers, bandwidth, C)
  product_z    : NON-entangling Z feature-map kernel -> SVM (ablation)   (same tuning grid)
  rbf_control  : RBF-SVM on the SAME PCA features                        (same number of trials)
All inputs are the supplied training-fitted PCA views. Selection uses VALIDATION only.
Run from the repository root:  python experiments/kernel_vs_control.py
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.svm import SVC
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit.quantum_info import Statevector
from challenge.data import load_data
from challenge.metrics import score

LABELS = ('alfalfa', 'maize', 'wheat')


def load_view(n, tables):
    with np.load(ROOT / f'data/views/pca{n}.npz', allow_pickle=False) as f:
        v = {k: f[k] for k in f.files}
    for s in ('train', 'validation', 'test'):
        np.testing.assert_array_equal(v['ids_' + s], tables[s].sample_id)
    return v


def feature_map(kind, n, reps):
    return zz_feature_map(n, reps=reps, entanglement='linear') if kind == 'quantum_zz' else z_feature_map(n, reps=reps)


def statevectors(fmap, X):
    params = list(fmap.parameters)
    return np.array([Statevector.from_instruction(fmap.assign_parameters(dict(zip(params, x)))).data for x in X])


def build_kernels(kind, n, reps, bw, X):
    """Exact fidelity kernel |<psi(x)|psi(x')>|^2 from statevectors (cached once per config)."""
    fm = feature_map(kind, n, reps)
    t0 = time.perf_counter()
    S = {k: statevectors(fm, bw * X[k]) for k in X}
    t_sv = time.perf_counter() - t0
    t0 = time.perf_counter()
    K = {'train': np.abs(S['train'].conj() @ S['train'].T) ** 2}
    for k in ('validation', 'test'):
        K[k] = np.abs(S[k].conj() @ S['train'].T) ** 2
    t_gram = time.perf_counter() - t0
    return K, dict(seconds_statevectors=t_sv, seconds_gram=t_gram)


def circuit_resources(kind, n, reps):
    c = transpile(feature_map(kind, n, reps), basis_gates=['h', 'p', 'cx', 'rx', 'ry', 'rz'], optimization_level=0)
    return dict(qubits=n, depth=c.depth(), two_qubit_gates=int(c.num_nonlocal_gates()), layers=reps)


def stratified_subset(y, frac, seed):
    if frac >= 1: return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = [rng.permutation(np.flatnonzero(y == c))[:max(1, round(frac * (y == c).sum()))] for c in LABELS]
    return np.sort(np.concatenate(idx))


def f1_of(y_true, y_pred):
    return score(y_true, y_pred, list(LABELS))['macro_f1']


def paired_bootstrap(y, pa, pb, groups, B=2000, seed=0):
    """Resample spatial blocks; return macro-F1(A)-macro-F1(B) mean and 95% percentile CI."""
    rng = np.random.default_rng(seed); g = np.asarray(groups); ug = np.unique(g)
    members = [np.flatnonzero(g == u) for u in ug]; d = []
    for _ in range(B):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        d.append(f1_of(y[idx], pa[idx]) - f1_of(y[idx], pb[idx]))
    return dict(mean=float(np.mean(d)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                prob_a_better=float(np.mean(np.array(d) > 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--components', type=int, nargs='+', default=[4, 8])
    ap.add_argument('--kinds', nargs='+', default=['quantum_zz', 'product_z'])
    ap.add_argument('--reps', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--bw', type=float, nargs='+', default=[0.01, 0.02, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0])
    ap.add_argument('--C', type=float, nargs='+', default=[0.3, 1, 3, 10, 30, 100])
    ap.add_argument('--fractions', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    ap.add_argument('--class-weight', default='none', choices=['none', 'balanced'])
    ap.add_argument('--out', default='outputs/kernel_vs_control')
    a = ap.parse_args()
    cw = None if a.class_weight == 'none' else 'balanced'
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    tables, _, _, _ = load_data()
    groups_map = json.loads((ROOT / 'data/validation_groups.json').read_text())
    val_groups = [groups_map[i] for i in tables['validation'].sample_id]
    n_trials = len(a.reps) * len(a.bw) * len(a.C)
    gammas = np.logspace(-4, 0, len(a.reps) * len(a.bw))      # control gets the SAME number of trials
    runs, full_preds = [], {}
    t_all = time.perf_counter()

    for n in a.components:
        v = load_view(n, tables)
        X = {'train': v['X_train'], 'validation': v['X_validation'], 'test': v['X_test']}
        ytr, yva = v['y_train'], v['y_validation']
        models = {}
        # ---- quantum kernels: cache every (layers, bandwidth) kernel once --------------------
        for kind in a.kinds:
            cache, grid_sec = {}, 0.0
            for reps in a.reps:
                for bw in a.bw:
                    K, tm = build_kernels(kind, n, reps, bw, X)
                    cache[(reps, bw)] = (K, tm); grid_sec += tm['seconds_statevectors'] + tm['seconds_gram']
            cands = [dict(reps=r, bw=b, C=C) for r in a.reps for b in a.bw for C in a.C]
            models[kind] = (cands, cache, grid_sec)
        cands = [dict(gamma=float(g), C=C) for g in gammas for C in a.C]
        models['rbf_control'] = (cands, None, 0.0)
        print(f'[pca{n}] kernels cached; trials per model = {n_trials}', flush=True)

        for kind, (cands, cache, grid_sec) in models.items():
            for frac in a.fractions:
                seeds = [0, 1, 2] if frac < 1 else [0]       # full training set is deterministic
                for seed in seeds:
                    idx = stratified_subset(ytr, frac, seed); ys = ytr[idx]
                    t_tune = time.perf_counter(); best, best_f1 = None, -1
                    for c in cands:
                        if kind == 'rbf_control':
                            clf = SVC(kernel='rbf', gamma=c['gamma'], C=c['C'], class_weight=cw).fit(X['train'][idx], ys)
                            pred = clf.predict(X['validation'])
                        else:
                            K = cache[(c['reps'], c['bw'])][0]
                            clf = SVC(kernel='precomputed', C=c['C'], class_weight=cw).fit(K['train'][np.ix_(idx, idx)], ys)
                            pred = clf.predict(K['validation'][:, idx])
                        f = f1_of(yva, pred)
                        if f > best_f1: best, best_f1 = c, f
                    tune_sec = time.perf_counter() - t_tune
                    # ---- refit the selected config on the training subset and time it -----------
                    t0 = time.perf_counter()
                    if kind == 'rbf_control':
                        clf = SVC(kernel='rbf', gamma=best['gamma'], C=best['C'], class_weight=cw).fit(X['train'][idx], ys)
                        fit_s = time.perf_counter() - t0; t0 = time.perf_counter()
                        pred = clf.predict(X['validation']); pred_s = time.perf_counter() - t0
                        res = dict(qubits=0, trainable_quantum_params=0, frozen_quantum_params=0)
                    else:
                        K, tm = cache[(best['reps'], best['bw'])]
                        clf = SVC(kernel='precomputed', C=best['C'], class_weight=cw).fit(K['train'][np.ix_(idx, idx)], ys)
                        fit_s = time.perf_counter() - t0
                        t0 = time.perf_counter(); pred = clf.predict(K['validation'][:, idx]); pred_s = time.perf_counter() - t0
                        res = circuit_resources(kind, n, best['reps'])
                        res.update(trainable_quantum_params=0, frozen_quantum_params=1,   # frozen: bandwidth
                                   statevector_simulations=len(idx) + len(yva),
                                   pairwise_fidelity_circuits_equiv=len(idx) * (len(idx) - 1) // 2 + len(yva) * len(idx),
                                   kernel_build_seconds=tm['seconds_statevectors'] + tm['seconds_gram'])
                    m = score(yva, pred, list(LABELS))
                    rec = dict(components=n, model=kind, fraction=frac, seed=seed, n_train=int(len(idx)),
                               deterministic=frac >= 1, chosen=best, trials=len(cands), tuning_seconds=tune_sec,
                               grid_kernel_seconds=grid_sec, fit_seconds=fit_s, predict_seconds=pred_s,
                               n_support_vectors=int(clf.n_support_.sum()), class_weight=a.class_weight, resources=res, **m)
                    runs.append(rec)
                    if frac >= 1: full_preds[(n, kind)] = pred
                    print(f'  pca{n} {kind:11s} frac={frac:<4} seed={seed} F1={m["macro_f1"]:.4f} acc={m["accuracy"]:.4f} '
                          f'best={best} fit={fit_s:.2f}s', flush=True)

    # ---- paired spatial-block bootstrap (full training set): each quantum model minus its control ----
    boots = {}
    for n in a.components:
        for kind in a.kinds:
            if (n, kind) in full_preds and (n, 'rbf_control') in full_preds:
                boots[f'pca{n}:{kind}-minus-rbf_control'] = paired_bootstrap(
                    np.asarray(load_view(n, tables)['y_validation']), full_preds[(n, kind)], full_preds[(n, 'rbf_control')], val_groups)

    (out / 'results.json').write_text(json.dumps(dict(config=vars(a), runs=runs, bootstrap=boots,
                                                       total_seconds=time.perf_counter() - t_all), indent=1, default=float))
    # ---- summary table (mean +- std over seeds; deterministic rows reuse one result) -----------------
    lines = ['| comps | model | train frac | macro-F1 (mean ± std) | acc (mean) | fit s | chosen (seed 0) |', '|---|---|---|---|---|---|---|']
    keys = sorted({(r['components'], r['model'], r['fraction']) for r in runs})
    for n, kind, frac in keys:
        rr = [r for r in runs if (r['components'], r['model'], r['fraction']) == (n, kind, frac)]
        f1 = np.array([r['macro_f1'] for r in rr]); ac = np.mean([r['accuracy'] for r in rr])
        lines.append(f'| {n} | {kind} | {frac} | {f1.mean():.4f} ± {f1.std():.4f}{" (det.)" if rr[0]["deterministic"] else ""} | '
                     f'{ac:.4f} | {np.mean([r["fit_seconds"] for r in rr]):.2f} | {rr[0]["chosen"]} |')
    lines += ['', '**Paired spatial-block bootstrap, full training, validation (quantum minus control, macro-F1):**', '']
    for k, b in boots.items():
        lines.append(f'- {k}: mean {b["mean"]:+.4f}, 95% CI [{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}]')
    (out / 'summary.md').write_text('\n'.join(lines))
    print('\n' + '\n'.join(lines)); print(f'\nTotal {time.perf_counter() - t_all:.0f}s. Saved to {out}')


if __name__ == '__main__':
    main()
