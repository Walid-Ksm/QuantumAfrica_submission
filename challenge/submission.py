"""Final model: fidelity quantum-kernel SVM (see configs/final_model.json for the frozen settings).

Input shape
  * 13 columns (raw features B1..B12, NDVI; default): StandardScaler -> PCA(8) -> StandardScaler is fitted on the
    TRAINING rows inside fit(), then an 8-qubit kernel is used.
  * 8 or 4 columns: treated as the supplied training-fitted PCA views (no further preprocessing); 8 or 4 qubits.
Interface: fit(X_train, y_train, seed=0), predict(X), resources(). The model is deterministic (the seed is unused).
"""
import json, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from qiskit import transpile
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'final_model.json'


def _feature_map(n, reps):
    return zz_feature_map(n, reps=reps, entanglement='linear')


def _statevectors(fmap, X):
    params = list(fmap.parameters)
    return np.array([Statevector.from_instruction(fmap.assign_parameters(dict(zip(params, x)))).data for x in X])


class Submission:
    def __init__(self, config_path=CONFIG):
        self.config = json.loads(Path(config_path).read_text())

    def _prepare(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        if fit:
            if X.shape[1] == 13:
                self.prep = make_pipeline(StandardScaler(), PCA(8, svd_solver='full'), StandardScaler()).fit(X)
            elif X.shape[1] in (4, 8):
                self.prep = None
            else:
                raise ValueError('Expected 13 raw features or 4/8 PCA features')
            self.dims = 8 if X.shape[1] == 13 else X.shape[1]
        return self.prep.transform(X) if self.prep is not None else X

    def fit(self, X_train, y_train, seed=0):
        t0 = time.perf_counter()
        Z = self._prepare(X_train, fit=True)
        self.hp = self.config['quantum_by_input_dims'][str(self.dims)]
        self.fmap = _feature_map(self.dims, self.hp['reps'])
        t1 = time.perf_counter()
        self.sv_train = _statevectors(self.fmap, self.hp['bw'] * Z)
        K = np.abs(self.sv_train.conj() @ self.sv_train.T) ** 2
        self.clf = SVC(kernel='precomputed', C=self.hp['C']).fit(K, np.asarray(y_train))
        self.n_train = len(Z)
        self.timing = dict(preprocessing_seconds=t1 - t0, fit_seconds=time.perf_counter() - t1)
        return self

    def predict(self, X):
        t0 = time.perf_counter()
        sv = _statevectors(self.fmap, self.hp['bw'] * self._prepare(X))
        pred = self.clf.predict(np.abs(sv.conj() @ self.sv_train.T) ** 2)
        self.timing['predict_seconds'] = time.perf_counter() - t0
        self.n_eval = len(sv)
        return pred

    def resources(self):
        c = transpile(self.fmap, basis_gates=['h', 'p', 'cx', 'rx', 'ry', 'rz'], optimization_level=0)
        n = self.n_train
        return dict(qubits=self.dims, layers=self.hp['reps'], depth=c.depth(), two_qubit_gates=int(c.num_nonlocal_gates()),
                    trainable_quantum_params=0, frozen_quantum_params=1,  # frozen: input bandwidth
                    readout_trainable_params=int(self.clf.dual_coef_.size + self.clf.intercept_.size),
                    support_vectors=int(self.clf.n_support_.sum()),
                    statevector_simulations=n + getattr(self, 'n_eval', 0),
                    pairwise_fidelity_circuits_equiv=n * (n - 1) // 2 + getattr(self, 'n_eval', 0) * n)
