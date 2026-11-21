| model | val macro-F1 (mean ± std, seeds 0/1/2) | val accuracy | fit s | predict s |
|---|---|---|---|---|
| quantum_kernel (final) | 0.9201 ± 0.0000 | 0.9200 | 1.79 | 0.38 |
| matched_rbf_control | 0.9326 ± 0.0000 | 0.9333 | 0.02 | 0.00 |
| reference_rbf_svm | 0.8797 ± 0.0000 | 0.8800 | 0.02 | 0.00 |

Per-class precision / recall (seed 0), classes = alfalfa, maize, wheat:
- quantum_kernel (final): precision [0.94, 0.939, 0.882], recall [0.94, 0.92, 0.9], confusion [[47, 0, 3], [1, 46, 3], [2, 3, 45]]
- matched_rbf_control: precision [0.941, 0.925, 0.935], recall [0.96, 0.98, 0.86], confusion [[48, 0, 2], [0, 49, 1], [3, 4, 43]]
- reference_rbf_svm: precision [0.887, 0.915, 0.84], recall [0.94, 0.86, 0.84], confusion [[47, 0, 3], [2, 43, 5], [4, 4, 42]]

Resources (seed 0):
- quantum_kernel (final): {'qubits': 8, 'layers': 1, 'depth': 23, 'two_qubit_gates': 14, 'trainable_quantum_params': 0, 'frozen_quantum_params': 1, 'readout_trainable_params': 445, 'support_vectors': 221, 'statevector_simulations': 1135, 'pairwise_fidelity_circuits_equiv': 619235}
- matched_rbf_control: {'qubits': 0, 'trainable_quantum_params': 0, 'readout_trainable_params': 405, 'support_vectors': 201}
- reference_rbf_svm: {'qubits': 0, 'trainable_quantum_params': 0}

Paired spatial-block bootstrap (quantum minus other), validation macro-F1:
- vs matched_rbf_control: mean -0.0118, 95% CI [-0.0466, +0.0277]
- vs reference_rbf_svm: mean +0.0422, 95% CI [-0.0006, +0.0953]