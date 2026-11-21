# Submission: quantum-kernel SVM vs matched classical control (Moroccan Crop Classification)

**Question.** Does a fidelity quantum-kernel SVM add anything beyond a classical RBF-SVM on the same inputs and preprocessing?

**Final model** (`challenge/submission.py`, frozen settings in `configs/final_model.json`): 13 raw features -> StandardScaler -> PCA(8) -> StandardScaler (fitted on training rows only) -> 8-qubit ZZ feature map, 1 layer, input bandwidth 0.02 -> exact fidelity kernel -> SVM (C=100). Interface: `fit(X_train, y_train, seed)`, `predict(X)`.

**Honest scope.** All quantum computation is exact classical statevector simulation (no noise, no IBM hardware result reported). Validation selected hyperparameters (144 trials per model, rule fixed before any test prediction) and scores all comparisons. The test set has no labels: only test predictions are produced, no test score.

## How to run (Python 3.12)
```
python -m venv .venv
.venv\Scripts\activate.bat          (Windows cmd)   |   source .venv/bin/activate  (Linux/macOS)
python -m pip install -r requirements.txt
python experiments/final_evaluation.py           # final model, matched control, reference; ~1 min
python experiments/kernel_vs_control.py          # full sweep (layers, bandwidth, 25/50/100% data); ~5-10 min
```
Or open `notebooks/Final_Submission.ipynb` (kernel "Crop challenge") and Run All (~5-15 min; it reruns the sweep if `outputs/kernel_vs_control/` is absent).

## Files
| Path | Role |
|---|---|
| `challenge/submission.py` | Final model |
| `configs/final_model.json` | Frozen hyperparameters, selection rule, control settings |
| `experiments/kernel_vs_control.py` | Controlled sweep: ZZ kernel, product-state kernel (ablation), RBF control; same 144-trial budget; 25/50/100% data; paired spatial-block bootstrap |
| `experiments/final_evaluation.py` | Final model vs control vs reference, seeds 0/1/2, all metrics and resources, test predictions |
| `notebooks/Final_Submission.ipynb` | Narrative, plots, findings, limitations |
| `outputs/final/test_predictions_quantum_kernel.csv` | Test predictions (221 rows: `sample_id`, `prediction`) |
| `QuantumAfrica_submission_presentation.pdf` | Final presentation deck (PDF) |
| `presentation.pptx` | Final presentation deck (PowerPoint) |

## Key results (validation, macro-F1)
Quantum kernel 0.920 | matched RBF control 0.933 | supplied reference 0.880. Quantum minus control: -0.012, 95% CI [-0.047, +0.028] (no measurable quantum contribution). Resources of the final model: 8 qubits, depth 23, 14 two-qubit gates, 0 trainable quantum parameters, 1 frozen quantum parameter (bandwidth), 445 trainable readout parameters, 1135 statevector simulations.

## References
Havlicek et al., Nature 567, 209 (2019); Shaydulin and Wild, Phys. Rev. A 106, 042407 (2022); Huang et al., Nat. Commun. 12, 2631 (2021); Qiskit 2.4.2; scikit-learn; supplied challenge repository helpers (data checks, metrics, reference model). Dataset: Ouassanouan et al., Scientific Data 13, 746 (2026), doi:10.1038/s41597-026-06993-y (CC BY-NC-ND 4.0).
