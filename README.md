# DATA 37711 — Homework 2

Solutions to Homework 2 for DATA 37711. The work covers three problems: probabilistic regression with a Gaussian MLP, high-cardinality categorical features, and Random Kitchen Sinks compared with a one-hidden-layer MLP.

All code, outputs, and written answers are in [`hw2.ipynb`](hw2.ipynb). The handwritten math derivations are in [`hw2.pdf`](hw2.pdf).

## Repository layout

| File | Description |
| --- | --- |
| `hw2.ipynb` | Main notebook: code, results, and discussion for all three problems |
| `hw2.pdf` | Handwritten mathematical derivations |
| `models.py` | `RandomKitchenSinks` (random Fourier features + closed-form ridge), `MLP` (1 hidden layer, PyTorch), and the hyper-parameter sweep helpers `sweep_rks_model` / `mlp_sweep` |
| `dataloader.py` | `HW2Dataset.from_csv`, which loads the Problem 3 train/val/test splits |
| `plotting.py` | `plot_surface_2d`, which plots the fitted surface of any `.predict` callable or `nn.Module` over a 2-D grid |
| `taxi_trips.parquet` | NYC taxi trips (523,554 rows, 7 features, target `trip_duration`), used in Problem 1 |
| `retail.csv` | Online-retail transactions (`Quantity`, `StockCode`, `CustomerID`; 397,924 rows), used in Problem 2 |
| `train.csv`, `val.csv`, `test.csv` | 2-D synthetic regression data (`x1`, `x2`, `y`) for Problem 3. The test labels are held out |
| `rks_test_predictions.csv`, `mlp_test_predictions.csv` | Final Problem 3 test-set predictions |

## Running

Python 3.10+ is required because `plotting.py` uses `X | Y` type hints. Install the dependencies:

```bash
pip install numpy pandas scipy scikit-learn torch matplotlib pyarrow jupyter
```

Then open the notebook:

```bash
jupyter notebook hw2.ipynb
```

Run the notebook from the repository root so the data files resolve. All randomness is seeded with `SEED = 0`. Generated `.png` figures are git-ignored.
