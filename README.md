# TrackML Hit-Graph Edge Classifier
 
Edge classification on **simulated** particle-tracking data from the public
[TrackML challenge](https://www.kaggle.com/competitions/trackml-particle-identification), built with PyTorch Geometric.
 
For each simulated collision event, the pipeline builds a k-nearest-neighbour (k-NN) graph over the detector hits and
trains a model to predict, for every candidate edge, whether its two hits came from the **same particle**. This
hit-pairing step is the first stage of track finding.
 
> Independent learning project using public data. It is **not** affiliated with, endorsed by, or an output of CERN.
 
**Scope:** edge classification only. The repository does **not** assemble tracks and does **not** compute the official
TrackML score, so the numbers below cannot be compared with challenge leaderboard results.
 
## Results
 
Held-out test events (never used for training, threshold or model selection):
 
<!-- RESULTS:START -->
| Item | Value |
|---|---|
| Events (train / validation / test) | 5 / 2 / 5 (12 total) |
| Graph size per event | about 59,000 hits, 294,900 edges |
| True same-particle edges | 12.5% |
| Selected model | `gcn` |
| Decision threshold (chosen on validation) | 0.85 |
| ROC-AUC | 0.997 (std < 0.001) |
| Precision | 0.937 |
| Recall | 0.976 |
| F1 | 0.956 |
<!-- RESULTS:END -->
 
Full numbers, curves and the confusion matrix are written to `runs/default/metrics.md` by each run.
 
## How it works
 
1. **Load:** read `event*-hits.csv` and `event*-truth.csv`, keep the inner barrel (r < 200 mm), join the true particle ids.
2. **Build graphs:** k-NN graph over hits.
   - Node features: `x, y, z, r, φ, η`
   - Edge features: `dx, dy, dz, dr, dφ, dist`
   - Label: do the two hits share a particle?
3. **Split by event** into disjoint train / validation / test sets. Fit z-score scalers on training events only.
4. **Train two models** with class-weighted BCE, AdamW and a cosine learning-rate schedule. Keep the epoch with the best validation ROC-AUC.
5. **Select on validation:** the decision threshold (max F1) and the final model.
6. **Report** ROC-AUC, precision, recall and F1 on the unseen test events.
| Model | Description |
|---|---|
| `edge_mlp` | Encodes each hit and edge, then classifies `[source, target, edge]`. No message passing, so it is an MLP, not a GNN. |
| `gcn` | Same classifier head, but hit embeddings first pass through two `GCNConv` layers (real message passing). |
 
## Quick start
 
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# install PyTorch for your platform first: https://pytorch.org/get-started/locally/
pip install -e ".[dev]"
```
 
Download `train_sample` from the Kaggle
[TrackML data page](https://www.kaggle.com/competitions/trackml-particle-identification/data) and point `--data-root`
at the extracted folder or the `.zip`. The data is not included here and must not be committed (it is git-ignored);
its use is governed by the competition rules.
 
```bash
# full run: train both models, select, evaluate on held-out events
trackml-edge run --data-root path/to/train_sample --out-dir runs/default
 
# faster run
trackml-edge run --data-root path/to/train_sample --n-train 2 --n-val 1 --n-test 3 --epochs 20
 
# re-evaluate the saved model on its recorded test events (reproducibility check)
trackml-edge evaluate --run runs/default --data-root path/to/train_sample
 
# 3D plot of a few truth trajectories
trackml-edge plot-tracks --data-root path/to/train_sample --out docs/tracks_3d.png
 
# try the code without the dataset (synthetic data, metrics are meaningless)
trackml-edge make-synthetic --out data/synthetic
trackml-edge run --data-root data/synthetic --out-dir runs/synthetic-demo --epochs 10
```
 
Each run saves its config, splits, metrics, model weights, scalers, plots and log in `--out-dir`.
Runs are seeded, and the tests check that two runs with the same seed give identical metrics.
 
To refresh the Results table above from a run: `python scripts/update_readme.py runs/default`
 
## Repository layout
 
```
configs/default.yaml        hyper-parameters and train/val/test event counts
src/trackml_edge/
    data.py                 find events, split, read one event
    graph.py                k-NN graph, features, scalers, PyG conversion
    models.py               the two classifiers
    train.py                training loop, validation-based checkpointing
    evaluate.py             prediction, threshold selection, metrics
    pipeline.py             end-to-end run and re-evaluation
    report.py, visualize.py results table and figures
    synthetic.py            fake TrackML-format events for smoke tests
    cli.py                  `trackml-edge` command
scripts/                    update_readme.py, export_resume_macros.py
tests/                      pytest suite (runs on synthetic data, no download needed)
notebooks/                  earlier single-file exploratory notebook
```
 
## Design decisions
 
- **Split by event:** no hit or particle appears in more than one split.
- **No preprocessing leakage:** scalers are fitted on training events only.
- **Validation drives all choices:** epoch, threshold and final model are picked on validation events. Test events are only scored.
- **Class imbalance:** most candidate edges are false, so the loss uses a capped positive-class weight, and precision and recall are reported next to ROC-AUC.
## Limitations
 
- Edge classification only: no track assembly and no official TrackML score.
- Inner barrel only (`r_max`, default 200 mm); other hits are ignored.
- The graph comes from a fixed k-NN rule, so true neighbours outside it can never be recovered.
- Only 12 simulated events were used. Results say nothing about real detector performance.
- More events and longer training are the obvious next step.
## Tests
 
```bash
pytest -q
```
 
## Reference
 
S. Amrouche et al., *The Tracking Machine Learning Challenge: Accuracy Phase*, arXiv:1904.06778 (2019).
 
## License
 
MIT (see `LICENSE`). The TrackML data is not covered by this license.
 