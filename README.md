# CERN_3d_particle_gnn_tracking
 # Geometric Deep Learning for 3D Particle Track Reconstruction

An end-to-end Graph Neural Network (GNN) pipeline built with **PyTorch Geometric** and **Plotly** to reconstruct 3D subatomic particle trajectories inside the Large Hadron Collider (LHC) detector at CERN.

## 📌 Project Overview
At the LHC, millions of proton-proton collisions generate high-density 3D spatial points (hits). Traditional trajectory algorithms scale quadratically ($O(N^2)$) with track density. This project frames particle track reconstruction as a **Graph Edge Classification problem**:
- **Nodes:** 3D spatial coordinates $(x, y, z)$ recorded by silicon sensors.
- **Edges:** Candidate track segments generated using spatial $k$-Nearest Neighbors ($k$-NN).
- **Model:** Graph Convolutional Network (GCN) + Edge Classifier MLP to predict physical particle connection probabilities.

## 📊 Model Performance Metrics
- **ROC-AUC Score:** `{auc:.4f}`
- **Edge Precision:** `{precision:.4f}`
- **Edge Recall:** `{recall:.4f}`
- **F1-Score:** `{f1:.4f}`

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, PyTorch Geometric, scikit-learn
- **Visualization:** Plotly 3D (Interactive spatial rendering)
- **Data Source:** CERN TrackML Particle Tracking Dataset

## 📁 Repository Structure
├── src/
│   ├── data_prep.py     # Graph construction & k-NN spatial graph processing
│   ├── model.py         # PyTorch Geometric GCN Edge Classifier
│   ├── train.py         # GPU training loop with weighted BCE loss
│   └── evaluate.py      # Benchmark evaluation & 3D visualization scripts
├── models/
│   └── gnn_track_model.pth
├── particle_3d_tracks.html  # Interactive Plotly 3D Visualization
└── README.md
