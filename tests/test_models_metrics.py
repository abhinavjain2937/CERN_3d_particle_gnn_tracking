import numpy as np
import torch

from trackml_edge.evaluate import choose_threshold, edge_metrics, safe_roc_auc
from trackml_edge.models import GCNEdgeClassifier, build_model


def _toy(n=30, e=90):
    x = torch.randn(n, 6)
    ei = torch.randint(0, n, (2, e))
    ea = torch.randn(e, 6)
    return x, ei, ea


def test_model_output_shapes_and_registry():
    x, ei, ea = _toy()
    for name in ("edge_mlp", "gcn"):
        m = build_model(name, hidden=16)
        out = m(x, ei, ea)
        assert out.shape == (ei.shape[1],)
    assert isinstance(build_model("gcn"), GCNEdgeClassifier)


def test_gcn_uses_neighbourhood_but_mlp_does_not():
    x, ei, ea = _toy()
    mlp, gcn = build_model("edge_mlp", 16), build_model("gcn", 16)
    mlp.eval(); gcn.eval()
    x2 = x.clone()
    # perturb a node that is NOT an endpoint of edge 0 but is a neighbour of its source
    src0, dst0 = ei[0, 0].item(), ei[1, 0].item()
    other = next(i for i in range(30) if i not in (src0, dst0))
    ei2 = ei.clone(); ei2[0, 1] = other; ei2[1, 1] = src0          # edge other -> src0
    x2[other] += 5.0
    a = mlp(x, ei2, ea)[0].item(); b = mlp(x2, ei2, ea)[0].item()
    assert abs(a - b) < 1e-6                                       # MLP: edge 0 unaffected
    c = gcn(x, ei2, ea)[0].item(); d = gcn(x2, ei2, ea)[0].item()
    assert abs(c - d) > 1e-6                                       # GCN: neighbour information flows in


def test_metrics_values():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.4, 0.6, 0.9])
    m = edge_metrics(y, p, 0.5)
    assert m["roc_auc"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0
    assert np.isnan(safe_roc_auc(np.zeros(4), p))


def test_threshold_choice_maximises_f1():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    thr = choose_threshold(y, p)
    assert 0.4 <= thr <= 0.6
