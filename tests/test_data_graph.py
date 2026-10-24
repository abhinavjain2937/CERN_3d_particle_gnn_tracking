import numpy as np
import pytest

from trackml_edge.data import (event_id, find_event_files, load_event_table, split_events,
                               truth_path)
from trackml_edge.graph import (EDGE_FEATURES, NODE_FEATURES, Standardizer, build_raw_graph,
                                fit_standardizers, to_pyg)


def test_find_and_split_events(synthetic_root):
    files = find_event_files(synthetic_root)
    assert len(files) == 8 and files == sorted(files)
    assert all(truth_path(f).exists() for f in files)
    sp = split_events(files, 3, 2, 3)
    names = [p.name for k in ("train", "val", "test") for p in sp[k]]
    assert len(names) == len(set(names)) == 8          # disjoint and exhaustive here
    with pytest.raises(ValueError):
        split_events(files, 5, 2, 5)                    # not enough events


def test_r_cut_applied(synthetic_root):
    f = find_event_files(synthetic_root)[0]
    full = load_event_table(f, r_max=None)
    cut = load_event_table(f, r_max=200.0)
    assert len(cut) < len(full)
    assert (np.hypot(cut.x, cut.y) < 200.0).all()


def test_graph_shapes_and_labels(synthetic_root):
    f = find_event_files(synthetic_root)[0]
    df = load_event_table(f, 200.0)
    k = 5
    g = build_raw_graph(df, k, event_id(f))
    n = len(df)
    assert g.node_raw.shape == (n, len(NODE_FEATURES))
    assert g.edge_index.shape == (2, n * k) and g.edge_raw.shape == (n * k, len(EDGE_FEATURES))
    assert not (g.edge_index[0] == g.edge_index[1]).any()      # no self loops
    # labels agree with the truth particle ids, and noise (id 0) is never positive
    pid = df.particle_id.to_numpy()
    expected = (pid[g.edge_index[0]] == pid[g.edge_index[1]]) & (pid[g.edge_index[0]] != 0)
    assert np.array_equal(g.label.astype(bool), expected)
    assert 0 < g.label.sum() < len(g.label)


def test_scalers_fit_on_training_only(synthetic_root):
    files = find_event_files(synthetic_root)
    raws = [build_raw_graph(load_event_table(f, 200.0), 5, event_id(f)) for f in files[:2]]
    other = build_raw_graph(load_event_table(files[5], 200.0), 5, event_id(files[5]))
    node_std, edge_std = fit_standardizers(raws)
    train_x = to_pyg(raws[0], node_std, edge_std).x.numpy()
    assert abs(np.vstack([node_std.transform(r.node_raw) for r in raws]).mean()) < 1e-6
    # an unseen event is transformed with the TRAINING statistics (not re-fitted)
    unseen = to_pyg(other, node_std, edge_std).x.numpy()
    assert np.allclose(unseen, (other.node_raw - node_std.mean) / node_std.scale, atol=1e-5)
    assert train_x.shape[1] == len(NODE_FEATURES)


def test_standardizer_roundtrip():
    x = np.random.default_rng(0).normal(3, 2, size=(50, 4))
    s = Standardizer().fit(x)
    s2 = Standardizer.from_state(s.state())
    assert np.allclose(s.transform(x), s2.transform(x))
