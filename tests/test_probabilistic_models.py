import networkx as nx
import numpy as np
import pytest
from sklearn.base import clone

from src.data.loader import FEATURE_COLUMNS, get_xy, load_cleveland
from src.evaluation.splits import outer_folds
from src.features.preprocessing import build_discrete_preprocessor
from src.models.bayesian_network import (EDGES, NODES, TARGET_NODE, BayesianNetworkClassifier,
                                         build_structure, edge_table)
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.pipelines import build_bn_pipeline, build_nb_pipeline


@pytest.fixture(scope="module")
def data():
    X, y = get_xy(load_cleveland())
    return X, y


@pytest.fixture(scope="module")
def coded(data):
    X, y = data
    folds = outer_folds(y)
    tr, va = folds[0]
    pre = build_discrete_preprocessor().fit(X.iloc[tr])
    return pre.transform(X.iloc[tr]), y.iloc[tr], pre.transform(X.iloc[va]), y.iloc[va]


@pytest.fixture(scope="module")
def bn(coded):
    Xtr, ytr, *_ = coded
    return BayesianNetworkClassifier().fit(Xtr, ytr)


# ---- graph ----
def test_graph_is_valid_dag_with_expected_nodes():
    m = build_structure()
    g = nx.DiGraph(m.edges())
    assert nx.is_directed_acyclic_graph(g)
    assert sorted(m.nodes()) == sorted(NODES) and len(NODES) == 14
    assert len(m.edges()) == len(EDGES) == 16
    assert len({(a, b) for a, b, *_ in EDGES}) == len(EDGES)
    assert sorted(m.get_parents(TARGET_NODE)) == ["age", "chol", "sex"]
    assert not any(c == TARGET_NODE and p in FEATURE_COLUMNS and p not in ("age", "sex", "chol") for p, c, *_ in EDGES)


def test_every_edge_documented():
    t = edge_table()
    assert t["rationale"].str.len().min() > 10
    assert set(t["kind"]) <= {"domain", "weak", "modelling"}


# ---- BN fitting and probabilities ----
def test_cpds_normalised_and_model_valid(bn):
    assert bn.model_.check_model()
    for cpd in bn.model_.get_cpds():
        vals = cpd.get_values()
        assert np.allclose(vals.sum(axis=0), 1.0)
        assert (vals > 0).all()  # Dirichlet smoothing: no zero cells


def test_bn_class_order_shape_normalisation(bn, coded):
    *_, Xva, yva = coded
    p = bn.predict_proba(Xva)
    assert list(bn.classes_) == [0, 1]
    assert p.shape == (len(Xva), 2)
    assert np.allclose(p.sum(axis=1), 1) and ((p >= 0) & (p <= 1)).all()
    assert set(bn.predict(Xva)) <= {0, 1}


def test_bn_matches_brute_force_enumeration(bn, coded):
    *_, Xva, _ = coded
    cpds = {c.variable: c for c in bn.model_.get_cpds()}
    parents = {n: bn.model_.get_parents(n) for n in NODES}
    for i in range(5):
        row = {c: int(Xva.iloc[i][c]) for c in FEATURE_COLUMNS}
        joint = []
        for t in (0, 1):
            full = {**row, TARGET_NODE: t}
            pr = 1.0
            for n in NODES:
                pr *= cpds[n].get_value(**{n: full[n], **{p: full[p] for p in parents[n]}})
            joint.append(pr)
        expected = np.array(joint) / sum(joint)
        assert np.allclose(bn.predict_proba(Xva.iloc[[i]])[0], expected, atol=1e-9)


def test_bn_fit_uses_only_given_data(coded):
    Xtr, ytr, *_ = coded
    a = BayesianNetworkClassifier().fit(Xtr, ytr)
    b = BayesianNetworkClassifier().fit(Xtr.iloc[:150], ytr.iloc[:150])
    pa = a.model_.get_cpds(TARGET_NODE).get_values()
    pb = b.model_.get_cpds(TARGET_NODE).get_values()
    assert not np.allclose(pa, pb)


# ---- BN error handling ----
def test_bn_missing_default_raises(bn, coded):
    *_, Xva, _ = coded
    row = Xva.iloc[[0]].astype(float); row.iloc[0, FEATURE_COLUMNS.index("ca")] = np.nan
    with pytest.raises(ValueError, match="Missing value in 'ca'"):
        bn.predict_proba(row)


def test_bn_marginalize_equals_dropping_evidence(coded):
    Xtr, ytr, Xva, _ = coded
    m = BayesianNetworkClassifier(missing="marginalize").fit(Xtr, ytr)
    row = Xva.iloc[[0]].astype(float); row.iloc[0, FEATURE_COLUMNS.index("thal")] = np.nan
    p = m.predict_proba(row)
    ev = {c: int(row.iloc[0][c]) for c in FEATURE_COLUMNS if c != "thal"}
    q = m._infer.query([TARGET_NODE], evidence=ev, show_progress=False)
    assert np.allclose(p[0], [q.get_value(**{TARGET_NODE: 0}), q.get_value(**{TARGET_NODE: 1})])
    assert np.allclose(p.sum(), 1)


def test_bn_unseen_state_raises_not_guessed(bn, coded):
    *_, Xva, _ = coded
    row = Xva.iloc[[0]].copy(); row["cp"] = 17
    with pytest.raises(ValueError, match="Unseen state"):
        bn.predict_proba(row)


def test_bn_input_validation(coded, bn):
    Xtr, ytr, Xva, _ = coded
    with pytest.raises(ValueError, match="must not be passed"):
        bn.predict_proba(Xva.assign(target=0))
    with pytest.raises(ValueError, match="Missing predictor"):
        bn.predict_proba(Xva.drop(columns=["age"]))
    with pytest.raises(TypeError):
        bn.predict_proba(Xva.to_numpy())
    with pytest.raises(ValueError, match="binary"):
        BayesianNetworkClassifier().fit(Xtr, ytr + 2)
    with pytest.raises(ValueError, match="pseudo_count"):
        BayesianNetworkClassifier(pseudo_count=0).fit(Xtr, ytr)
    with pytest.raises(ValueError, match="missing must be"):
        BayesianNetworkClassifier(missing="ignore").fit(Xtr, ytr)
    with pytest.raises(RuntimeError, match="not fitted"):
        BayesianNetworkClassifier().predict_proba(Xva)


# ---- Naive Bayes ----
def test_nb_probabilities_and_class_order(coded):
    Xtr, ytr, Xva, yva = coded
    nb = NaiveBayesClassifier().fit(Xtr, ytr)
    p = nb.predict_proba(Xva)
    assert list(nb.classes_) == [0, 1]
    assert p.shape == (len(Xva), 2) and np.allclose(p.sum(axis=1), 1)
    # column 1 is the positive class: it should be higher on average for true positives
    assert p[yva.to_numpy() == 1, 1].mean() > p[yva.to_numpy() == 0, 1].mean()


def test_nb_class_order_independent_of_row_order(coded):
    Xtr, ytr, *_ = coded
    flipped = ytr.iloc[::-1]
    nb = NaiveBayesClassifier().fit(Xtr.iloc[::-1], flipped)
    assert list(nb.classes_) == [0, 1]


def test_nb_errors(coded):
    Xtr, ytr, Xva, _ = coded
    nb = NaiveBayesClassifier().fit(Xtr, ytr)
    with pytest.raises(ValueError, match="Unseen state"):
        nb.predict_proba(Xva.assign(cp=17))
    with pytest.raises(ValueError, match="missing"):
        nb.predict_proba(Xva.astype(float).assign(ca=np.nan))
    with pytest.raises(ValueError, match="must not be passed"):
        nb.predict_proba(Xva.assign(num=0))
    with pytest.raises(ValueError, match="Both classes"):
        NaiveBayesClassifier().fit(Xtr, ytr * 0)
    with pytest.raises(RuntimeError, match="not fitted"):
        NaiveBayesClassifier().predict_proba(Xva)


# ---- pipelines and identical folds ----
def test_outer_folds_deterministic_and_stratified(data):
    X, y = data
    f1, f2 = outer_folds(y), outer_folds(y)
    assert len(f1) == 25  # 5 folds x 5 repeats
    for (a, b), (c, d) in zip(f1, f2):
        assert np.array_equal(a, c) and np.array_equal(b, d)
        assert len(set(a) & set(b)) == 0
    for r in range(5):  # each repeat partitions the patients
        assert sorted(np.concatenate([va for _, va in f1[5 * r:5 * r + 5]])) == list(range(len(y)))
    rates = [y.iloc[va].mean() for _, va in f1]
    assert max(rates) - min(rates) < 0.06


def test_both_models_run_on_identical_folds(data):
    X, y = data
    folds = outer_folds(y)
    seen = {"bn": [], "nb": []}
    for tr, va in folds[:5]:
        for name, builder in (("bn", build_bn_pipeline), ("nb", build_nb_pipeline)):
            pipe = clone(builder()).fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[va])
            assert p.shape == (len(va), 2) and np.allclose(p.sum(axis=1), 1)
            seen[name].append((tr.tobytes(), va.tobytes()))
    assert seen["bn"] == seen["nb"]


def test_pipeline_handles_missing_ca_thal_via_preprocessor(data):
    X, y = data
    pipe = build_bn_pipeline().fit(X.iloc[:250], y.iloc[:250])
    rows = X[X[["ca", "thal"]].isna().any(axis=1)]
    p = pipe.predict_proba(rows)
    assert p.shape == (len(rows), 2) and not np.isnan(p).any()


def test_pipeline_rejects_target_in_predictors(data):
    X, y = data
    df = load_cleveland()
    with pytest.raises(ValueError, match="must not be passed"):
        build_nb_pipeline().fit(X.assign(num=df["num"]), y)
