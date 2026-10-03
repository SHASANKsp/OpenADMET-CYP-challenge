"""
cyp_lib.py - modular toolkit for the CYP direct-inhibition regression task.

Every stage is a standalone function so cells in the notebook stay plug-and-play:
featurize -> EDA -> chemical-space viz -> Tanimoto split -> preprocessing ->
model zoo -> ST-RAE metric -> per-isoform CV comparison -> tune -> submission.

Licensing: RDKit (BSD-3), scikit-learn (BSD-3), LightGBM (MIT), XGBoost (Apache-2.0),
CatBoost (Apache-2.0), pandas/numpy/scipy/matplotlib/seaborn (BSD). All clean for
commercial use. Optional PyTorch / chemprop paths are import-guarded.
"""
from __future__ import annotations
import warnings, time
from dataclasses import dataclass, field
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Descriptors3D, AllChem, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from scipy.stats import spearmanr, kendalltau

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

CYPS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TARGET  = "{c}_pIC50_direct_inhibition"
CONF_HI = "{c}_pIC50_direct_inhibition_conf_high"
CONF_LO = "{c}_pIC50_direct_inhibition_conf_low"
ISO_COLORS = {"CYP1A2": "#38bdf8", "CYP2C9": "#a78bfa", "CYP2D6": "#f5a524", "CYP3A4": "#2dd4bf"}

# =====================================================================================
# STAGE 1-2 : FEATURIZATION  (RDKit 2D + 3D descriptors + Morgan fingerprint)
# =====================================================================================
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_D2_NAMES = [n for n, _ in Descriptors._descList]
_D2_FUNCS = [f for _, f in Descriptors._descList]
_D3 = {  # fast scalar 3D shape descriptors
    "Asphericity": Descriptors3D.Asphericity, "Eccentricity": Descriptors3D.Eccentricity,
    "InertialShapeFactor": Descriptors3D.InertialShapeFactor, "NPR1": Descriptors3D.NPR1,
    "NPR2": Descriptors3D.NPR2, "PMI1": Descriptors3D.PMI1, "PMI2": Descriptors3D.PMI2,
    "PMI3": Descriptors3D.PMI3, "RadiusOfGyration": Descriptors3D.RadiusOfGyration,
    "SpherocityIndex": Descriptors3D.SpherocityIndex,
}


def _fp_array(mol):
    a = np.zeros((2048,), dtype=np.int8); DataStructs.ConvertToNumpyArray(_MORGAN.GetFingerprint(mol), a)
    return a.astype(np.float32)


def _desc2d(mol):
    out = np.empty(len(_D2_FUNCS))
    for i, f in enumerate(_D2_FUNCS):
        try: out[i] = f(mol)
        except Exception: out[i] = np.nan
    return out


def _desc3d(mol, seed=0xC0FFEE):
    v = np.full(len(_D3), np.nan)
    try:
        mh = Chem.AddHs(mol); p = AllChem.ETKDGv3(); p.randomSeed = seed
        if AllChem.EmbedMolecule(mh, p) != 0 and AllChem.EmbedMolecule(mh, useRandomCoords=True, randomSeed=seed) != 0:
            return v
        try: AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
        except Exception: pass
        for i, fn in enumerate(_D3.values()):
            try: v[i] = fn(mh)
            except Exception: pass
    except Exception: pass
    return v


@dataclass
class FeatureSet:
    """Holds the three feature blocks separately so each can be analysed on its own."""
    blocks: dict               # {'fp': arr, '2d': arr, '3d': arr}
    names: dict                # {'fp': [...], '2d': [...], '3d': [...]}
    fps: list                  # ExplicitBitVect per molecule (for Tanimoto work)
    valid: np.ndarray          # bool mask of parseable SMILES

    def matrix(self, use=("fp", "2d", "3d")):
        """Concatenate chosen blocks -> (X, names, kinds)."""
        use = [u for u in use if u in self.blocks]
        X = np.concatenate([self.blocks[u] for u in use], axis=1)
        names = sum([self.names[u] for u in use], [])
        kinds = sum([[u] * self.blocks[u].shape[1] for u in use], [])
        return X.astype(np.float32), names, np.array(kinds)


def featurize(smiles, use_3d=True, verbose=True) -> FeatureSet:
    """Stage 1. SMILES -> FeatureSet(fp 2048, 2D ~217, 3D 10)."""
    t0 = time.time(); mols, fps, valid = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        mols.append(m); valid.append(m is not None)
        fps.append(_MORGAN.GetFingerprint(m) if m is not None else None)
    valid = np.array(valid); n = len(mols)
    fp = np.full((n, 2048), np.nan, np.float32)
    d2 = np.full((n, len(_D2_NAMES)), np.nan, np.float32)
    d3 = np.full((n, len(_D3)), np.nan, np.float32) if use_3d else None
    for i, m in enumerate(mols):
        if m is None: continue
        fp[i] = _fp_array(m); d2[i] = _desc2d(m)
        if use_3d: d3[i] = _desc3d(m)
        if verbose and (i + 1) % 500 == 0: print(f"    {i+1}/{n} ({time.time()-t0:.0f}s)")
    blocks = {"fp": fp, "2d": d2}; names = {"fp": [f"fp_{i}" for i in range(2048)], "2d": list(_D2_NAMES)}
    if use_3d: blocks["3d"] = d3; names["3d"] = list(_D3.keys())
    if verbose: print(f"  featurized {n} molecules in {time.time()-t0:.0f}s ({valid.sum()} valid)")
    return FeatureSet(blocks, names, fps, valid)


def targets_dict(df):
    return {c: df[TARGET.format(c=c)].values.astype(float) for c in CYPS}


# =====================================================================================
# STAGE 3 : FEATURE EDA + PLOTS
# =====================================================================================
def plot_target_distributions(df):
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.4))
    for a, c in zip(ax, CYPS):
        y = df[TARGET.format(c=c)].dropna()
        a.hist(y, bins=40, color=ISO_COLORS[c], alpha=.85)
        a.axvline(4, color="crimson", ls="--"); a.set_title(f"{c}  n={len(y)}"); a.set_xlabel("pIC50")
    fig.suptitle("Direct-inhibition pIC50 distributions (red = low-activity threshold)"); fig.tight_layout(); plt.show()


def plot_ci_width_vs_activity(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in CYPS:
        p = df[TARGET.format(c=c)]; w = df[CONF_HI.format(c=c)] - df[CONF_LO.format(c=c)]
        m = p.notna() & w.notna(); ax.scatter(p[m], w[m], s=5, alpha=.25, color=ISO_COLORS[c], label=c)
    ax.axvline(4, color="crimson", ls="--"); ax.set_xlabel("pIC50"); ax.set_ylabel("95% CI width")
    ax.set_title("Credible-interval width vs potency (wider at low activity -> ST-RAE lever)")
    ax.legend(); fig.tight_layout(); plt.show()


def feature_target_spearman(feat, y, kinds_use=("2d", "3d")):
    """Return {isoform: DataFrame(feature, rho)} sorted by |rho|."""
    X, names, kinds = feat.matrix(("fp", "2d", "3d"))
    mask = np.isin(kinds, kinds_use); idx = np.where(mask)[0]
    res = {}
    for c in CYPS:
        yc = y[c]; ok = feat.valid & np.isfinite(yc)
        rows = []
        for j in idx:
            col = X[ok, j]; g = np.isfinite(col)
            if g.sum() > 30 and np.nanstd(col[g]) > 0:
                rows.append((names[j], spearmanr(col[g], yc[ok][g]).statistic))
        d = pd.DataFrame(rows, columns=["feature", "rho"]).dropna()
        res[c] = d.reindex(d.rho.abs().sort_values(ascending=False).index)
    return res


def plot_feature_target_spearman(assoc, top_k=12):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a, c in zip(ax, CYPS):
        d = assoc[c].head(top_k).iloc[::-1]
        a.barh(d.feature, d.rho, color=ISO_COLORS[c]); a.axvline(0, color="k", lw=.6)
        a.set_title(f"{c}: top |Spearman| features"); a.set_xlabel("rho vs pIC50")
    fig.tight_layout(); plt.show()


def plot_descriptor_correlation(feat, top_var=30):
    """Correlation heatmap of the highest-variance 2D descriptors (redundancy audit)."""
    X, names, kinds = feat.matrix(("2d",))
    Xv = X[feat.valid]; var = np.nanvar(Xv, axis=0); top = np.argsort(-var)[:top_var]
    sub = pd.DataFrame(Xv[:, top], columns=[names[i] for i in top]).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(sub, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .6}, ax=ax)
    ax.set_title(f"Pairwise correlation of top-{top_var} variance 2D descriptors"); fig.tight_layout(); plt.show()


# =====================================================================================
# STAGE 4 : CHEMICAL-SPACE & CLUSTERING VISUALISATION
# =====================================================================================
def _embed(X, method="pca", seed=0):
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    Xi = StandardScaler().fit_transform(Xi)
    if method == "tsne":
        return TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(Xi)
    return PCA(n_components=2, random_state=seed).fit_transform(Xi)


def plot_space_train_test(feat_tr, feat_te, block="fp", method="pca"):
    """Embed TRAIN+TEST of one block together; show chemical-space overlap of the two sets."""
    Xtr, _, _ = feat_tr.matrix((block,)); Xte, _, _ = feat_te.matrix((block,))
    X = np.vstack([Xtr[feat_tr.valid], Xte[feat_te.valid]])
    emb = _embed(X, method)
    ntr = feat_tr.valid.sum()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(emb[:ntr, 0], emb[:ntr, 1], s=7, alpha=.35, color="#2dd4bf", label=f"train ({ntr})")
    ax.scatter(emb[ntr:, 0], emb[ntr:, 1], s=14, alpha=.7, color="#fb7185", label=f"test ({len(emb)-ntr})")
    ax.set_title(f"Chemical space ({block}, {method.upper()}): train vs test coverage")
    ax.legend(); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); fig.tight_layout(); plt.show()


def plot_space_colored(feat, color_by, title, block="fp", method="pca", cmap="tab20"):
    """Embed one block of TRAIN; colour points by a vector (cluster id, fold, or a target)."""
    X, _, _ = feat.matrix((block,)); emb = _embed(X[feat.valid], method)
    cv = np.asarray(color_by)[feat.valid]
    fig, ax = plt.subplots(figsize=(7, 6))
    if cv.dtype.kind in "iu" and len(set(cv)) > 20:   # many clusters -> continuous cmap
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=8, c=cv % 20, cmap=cmap, alpha=.6)
    else:
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=10, c=cv, cmap="viridis", alpha=.7); fig.colorbar(sc, ax=ax, shrink=.7)
    ax.set_title(f"{title} ({block}, {method.upper()})"); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.tight_layout(); plt.show()


# =====================================================================================
# STAGE 5 : TANIMOTO / BUTINA SPLIT
# =====================================================================================
def butina_clusters(fps, cutoff=0.65):
    """Cluster by Tanimoto DISTANCE (1 - similarity). cutoff 0.65 == 0.35 similarity floor."""
    idx = [i for i, f in enumerate(fps) if f is not None]; fv = [fps[i] for i in idx]; n = len(fv)
    dists = []
    for i in range(1, n):
        dists.extend(1 - s for s in DataStructs.BulkTanimotoSimilarity(fv[i], fv[:i]))
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    lab = np.full(len(fps), -1)
    for cid, members in enumerate(clusters):
        for loc in members: lab[idx[loc]] = cid
    nxt = lab.max() + 1
    for i in range(len(fps)):
        if lab[i] == -1: lab[i] = nxt; nxt += 1
    return lab


def cluster_group_folds(labels, n_splits=5):
    """GroupKFold on cluster id so analogue series never span folds. Returns fold id per molecule."""
    fold = np.full(len(labels), -1)
    for f, (_, va) in enumerate(GroupKFold(n_splits).split(np.zeros(len(labels)), groups=labels)):
        fold[va] = f
    return fold


def split_diagnostics(fps, fold, labels):
    f0 = fold == 0
    tr = [fps[i] for i in np.where(~f0)[0] if fps[i] is not None]
    va = [fps[i] for i in np.where(f0)[0] if fps[i] is not None]
    ms = [max(DataStructs.BulkTanimotoSimilarity(f, tr)) for f in va[:400]]
    return {"n_clusters": int(len(set(labels))),
            "fold0_val_maxTanimoto_median": round(float(np.median(ms)), 3),
            "fold0_val_maxTanimoto_p90": round(float(np.percentile(ms, 90)), 3)}


# =====================================================================================
# STAGE 6 : PREPROCESSING (scaling + feature selection), family-aware
# =====================================================================================
def build_preprocessor(family, k_best=None, pca=None):
    """Return a transformer appropriate to the model family.
    Trees/boosters: impute only (scale-invariant, NaN handled or imputed).
    Linear/kernel/knn/mlp: impute + drop zero-variance + standardize + optional SelectKBest / PCA."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if family in ("linear", "svm", "knn", "mlp"):
        steps += [("var", VarianceThreshold(1e-8)), ("scale", StandardScaler())]
        if k_best:  steps.append(("select", SelectKBest(f_regression, k=k_best)))
        if pca:     steps.append(("pca", PCA(n_components=pca, random_state=0)))
    return Pipeline(steps)


# =====================================================================================
# STAGE 7 : MODEL ZOO  (classical -> boosting -> neural); plug-and-play registry
# =====================================================================================
def model_registry(seed=0, k_best=300):
    """name -> dict(family, make()); make() returns a full ('prep'->'est') pipeline.
    Add a model == add one entry here."""
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    reg = {}

    def add(name, family, est, **pp):
        reg[name] = dict(family=family,
                         make=lambda e=est, f=family, pp=pp: Pipeline([("prep", build_preprocessor(f, **pp)), ("est", clone(e))]))

    # --- linear ---
    add("Ridge", "linear", Ridge(alpha=5.0, random_state=seed), k_best=k_best)
    add("ElasticNet", "linear", ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=seed), k_best=k_best)
    # --- kernel / neighbours ---
    add("SVR_rbf", "svm", SVR(C=5.0, gamma="scale"), k_best=k_best)
    add("KNN", "knn", KNeighborsRegressor(n_neighbors=7, weights="distance"), k_best=k_best)
    # --- bagging trees ---
    add("RandomForest", "tree", RandomForestRegressor(n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=seed))
    add("ExtraTrees", "tree", ExtraTreesRegressor(n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=seed))
    # --- boosting ---
    add("HistGBR", "tree", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=seed))
    try:
        from lightgbm import LGBMRegressor
        add("LightGBM", "tree", LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
            subsample=.8, subsample_freq=1, colsample_bytree=.6, reg_lambda=5., n_jobs=-1, random_state=seed, verbose=-1))
    except Exception: pass
    try:
        from xgboost import XGBRegressor
        add("XGBoost", "tree", XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=.8,
            colsample_bytree=.6, reg_lambda=5., n_jobs=-1, random_state=seed, verbosity=0))
    except Exception: pass
    try:
        from catboost import CatBoostRegressor
        add("CatBoost", "tree", CatBoostRegressor(iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=5.,
            random_seed=seed, verbose=0))
    except Exception: pass
    # --- neural net (deep learning; sklearn MLP) ---
    add("MLP", "mlp", MLPRegressor(hidden_layer_sizes=(512, 256, 128), activation="relu", alpha=1e-3,
        batch_size=128, learning_rate_init=1e-3, early_stopping=True, max_iter=300, random_state=seed), k_best=k_best)
    return reg


# =====================================================================================
# STAGE 8 : ST-RAE METRIC  (organizers' primary) + secondary metrics
# =====================================================================================
def soft_threshold_ae(pred, lo, hi, true=None):
    pred = np.asarray(pred, float); err = np.zeros_like(pred)
    ci = np.isfinite(lo) & np.isfinite(hi)
    err[ci & (pred < lo)] = (lo - pred)[ci & (pred < lo)]
    err[ci & (pred > hi)] = (pred - hi)[ci & (pred > hi)]
    if true is not None:
        nb = ~ci & np.isfinite(true); err[nb] = np.abs(pred - true)[nb]
    return err


def st_rae(pred, true, lo, hi):
    """Soft-threshold Relative Absolute Error for one endpoint.
    Numerator = model soft-threshold error; denominator = naive-mean predictor's.
    Denominator is a per-endpoint constant, so model RANKING is normalisation-invariant."""
    pred, true, lo, hi = map(lambda z: np.asarray(z, float), (pred, true, lo, hi))
    num = soft_threshold_ae(pred, lo, hi, true).sum()
    base = soft_threshold_ae(np.full_like(true, np.nanmean(true)), lo, hi, true).sum()
    return float(num / max(base, 1e-9)), float(soft_threshold_ae(pred, lo, hi, true).mean())


def score_endpoint(y, p, lo, hi):
    srae, stae = st_rae(p, y, lo, hi)
    return dict(ST_RAE=round(srae, 4), ST_AE=round(stae, 4), MAE=round(mean_absolute_error(y, p), 4),
                R2=round(r2_score(y, p), 4), Spearman=round(spearmanr(p, y).statistic, 4),
                Kendall=round(kendalltau(p, y).statistic, 4))


# =====================================================================================
# STAGE 9 : PER-ISOFORM, PER-MODEL CROSS-VALIDATED COMPARISON
# =====================================================================================
def cv_oof(pipe, X, y, labeled, fold):
    """Out-of-fold predictions for one estimator on one isoform."""
    oof = np.full(len(y), np.nan)
    for f in range(int(fold.max()) + 1):
        tr = labeled & (fold != f); va = labeled & (fold == f)
        if tr.sum() < 20 or va.sum() == 0: continue
        est = pipe(); est.fit(X[tr], y[tr]); oof[va] = est.predict(X[va])
    return oof


def compare_models(feat, df, fold, registry, use=("fp", "2d", "3d"), verbose=True):
    """For every (isoform, model): series-aware CV -> metrics row. Returns (results_df, oof_store)."""
    X, _, _ = feat.matrix(use); rows = []; oof_store = {}
    for c in CYPS:
        col = TARGET.format(c=c); y = df[col].values.astype(float)
        lo, hi = df[CONF_LO.format(c=c)].values, df[CONF_HI.format(c=c)].values
        labeled = feat.valid & np.isfinite(y)
        for name, spec in registry.items():
            t0 = time.time(); oof = cv_oof(spec["make"], X, y, labeled, fold)
            m = labeled & np.isfinite(oof)
            met = score_endpoint(y[m], oof[m], lo[m], hi[m])
            met.update(isoform=c, model=name, n=int(m.sum()), sec=round(time.time() - t0, 1))
            rows.append(met); oof_store[(c, name)] = oof
            if verbose: print(f"  {c:7s} {name:13s} ST-RAE={met['ST_RAE']:.3f} R2={met['R2']:+.3f} ({met['sec']}s)")
    res = pd.DataFrame(rows)[["isoform", "model", "n", "ST_RAE", "ST_AE", "MAE", "R2", "Spearman", "Kendall", "sec"]]
    return res, oof_store


def plot_model_comparison(res):
    piv = res.pivot(index="model", columns="isoform", values="ST_RAE")
    fig, ax = plt.subplots(figsize=(8, max(4, .5 * len(piv))))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis_r", cbar_kws={"label": "ST-RAE (lower better)"}, ax=ax)
    ax.set_title("Model x isoform : cross-validated ST-RAE"); fig.tight_layout(); plt.show()


def select_best(res):
    """Best (lowest ST-RAE) model per isoform."""
    best = {}
    for c in CYPS:
        sub = res[res.isoform == c].sort_values("ST_RAE"); best[c] = sub.iloc[0]["model"]
    return best


# =====================================================================================
# STAGE 10 : FINE-TUNE the selected model per isoform (ST-RAE objective, series-aware)
# =====================================================================================
def finetune(model_name, feat, df, fold, isoform, seed=0, n_iter=12, use=("fp", "2d", "3d")):
    """Randomised search over a small per-family grid, scored by true OOF ST-RAE on the folds."""
    import random; random.seed(seed); rng = np.random.default_rng(seed)
    X, _, _ = feat.matrix(use); col = TARGET.format(c=isoform); y = df[col].values.astype(float)
    lo, hi = df[CONF_LO.format(c=isoform)].values, df[CONF_HI.format(c=isoform)].values
    labeled = feat.valid & np.isfinite(y)
    fam = model_registry(seed)[model_name]["family"]

    GRID = {  # est__<param> : samplers
        "LightGBM": dict(est__num_leaves=[31, 63, 127], est__learning_rate=[.02, .03, .05],
                         est__colsample_bytree=[.5, .6, .8], est__reg_lambda=[1., 5., 10.], est__n_estimators=[400, 600, 900]),
        "XGBoost": dict(est__max_depth=[4, 6, 8], est__learning_rate=[.02, .03, .05],
                        est__subsample=[.7, .8, 1.], est__reg_lambda=[1., 5., 10.], est__n_estimators=[400, 600, 900]),
        "CatBoost": dict(est__depth=[4, 6, 8], est__learning_rate=[.02, .03, .05], est__l2_leaf_reg=[1., 5., 9.]),
        "RandomForest": dict(est__n_estimators=[300, 500, 800], est__max_features=["sqrt", .3, .5], est__min_samples_leaf=[1, 2, 5]),
        "ExtraTrees": dict(est__n_estimators=[300, 500, 800], est__max_features=["sqrt", .3, .5], est__min_samples_leaf=[1, 2, 5]),
        "HistGBR": dict(est__learning_rate=[.03, .05, .1], est__max_iter=[300, 500, 800], est__l2_regularization=[0., 1., 5.]),
        "Ridge": dict(est__alpha=[1., 5., 10., 50.]),
        "ElasticNet": dict(est__alpha=[.003, .01, .03], est__l1_ratio=[.1, .3, .5]),
        "SVR_rbf": dict(est__C=[1., 5., 10.], est__gamma=["scale", "auto"]),
        "KNN": dict(est__n_neighbors=[5, 7, 11, 15], est__weights=["uniform", "distance"]),
        "MLP": dict(est__alpha=[1e-4, 1e-3, 1e-2], est__hidden_layer_sizes=[(256, 128), (512, 256, 128)], est__learning_rate_init=[5e-4, 1e-3]),
    }
    grid = GRID.get(model_name, {})
    base = model_registry(seed)[model_name]["make"]
    if not grid:
        oof = cv_oof(base, X, y, labeled, fold); m = labeled & np.isfinite(oof)
        return base(), st_rae(oof[m], y[m], lo[m], hi[m])[0], {}
    keys = list(grid)
    best_s, best_p = np.inf, {}
    for _ in range(n_iter):
        params = {k: grid[k][rng.integers(len(grid[k]))] for k in keys}
        def mk(p=params): e = base(); e.set_params(**p); return e
        oof = cv_oof(mk, X, y, labeled, fold); m = labeled & np.isfinite(oof)
        s = st_rae(oof[m], y[m], lo[m], hi[m])[0]
        if s < best_s: best_s, best_p = s, params
    final = base(); final.set_params(**best_p)
    return final, round(best_s, 4), best_p


# =====================================================================================
# STAGE 11 : FINAL FIT -> TEST PREDICTION -> SUBMISSION  (with extrapolation safeguard)
# =====================================================================================
def fit_predict_submission(best_estimators, feat_tr, df_tr, feat_te, df_te,
                           use=("fp", "2d", "3d"), clip=True, clip_pad=0.25):
    """Refit each isoform's chosen+tuned estimator on all labeled rows; predict test.
    clip=True winsorizes predictions to the training label range (+/- pad) per isoform -
    this is the guard against the CYP2D6 extrapolation blow-up (R2 went strongly negative
    when unconstrained predictions ran past the shifted test distribution)."""
    Xtr, _, _ = feat_tr.matrix(use); Xte, _, _ = feat_te.matrix(use)
    sub = pd.DataFrame({"SMILES": df_te["SMILES"], "Molecule_Name": df_te["Molecule_Name"]})
    for c in CYPS:
        col = TARGET.format(c=c); y = df_tr[col].values.astype(float); labeled = feat_tr.valid & np.isfinite(y)
        est = best_estimators[c]; est.fit(Xtr[labeled], y[labeled])
        p = np.full(len(Xte), np.nan); p[feat_te.valid] = est.predict(Xte[feat_te.valid])
        p[~np.isfinite(p)] = np.nanmean(y[labeled])
        if clip:
            lo, hi = np.nanmin(y[labeled]) - clip_pad, np.nanmax(y[labeled]) + clip_pad
            p = np.clip(p, lo, hi)
        sub[col] = p
    return sub


# =====================================================================================
# STAGE 12 : DIAGNOSTICS on the chosen models
# =====================================================================================
def plot_pred_vs_actual(oof_store, best, df):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    for a, c in zip(ax, CYPS):
        col = TARGET.format(c=c); y = df[col].values.astype(float); oof = oof_store[(c, best[c])]
        m = np.isfinite(y) & np.isfinite(oof)
        a.scatter(y[m], oof[m], s=8, alpha=.3, color=ISO_COLORS[c])
        lim = [min(y[m].min(), oof[m].min()), max(y[m].max(), oof[m].max())]
        a.plot(lim, lim, "k--", lw=.8); a.set_xlabel("actual pIC50"); a.set_ylabel("OOF pred")
        a.set_title(f"{c}  best={best[c]}  R2={r2_score(y[m], oof[m]):.2f}")
    fig.suptitle("Out-of-fold predicted vs actual (chosen model per isoform)"); fig.tight_layout(); plt.show()
