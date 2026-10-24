"""
Similarity-aware train/test split for QSPR / QSAR modelling.

Drop-in replacement for sklearn's train_test_split that keeps the *same*
regions of chemical space represented in both train and test, in the same
proportions. It does this by clustering molecules on Tanimoto similarity
(Taylor-Butina) and then splitting each cluster proportionally.

This is the opposite goal of a scaffold / leave-cluster-out split: use that
when you want a hard, dissimilar test set to probe extrapolation. Use *this*
when you want train and test drawn from the same distribution (interpolation),
which is what "a similar share of molecules in train and test" asks for.
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina


def _fingerprints(smiles, radius=2, n_bits=2048):
    """Morgan (ECFP-like) bit fingerprints. Returns (fps, valid_index_mask)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, ok = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else s
        if mol is None:
            ok.append(False)
            continue
        fps.append(gen.GetFingerprint(mol))
        ok.append(True)
    return fps, np.asarray(ok, dtype=bool)


def _butina_clusters(fps, cutoff):
    """Taylor-Butina clustering. cutoff is a *distance* (1 - Tanimoto)."""
    n = len(fps)
    # Condensed lower-triangle distance matrix, as Butina expects.
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    # clusters: tuple of tuples of indices (into the fp list), largest first.
    labels = np.empty(n, dtype=int)
    for cid, members in enumerate(clusters):
        for idx in members:
            labels[idx] = cid
    return labels


def similarity_train_test_split(
    X,
    y,
    smiles,
    test_size=0.2,
    random_state=42,
    cutoff=0.35,
    radius=2,
    n_bits=2048,
    return_indices=False,
    plot=True,
    embedding="tsne",
):
    """
    Similarity-stratified train/test split.

    Parameters
    ----------
    X, y : array-like, length n
        Your feature matrix (descriptors) and targets. Returned split in the
        SAME order and type-ish as train_test_split: X_train, X_test, y_train, y_test.
    smiles : sequence of str (or RDKit Mol), length n
        One SMILES per row of X. This is the ONLY extra argument you must add
        versus the plain train_test_split call. Order must match X / y.
    test_size : float
        Fraction held out, applied *within each similarity cluster* so the
        global fraction and the per-region fraction both land near this value.
    random_state : int
        Controls the within-cluster shuffling. Reproducible.
    cutoff : float
        Butina Tanimoto *distance* cutoff (1 - similarity). 0.35 means molecules
        within ~0.65 Tanimoto are grouped. Lower -> tighter, more clusters.
    radius, n_bits : int
        Morgan fingerprint settings.
    return_indices : bool
        If True, also return (train_idx, test_idx) as a 5th/6th value.
    plot : bool
        If True, show (not save) a two-panel map comparing this split with a
        plain random train_test_split of the same size, so you can see where
        each strategy places its held-out molecules in chemical space.
    embedding : "tsne" | "mds" | "pca"
        2D projection used only for the plot. Ignored unless plot=True.

    Returns
    -------
    X_train, X_test, y_train, y_test
        (plus train_idx, test_idx if return_indices=True)
    """
    X = np.asarray(X) if not hasattr(X, "iloc") else X
    y = np.asarray(y) if not hasattr(y, "iloc") else y
    smiles = list(smiles)

    n = len(smiles)
    if len(X) != n or len(y) != n:
        raise ValueError(f"X ({len(X)}), y ({len(y)}) and smiles ({n}) must align.")

    fps, valid = _fingerprints(smiles, radius=radius, n_bits=n_bits)
    valid_positions = np.where(valid)[0]  # original indices of parseable mols
    if len(valid_positions) < n:
        bad = n - len(valid_positions)
        print(f"[similarity_split] warning: {bad} SMILES failed to parse and "
              f"were dropped from the split.")

    labels = _butina_clusters(fps, cutoff)  # labels indexed over valid mols only

    # Group original positions by cluster label.
    by_cluster = defaultdict(list)
    for fp_pos, orig_pos in enumerate(valid_positions):
        by_cluster[labels[fp_pos]].append(orig_pos)

    rng = np.random.default_rng(random_state)
    cluster_ids = sorted(by_cluster, key=lambda c: -len(by_cluster[c]))
    for cid in cluster_ids:
        rng.shuffle(by_cluster[cid])

    n_valid = len(valid_positions)
    target_test = int(round(n_valid * test_size))

    # Largest-remainder apportionment of the test quota across clusters.
    # Each cluster's ideal test count is k * test_size; take the floor, then
    # hand out the leftover slots to the clusters with the largest fractional
    # parts. This keeps multi-member clusters split internally AND lets the
    # global test fraction land on target even when most clusters are tiny
    # (singletons), instead of every small cluster rounding to zero test.
    ideals = {cid: len(by_cluster[cid]) * test_size for cid in cluster_ids}
    base = {cid: int(np.floor(ideals[cid])) for cid in cluster_ids}
    # never take an entire cluster into test: keeps every region seen in train
    base = {cid: min(base[cid], len(by_cluster[cid]) - 1) for cid in cluster_ids}
    assigned = sum(base.values())
    leftover = target_test - assigned

    remainders = sorted(
        cluster_ids,
        key=lambda c: (ideals[c] - np.floor(ideals[c]), len(by_cluster[c])),
        reverse=True,
    )
    extra = defaultdict(int)
    for cid in remainders:
        if leftover <= 0:
            break
        if base[cid] + extra[cid] < len(by_cluster[cid]) - 1:
            extra[cid] += 1
            leftover -= 1
    # If still short (many singletons capped at k-1), allow singletons -> test
    # as a last resort, largest-fractional first, so the ratio is still met.
    if leftover > 0:
        for cid in remainders:
            if leftover <= 0:
                break
            cap = len(by_cluster[cid])
            if base[cid] + extra[cid] < cap:
                extra[cid] += 1
                leftover -= 1

    train_idx, test_idx = [], []
    for cid in cluster_ids:
        members = by_cluster[cid]
        n_test = base[cid] + extra[cid]
        test_idx.extend(members[:n_test])
        train_idx.extend(members[n_test:])

    train_idx = np.array(sorted(train_idx), dtype=int)
    test_idx = np.array(sorted(test_idx), dtype=int)

    if plot:
        from sklearn.model_selection import train_test_split as _sk_split
        _, rand_test_idx = _sk_split(
            valid_positions, test_size=test_size, random_state=random_state)
        _draw_split_comparison(
            fps, valid_positions,
            sim_test=set(int(i) for i in test_idx),
            rand_test=set(int(i) for i in rand_test_idx),
            embedding=embedding, random_state=random_state,
            savepath=None)  # savepath=None -> plt.show(), nothing written

    def take(a, idx):
        return a.iloc[idx] if hasattr(a, "iloc") else a[idx]

    out = (take(X, train_idx), take(X, test_idx),
           take(y, train_idx), take(y, test_idx))
    if return_indices:
        out = out + (train_idx, test_idx)
    return out


# ---------------------------------------------------------------------------
# Visualisation: compare how similarity-split vs random-split carve up
# chemical space. Produces a two-panel cluster map with test molecules ringed.
# ---------------------------------------------------------------------------

def _tanimoto_distance_matrix(fps):
    """Square dense Tanimoto distance matrix (1 - similarity)."""
    from rdkit import DataStructs
    n = len(fps)
    D = np.zeros((n, n))
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        D[i] = [1.0 - s for s in sims]
    return D


def _draw_split_comparison(fps, orig, sim_test, rand_test,
                           embedding="tsne", random_state=42,
                           savepath=None, figsize=(15, 7)):
    """
    Two-panel grey/red map: similarity split vs random split, same coordinates.
    savepath=None -> plt.show() and write nothing; else save to that path.
    Shared by similarity_train_test_split(plot=True) and plot_split_comparison.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    orig = np.asarray(orig)
    sim_test = set(int(i) for i in sim_test)
    rand_test = set(int(i) for i in rand_test)

    # --- 2D embedding of chemical space ---------------------------------
    D = _tanimoto_distance_matrix(fps)
    emb = embedding
    if len(fps) < 6:
        emb = "pca"
    if emb == "tsne":
        from sklearn.manifold import TSNE
        perp = max(5, min(30, len(fps) // 4))
        coords = TSNE(n_components=2, metric="precomputed", init="random",
                      perplexity=perp, random_state=random_state).fit_transform(D)
    elif emb == "mds":
        from sklearn.manifold import MDS
        coords = MDS(n_components=2, dissimilarity="precomputed",
                     random_state=random_state, normalized_stress="auto"
                     ).fit_transform(D)
    else:  # pca on raw bits
        from sklearn.decomposition import PCA
        bits = np.array([list(fp.ToBitString()) for fp in fps], dtype=float)
        coords = PCA(n_components=2, random_state=random_state).fit_transform(bits)

    # map valid-fp row -> original position, and build per-original coords
    coord_by_orig = {int(orig[i]): coords[i] for i in range(len(orig))}

    TRAIN_C = "#B9C2CC"   # muted grey-blue
    TEST_C = "#D1495B"    # clear red

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    def draw(ax, test_set, title):
        xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
        for p in orig:
            p = int(p)
            x, yy = coord_by_orig[p]
            if p in test_set:
                xs_te.append(x); ys_te.append(yy)
            else:
                xs_tr.append(x); ys_tr.append(yy)
        ax.scatter(xs_tr, ys_tr, c=TRAIN_C, s=45, alpha=0.8,
                   edgecolors="white", linewidths=0.4, zorder=2, label="train")
        ax.scatter(xs_te, ys_te, c=TEST_C, s=70, alpha=0.95,
                   edgecolors="white", linewidths=0.6, zorder=3, label="test")
        ax.set_title(f"{title}  —  {len(test_set)}/{len(orig)} test "
                     f"({len(test_set)/len(orig):.0%})", fontsize=12)
        ax.set_xlabel(f"{emb.upper()}-1"); ax.set_ylabel(f"{emb.upper()}-2")
        ax.tick_params(labelsize=8)

    draw(axes[0], sim_test, "Similarity split")
    draw(axes[1], rand_test, "Random train_test_split")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TRAIN_C,
               markeredgecolor="white", markersize=9, label="train"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEST_C,
               markeredgecolor="white", markersize=9, label="test"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Where the test set sits in chemical space", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


def plot_split_comparison(
    smiles,
    sim_test_idx=None,
    rand_test_idx=None,
    test_size=0.2,
    random_state=42,
    cutoff=0.35,
    radius=2,
    n_bits=2048,
    embedding="tsne",
    savepath=None,
    figsize=(15, 7),
):
    """
    Standalone version of the comparison plot (grey train / red test, two panels).

    Usually you don't need this — pass plot=True to similarity_train_test_split.
    Use this when you want to render the figure for an arbitrary SMILES list, or
    pass your own precomputed sim_test_idx / rand_test_idx. savepath=None shows
    the figure; give a path to save instead. Returns the matplotlib Figure.
    """
    smiles = list(smiles)
    idx_all = np.arange(len(smiles))
    fps, valid = _fingerprints(smiles, radius=radius, n_bits=n_bits)
    orig = np.where(valid)[0]

    if sim_test_idx is None:
        *_, sim_test_idx = similarity_train_test_split(
            idx_all, idx_all, smiles, test_size=test_size,
            random_state=random_state, cutoff=cutoff, radius=radius,
            n_bits=n_bits, return_indices=True)
    if rand_test_idx is None:
        from sklearn.model_selection import train_test_split as _sk_split
        _, rand_test_idx = _sk_split(
            orig, test_size=test_size, random_state=random_state)

    return _draw_split_comparison(
        fps, orig, sim_test_idx, rand_test_idx,
        embedding=embedding, random_state=random_state,
        savepath=savepath, figsize=figsize)
