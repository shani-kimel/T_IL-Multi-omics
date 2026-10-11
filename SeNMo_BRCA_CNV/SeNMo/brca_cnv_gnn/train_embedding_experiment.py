"""
Lightweight A/B: does adding CNV to SeNMo's own precomputed BRCA embeddings
improve survival C-index?

Config A (no CNV): Linear(48 -> 1) Cox head on the precomputed SeNMo embedding.
Config B (+CNV):   Linear(48+7460 -> 1) Cox head on [embedding, standardized CNA].

This mirrors SeNMo's own official fine-tuning recipe (freeze the whole
pretrained encoder, train only the classifier) -- except here the "frozen
encoder output" is literally the real embedding from the real released
pretrained model, so there's nothing to re-derive or match dimensions for.
Same 5-fold split as everywhere else in this project (train_data.csv/
test_data.csv, stratified, seed 66).
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna.pkl"
GNN_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna_gnn.pkl"
SEGMENT_GNN_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna_segment_gnn.pkl"
GNN_COXMTL_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna_gnn_coxmtl.pkl"
GENE_GNN_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_gene_gnn.pkl"
SEGMENT_GNN_COXMTL_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna_segment_gnn_coxmtl.pkl"
CNV_MRNA_VAE_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cnv_mrna_vae.pkl"
GNN_CROSSMODAL_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna_gnn_crossmodal.pkl"
GNN_CROSSMODAL_HD32_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna_gnn_crossmodal_hd32.pkl"
CNV_METHYLATION_VAE_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cnv_methylation_vae.pkl"
CNV_MUTATION_VAE_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cnv_mutation_vae.pkl"
GNN_CROSSMODAL_MUTATION_DATA_PATH = "/Users/yariv/Desktop/shani project/SeNMo_BRCA_CNV/data/embedding_experiment/brca_embedding_cna_gnn_crossmodal_mutation.pkl"
NUM_FOLDS = 5
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-3
SEED = 66


def cox_loss(survtime, censor, hazard_pred):
    n = len(survtime)
    R_mat = (survtime.reshape(1, -1) >= survtime.reshape(-1, 1)).float()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    return -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)


def make_model(input_dim, model_type):
    if model_type == "linear":
        return nn.Linear(input_dim, 1)
    elif model_type == "mlp":
        # Small + heavily regularized on purpose: ~600 training patients per
        # fold is little enough that even this is a real overfitting risk.
        return nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )
    raise ValueError(model_type)


def train_one_fold(X_train, e_train, t_train, X_test, e_test, t_test, seed=SEED, model_type="linear"):
    torch.manual_seed(seed)
    model = make_model(X_train.shape[1], model_type)
    mlp_weight_decay = WEIGHT_DECAY * 3 if model_type == "mlp" else WEIGHT_DECAY
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=mlp_weight_decay)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    e_train_t = torch.tensor(e_train, dtype=torch.float32)
    t_train_t = torch.tensor(t_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = cox_loss(t_train_t, e_train_t, pred)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_risk = model(X_train_t).numpy().reshape(-1)
        test_risk = model(X_test_t).numpy().reshape(-1)

    train_c = concordance_index(t_train, -train_risk, e_train)
    test_c = concordance_index(t_test, -test_risk, e_test)
    return train_c, test_c, train_risk, test_risk


def run_config(data, use_cna, label, pca_components=None, model_type="linear"):
    embedding = data["embedding"]
    cna = data["cna"]
    e_all, t_all, fold_all = data["e"], data["t"], data["fold"]

    train_cs, test_cs = [], []
    explained_var = []
    for fold in range(1, NUM_FOLDS + 1):
        test_mask = fold_all == fold
        train_mask = ~test_mask

        if use_cna:
            scaler = StandardScaler()
            cna_train = scaler.fit_transform(cna[train_mask])
            cna_test = scaler.transform(cna[test_mask])
            if pca_components is not None:
                pca = PCA(n_components=pca_components, random_state=SEED)
                cna_train = pca.fit_transform(cna_train)
                cna_test = pca.transform(cna_test)
                explained_var.append(pca.explained_variance_ratio_.sum())
            X_train = np.concatenate([embedding[train_mask], cna_train], axis=1)
            X_test = np.concatenate([embedding[test_mask], cna_test], axis=1)
        else:
            X_train = embedding[train_mask]
            X_test = embedding[test_mask]

        train_c, test_c, _, _ = train_one_fold(
            X_train, e_all[train_mask], t_all[train_mask],
            X_test, e_all[test_mask], t_all[test_mask],
            model_type=model_type,
        )
        train_cs.append(train_c)
        test_cs.append(test_c)
        ev_str = f"  (PCA explained var={explained_var[-1]:.1%})" if explained_var else ""
        print(f"[{label}] fold {fold}: train C-index={train_c:.4f}  test C-index={test_c:.4f} "
              f"(n_train={train_mask.sum()} n_test={test_mask.sum()}){ev_str}")

    print(f"[{label}] mean train C-index = {np.mean(train_cs):.4f} +/- {np.std(train_cs):.4f}")
    print(f"[{label}] mean test  C-index = {np.mean(test_cs):.4f} +/- {np.std(test_cs):.4f}")
    return train_cs, test_cs


def run_config_extra_features(embedding, extra, e_all, t_all, fold_all, label, standardize=True, model_type="linear"):
    """Like run_config, but for a precomputed extra feature block (e.g. a GNN
    embedding) that doesn't need PCA -- it's already compact. Still
    standardizes fold-wise (fit on train only) for parity with the raw/PCA
    CNA configs."""
    train_cs, test_cs = [], []
    for fold in range(1, NUM_FOLDS + 1):
        test_mask = fold_all == fold
        train_mask = ~test_mask

        extra_train, extra_test = extra[train_mask], extra[test_mask]
        if standardize:
            scaler = StandardScaler()
            extra_train = scaler.fit_transform(extra_train)
            extra_test = scaler.transform(extra_test)

        X_train = np.concatenate([embedding[train_mask], extra_train], axis=1)
        X_test = np.concatenate([embedding[test_mask], extra_test], axis=1)

        train_c, test_c, _, _ = train_one_fold(
            X_train, e_all[train_mask], t_all[train_mask],
            X_test, e_all[test_mask], t_all[test_mask],
            model_type=model_type,
        )
        train_cs.append(train_c)
        test_cs.append(test_c)
        print(f"[{label}] fold {fold}: train C-index={train_c:.4f}  test C-index={test_c:.4f} "
              f"(n_train={train_mask.sum()} n_test={test_mask.sum()})")

    print(f"[{label}] mean train C-index = {np.mean(train_cs):.4f} +/- {np.std(train_cs):.4f}")
    print(f"[{label}] mean test  C-index = {np.mean(test_cs):.4f} +/- {np.std(test_cs):.4f}")
    return train_cs, test_cs


def main():
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['patient_ids'])} patients, embedding dim=48, CNA dim={data['cna'].shape[1]}")
    print()

    print("=" * 60)
    print("Config A: no CNV (embedding only)")
    print("=" * 60)
    a_train, a_test = run_config(data, use_cna=False, label="no-CNV")

    print()
    print("=" * 60)
    print("Control: no CNV, MLP head instead of linear (isolates head-vs-CNV effect)")
    print("=" * 60)
    _, a_mlp_test = run_config(data, use_cna=False, label="no-CNV-MLPhead", model_type="mlp")

    print()
    print("=" * 60)
    print("Config B: +CNV, raw (embedding + CNA)")
    print("=" * 60)
    b_train, b_test = run_config(data, use_cna=True, label="+CNV-raw")

    pca_results = {}
    for k in [10, 20, 50]:
        print()
        print("=" * 60)
        print(f"Config C: +CNV, PCA({k}) (embedding + CNA reduced to {k} components)")
        print("=" * 60)
        pca_results[k] = run_config(data, use_cna=True, label=f"+CNV-PCA{k}", pca_components=k)

    # Extra-embedding configs (D onward): each is a pkl produced by one of the
    # preprocessing/extract_*.py scripts, sharing the same patient/fold set as
    # DATA_PATH. Add a new (label, path, feature_key) entry here to compare a
    # new embedding -- no other code changes needed.
    extra_configs = [
        ("+CNV-GNN", GNN_DATA_PATH, "cna_gnn", "linear"),
        ("+CNV-SegGNN", SEGMENT_GNN_DATA_PATH, "cna_segment_gnn", "linear"),
        ("+CNV-GNN-CoxMTL", GNN_COXMTL_DATA_PATH, "cna_gnn_coxmtl", "linear"),
        ("+CNV-SegGNN-CoxMTL", SEGMENT_GNN_COXMTL_DATA_PATH, "cna_segment_gnn_coxmtl", "linear"),
        # Nonlinear-fusion variant of the current best (reconstruction-only grid GNN):
        # does a small MLP head extract more from the same frozen embedding than a linear probe?
        ("+CNV-GNN-MLPhead", GNN_DATA_PATH, "cna_gnn", "mlp"),
        ("+CNV-SegGNN-MLPhead", SEGMENT_GNN_DATA_PATH, "cna_segment_gnn", "mlp"),
        # Genes-as-nodes: real gene identities/coordinates (Ensembl GRCh37) with
        # genomic-neighbor + STRING PPI edges, instead of arbitrary bins or per-patient segments.
        ("+CNV-GeneGNN", GENE_GNN_DATA_PATH, "cna_gene_gnn", "linear"),
        ("+CNV-GeneGNN-MLPhead", GENE_GNN_DATA_PATH, "cna_gene_gnn", "mlp"),
        # Cross-modal CNV<->mRNA VAE (pretrained on the non-BRCA pan-cancer
        # cohort, frozen, applied to BRCA -- see preprocessing/cnv_mrna_vae_model.py):
        # a different prior than the GNN configs (expression-predictiveness
        # instead of genomic/PPI adjacency) on which CNV variation to keep.
        ("+CNV-CrossModalVAE", CNV_MRNA_VAE_DATA_PATH, "cnv_mrna_vae", "linear"),
        ("+CNV-CrossModalVAE-MLPhead", CNV_MRNA_VAE_DATA_PATH, "cnv_mrna_vae", "mlp"),
        # Fused model: the grid-GNN encoder trained jointly with a
        # cross-modal (CNV->expression) head, instead of training GNN and
        # cross-modal VAE separately and ensembling afterward.
        ("+CNV-GNN-CrossModal", GNN_CROSSMODAL_DATA_PATH, "cna_gnn_crossmodal", "linear"),
        ("+CNV-GNN-CrossModal-MLPhead", GNN_CROSSMODAL_DATA_PATH, "cna_gnn_crossmodal", "mlp"),
        # Same fused model, hidden_dim doubled 16->32 to relieve the
        # self-recon/cross-modal capacity competition seen in the hd16 run.
        ("+CNV-GNN-CrossModal-hd32", GNN_CROSSMODAL_HD32_DATA_PATH, "cna_gnn_crossmodal_hd32", "linear"),
        ("+CNV-GNN-CrossModal-hd32-MLPhead", GNN_CROSSMODAL_HD32_DATA_PATH, "cna_gnn_crossmodal_hd32", "mlp"),
        # Same cross-modal VAE recipe as +CNV-CrossModalVAE, swapping mRNA
        # for methylation / mutation as the second modality.
        ("+CNV-MethylationVAE", CNV_METHYLATION_VAE_DATA_PATH, "cnv_methylation_vae", "linear"),
        ("+CNV-MethylationVAE-MLPhead", CNV_METHYLATION_VAE_DATA_PATH, "cnv_methylation_vae", "mlp"),
        ("+CNV-MutationVAE", CNV_MUTATION_VAE_DATA_PATH, "cnv_mutation_vae", "linear"),
        ("+CNV-MutationVAE-MLPhead", CNV_MUTATION_VAE_DATA_PATH, "cnv_mutation_vae", "mlp"),
        # Fused GNN+mutation, mirroring +CNV-GNN-CrossModal's recipe but with
        # the modality that actually won standalone (mutation, not mRNA).
        ("+CNV-GNN-CrossModal-Mutation", GNN_CROSSMODAL_MUTATION_DATA_PATH, "cna_gnn_crossmodal_mutation", "linear"),
        ("+CNV-GNN-CrossModal-Mutation-MLPhead", GNN_CROSSMODAL_MUTATION_DATA_PATH, "cna_gnn_crossmodal_mutation", "mlp"),
    ]
    extra_results = {}
    for label, path, key, model_type in extra_configs:
        if not os.path.exists(path):
            print(f"\n(skipping {label}: {path} not found)")
            continue
        print()
        print("=" * 60)
        print(f"Config: {label} (embedding + {key}, {model_type} head)")
        print("=" * 60)
        with open(path, "rb") as f:
            extra_data = pickle.load(f)
        assert list(extra_data["patient_ids"]) == list(data["patient_ids"]), \
            f"{path} must use the identical patient/fold set as the base pkl for a fair comparison"
        print(f"{key} embedding dim = {extra_data[key].shape[1]}")
        _, extra_test = run_config_extra_features(
            extra_data["embedding"], extra_data[key], extra_data["e"], extra_data["t"], extra_data["fold"],
            label=label, model_type=model_type,
        )
        extra_results[label] = extra_test

    extra_results["no-CNV-MLPhead"] = a_mlp_test

    # Ensemble: grid-GNN and gene-GNN showed complementary fold-wise strengths
    # (grid wins folds 1/4/5, gene-graph wins folds 2/3 -- notably fold 2, the
    # historically hardest one). Concatenate both frozen embeddings alongside
    # the SeNMo embedding into one MLP head -- no new pretraining needed.
    if os.path.exists(GNN_DATA_PATH) and os.path.exists(GENE_GNN_DATA_PATH):
        print()
        print("=" * 60)
        print("Config: +CNV-GNN+GeneGNN-MLPhead (embedding + cna_gnn + cna_gene_gnn, mlp head)")
        print("=" * 60)
        with open(GNN_DATA_PATH, "rb") as f:
            gnn_data = pickle.load(f)
        with open(GENE_GNN_DATA_PATH, "rb") as f:
            gene_gnn_data = pickle.load(f)
        assert list(gnn_data["patient_ids"]) == list(gene_gnn_data["patient_ids"]) == list(data["patient_ids"])
        combined = np.concatenate([gnn_data["cna_gnn"], gene_gnn_data["cna_gene_gnn"]], axis=1)
        _, combo_test = run_config_extra_features(
            gnn_data["embedding"], combined, gnn_data["e"], gnn_data["t"], gnn_data["fold"],
            label="+CNV-GNN+GeneGNN-MLPhead", model_type="mlp",
        )
        extra_results["+CNV-GNN+GeneGNN-MLPhead"] = combo_test

    # Prediction-level ensemble: feature concatenation made things worse
    # (0.7789, below either model alone), likely because the small MLP head
    # gets dominated by whichever feature block has larger/more consistent
    # gradients rather than adaptively blending both. Instead, train the two
    # MLP-head models completely independently per fold (each exactly
    # reproducing its own standalone config above) and average their risk
    # scores post-hoc -- standardized first, since Cox-model risk outputs
    # have an arbitrary scale that can differ between models.
    if os.path.exists(GNN_DATA_PATH) and os.path.exists(GENE_GNN_DATA_PATH):
        print()
        print("=" * 60)
        print("Config: +CNV-GNN+GeneGNN-PredEnsemble (average standardized risk scores of the two independent MLP models)")
        print("=" * 60)
        with open(GNN_DATA_PATH, "rb") as f:
            gnn_data = pickle.load(f)
        with open(GENE_GNN_DATA_PATH, "rb") as f:
            gene_gnn_data = pickle.load(f)
        assert list(gnn_data["patient_ids"]) == list(gene_gnn_data["patient_ids"]) == list(data["patient_ids"])

        senmo_embedding = gnn_data["embedding"]
        e_all, t_all, fold_all = gnn_data["e"], gnn_data["t"], gnn_data["fold"]
        ens_train_cs, ens_test_cs = [], []
        for fold in range(1, NUM_FOLDS + 1):
            test_mask = fold_all == fold
            train_mask = ~test_mask

            def fold_features(extra):
                scaler = StandardScaler()
                extra_train = scaler.fit_transform(extra[train_mask])
                extra_test = scaler.transform(extra[test_mask])
                X_train = np.concatenate([senmo_embedding[train_mask], extra_train], axis=1)
                X_test = np.concatenate([senmo_embedding[test_mask], extra_test], axis=1)
                return X_train, X_test

            results_by_model = []
            for extra in (gnn_data["cna_gnn"], gene_gnn_data["cna_gene_gnn"]):
                X_train, X_test = fold_features(extra)
                _, _, train_risk, test_risk = train_one_fold(
                    X_train, e_all[train_mask], t_all[train_mask],
                    X_test, e_all[test_mask], t_all[test_mask],
                    model_type="mlp",
                )
                risk_scaler = StandardScaler()
                train_risk_std = risk_scaler.fit_transform(train_risk.reshape(-1, 1)).reshape(-1)
                test_risk_std = risk_scaler.transform(test_risk.reshape(-1, 1)).reshape(-1)
                results_by_model.append((train_risk_std, test_risk_std))

            ens_train_risk = np.mean([r[0] for r in results_by_model], axis=0)
            ens_test_risk = np.mean([r[1] for r in results_by_model], axis=0)
            ens_train_c = concordance_index(t_all[train_mask], -ens_train_risk, e_all[train_mask])
            ens_test_c = concordance_index(t_all[test_mask], -ens_test_risk, e_all[test_mask])
            ens_train_cs.append(ens_train_c)
            ens_test_cs.append(ens_test_c)
            print(f"[+CNV-GNN+GeneGNN-PredEnsemble] fold {fold}: train C-index={ens_train_c:.4f}  "
                  f"test C-index={ens_test_c:.4f} (n_train={train_mask.sum()} n_test={test_mask.sum()})")

        print(f"[+CNV-GNN+GeneGNN-PredEnsemble] mean train C-index = {np.mean(ens_train_cs):.4f} +/- {np.std(ens_train_cs):.4f}")
        print(f"[+CNV-GNN+GeneGNN-PredEnsemble] mean test  C-index = {np.mean(ens_test_cs):.4f} +/- {np.std(ens_test_cs):.4f}")
        extra_results["+CNV-GNN+GeneGNN-PredEnsemble"] = ens_test_cs

    def run_pred_ensemble(label, components):
        """Generalized version of the GNN+GeneGNN-PredEnsemble above: average
        the standardized risk scores of independently-trained MLP models,
        one per (data_path, feature_key) component."""
        datasets = []
        for path, key in components:
            with open(path, "rb") as f:
                d = pickle.load(f)
            assert list(d["patient_ids"]) == list(data["patient_ids"])
            datasets.append((d, key))

        senmo_embedding, e_all, t_all, fold_all = datasets[0][0]["embedding"], datasets[0][0]["e"], datasets[0][0]["t"], datasets[0][0]["fold"]
        ens_test_cs = []
        for fold in range(1, NUM_FOLDS + 1):
            test_mask = fold_all == fold
            train_mask = ~test_mask
            results_by_model = []
            for d, key in datasets:
                scaler = StandardScaler()
                extra_train = scaler.fit_transform(d[key][train_mask])
                extra_test = scaler.transform(d[key][test_mask])
                X_train = np.concatenate([senmo_embedding[train_mask], extra_train], axis=1)
                X_test = np.concatenate([senmo_embedding[test_mask], extra_test], axis=1)
                _, _, train_risk, test_risk = train_one_fold(
                    X_train, e_all[train_mask], t_all[train_mask],
                    X_test, e_all[test_mask], t_all[test_mask],
                    model_type="mlp",
                )
                risk_scaler = StandardScaler().fit(train_risk.reshape(-1, 1))
                results_by_model.append(risk_scaler.transform(test_risk.reshape(-1, 1)).reshape(-1))
            ens_test_risk = np.mean(results_by_model, axis=0)
            ens_test_c = concordance_index(t_all[test_mask], -ens_test_risk, e_all[test_mask])
            ens_test_cs.append(ens_test_c)
            print(f"[{label}] fold {fold}: test C-index={ens_test_c:.4f} (n_test={test_mask.sum()})")
        print(f"[{label}] mean test  C-index = {np.mean(ens_test_cs):.4f} +/- {np.std(ens_test_cs):.4f}")
        extra_results[label] = ens_test_cs

    # Mutation turned out to be a much stronger cross-modal partner for CNV
    # than mRNA or methylation (+CNV-MutationVAE-MLPhead alone: 0.7938, vs.
    # ~0.775-0.779 for the mRNA/methylation variants) -- plausibly because
    # CNV and mutation are both DNA-level alterations (shared driver
    # genes/pathways, LOH patterns) rather than the noisier, more indirect
    # CNV->expression relationship. Ensembling it with the grid-GNN gives
    # this project's best result so far.
    if os.path.exists(GNN_DATA_PATH) and os.path.exists(CNV_MUTATION_VAE_DATA_PATH):
        print()
        print("=" * 60)
        print("Config: +CNV-GNN+MutationVAE-PredEnsemble")
        print("=" * 60)
        run_pred_ensemble("+CNV-GNN+MutationVAE-PredEnsemble",
                           [(GNN_DATA_PATH, "cna_gnn"), (CNV_MUTATION_VAE_DATA_PATH, "cnv_mutation_vae")])

    if os.path.exists(GNN_DATA_PATH) and os.path.exists(GENE_GNN_DATA_PATH) and os.path.exists(CNV_MUTATION_VAE_DATA_PATH):
        print()
        print("=" * 60)
        print("Config: +CNV-GNN+GeneGNN+MutationVAE-PredEnsemble")
        print("=" * 60)
        run_pred_ensemble("+CNV-GNN+GeneGNN+MutationVAE-PredEnsemble",
                           [(GNN_DATA_PATH, "cna_gnn"), (GENE_GNN_DATA_PATH, "cna_gene_gnn"),
                            (CNV_MUTATION_VAE_DATA_PATH, "cnv_mutation_vae")])

    print()
    print("=" * 60)
    print("SUMMARY (test C-index)")
    print("=" * 60)
    col_width = 14
    header = f"{'Fold':<6}{'no-CNV':<12}{'+CNV-raw':<12}" + "".join(f"+CNV-PCA{k:<{col_width-8}}" for k in pca_results)
    header += "".join(f"{label:<{col_width}}" for label in extra_results)
    print(header)
    for i in range(NUM_FOLDS):
        row = f"{i+1:<6}{a_test[i]:<12.4f}{b_test[i]:<12.4f}"
        row += "".join(f"{pca_results[k][1][i]:<{col_width}.4f}" for k in pca_results)
        row += "".join(f"{extra_results[label][i]:<{col_width}.4f}" for label in extra_results)
        print(row)
    mean_row = f"{'Mean':<6}{np.mean(a_test):<12.4f}{np.mean(b_test):<12.4f}"
    mean_row += "".join(f"{np.mean(pca_results[k][1]):<{col_width}.4f}" for k in pca_results)
    mean_row += "".join(f"{np.mean(extra_results[label]):<{col_width}.4f}" for label in extra_results)
    print(mean_row)
    std_row = f"{'Std':<6}{np.std(a_test):<12.4f}{np.std(b_test):<12.4f}"
    std_row += "".join(f"{np.std(pca_results[k][1]):<{col_width}.4f}" for k in pca_results)
    std_row += "".join(f"{np.std(extra_results[label]):<{col_width}.4f}" for label in extra_results)
    print(std_row)


if __name__ == "__main__":
    main()
