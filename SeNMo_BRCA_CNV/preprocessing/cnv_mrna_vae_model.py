"""
Cross-modal CNV<->(mRNA|methylation|mutation) VAE: two modality-specific
encoders sharing one latent space, trained with self- and cross-
reconstruction plus a latent-alignment term, so that the CNV encoder alone
(the only one available at BRCA inference time, since we always have CNV
but want to add other-modality-informed structure to it) is pulled toward
encoding CNV variation that is actually reflected in the second modality,
rather than passenger/technical noise. Originally built for CNV<->mRNA;
generalized via `modality_b_dist` to also cover methylation (continuous
beta values in [0,1], still Gaussian/MSE) and mutation (binary per-gene
presence, Bernoulli/BCE -- MSE would barely penalize missing the rare 1s
in such an imbalanced binary target).

Deliberately simpler than TMO-Net's TMO_Net_model.py cross-modal machinery
(no product-of-experts fusion, no adversarial modality discriminator): with
one pair of modalities and a latent-alignment loss doing the same alignment
job directly, that extra machinery is complexity this problem doesn't need,
and it's exactly the kind of thing that can destabilize training on a
biomedical cohort this size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.3):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(dims[-1], latent_dim)
        self.logvar = nn.Linear(dims[-1], latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu(h)
        # Clamped for the same reason as TMO_Net_model.py's encoder: an
        # unclamped log-var can drift to extremes over many epochs and blow
        # up exp(log_var) in the KL term / reparameterization.
        logvar = torch.clamp(self.logvar(h), min=-10, max=10)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        dims = [latent_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], output_dim)

    def forward(self, z):
        return self.out(self.backbone(z))


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def kl_divergence(mu, logvar):
    """KL(N(mu, exp(logvar)) || N(0, 1)), summed over latent dims, mean over batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()


class CrossModalVAE(nn.Module):
    def __init__(self, cnv_dim, mrna_dim, latent_dim=32, hidden_dims=(256, 64), dropout=0.3):
        super().__init__()
        self.encoder_cnv = Encoder(cnv_dim, hidden_dims, latent_dim, dropout)
        self.encoder_mrna = Encoder(mrna_dim, hidden_dims, latent_dim, dropout)
        decoder_hidden = tuple(reversed(hidden_dims))
        self.decoder_cnv = Decoder(latent_dim, cnv_dim, decoder_hidden, dropout)
        self.decoder_mrna = Decoder(latent_dim, mrna_dim, decoder_hidden, dropout)

    def forward(self, cnv, mrna):
        mu_cnv, logvar_cnv = self.encoder_cnv(cnv)
        mu_mrna, logvar_mrna = self.encoder_mrna(mrna)
        z_cnv = reparameterize(mu_cnv, logvar_cnv)
        z_mrna = reparameterize(mu_mrna, logvar_mrna)

        recon_cnv_self = self.decoder_cnv(z_cnv)
        recon_mrna_self = self.decoder_mrna(z_mrna)
        # Cross-reconstruction: decode the OTHER modality from this modality's
        # own latent -- this is what forces encoder_cnv to keep only the CNV
        # variation that predicts expression.
        recon_mrna_cross = self.decoder_mrna(z_cnv)
        recon_cnv_cross = self.decoder_cnv(z_mrna)

        return {
            "mu_cnv": mu_cnv, "logvar_cnv": logvar_cnv,
            "mu_mrna": mu_mrna, "logvar_mrna": logvar_mrna,
            "recon_cnv_self": recon_cnv_self, "recon_mrna_self": recon_mrna_self,
            "recon_mrna_cross": recon_mrna_cross, "recon_cnv_cross": recon_cnv_cross,
        }

    def forward_losses(self, cnv, mrna, modality_b_dist="gaussian", cross_direction="both"):
        """Like compute_loss, but returns the raw, unweighted loss TENSORS
        (recon_self, recon_cross, align, kl) with no combination applied --
        lets a caller (e.g. an HPO search) impose its own weighting scheme
        without this class needing to know about it. compute_loss itself is
        left untouched so every existing pretrain_*.py script keeps working
        unchanged."""
        out = self(cnv, mrna)
        modality_b_loss_fn = F.mse_loss if modality_b_dist == "gaussian" else F.binary_cross_entropy_with_logits
        recon_self = F.mse_loss(out["recon_cnv_self"], cnv) + modality_b_loss_fn(out["recon_mrna_self"], mrna)

        a_to_b = modality_b_loss_fn(out["recon_mrna_cross"], mrna)
        b_to_a = F.mse_loss(out["recon_cnv_cross"], cnv)
        if cross_direction == "a_to_b":
            recon_cross = a_to_b
        elif cross_direction == "b_to_a":
            recon_cross = b_to_a
        elif cross_direction == "both":
            recon_cross = a_to_b + b_to_a
        else:
            raise ValueError(cross_direction)

        kl = kl_divergence(out["mu_cnv"], out["logvar_cnv"]) + kl_divergence(out["mu_mrna"], out["logvar_mrna"])
        align = F.mse_loss(out["mu_cnv"], out["mu_mrna"])
        return {"recon_self": recon_self, "recon_cross": recon_cross, "align": align, "kl": kl}

    def compute_loss(self, cnv, mrna, cross_weight=0.5, align_weight=0.5, kl_weight=0.01,
                      modality_b_dist="gaussian", cross_direction="both"):
        """cross_direction selects which cross-decode path(s) contribute to
        recon_cross -- 'a_to_b' (decode modality B from z_cnv only), 'b_to_a'
        (decode CNV from z_B only), or 'both' (default, original behavior).
        Used to ablate which direction of cross-reconstruction is actually
        doing the work of shaping the CNV encoder."""
        out = self(cnv, mrna)
        modality_b_loss_fn = F.mse_loss if modality_b_dist == "gaussian" else F.binary_cross_entropy_with_logits
        recon_self = F.mse_loss(out["recon_cnv_self"], cnv) + modality_b_loss_fn(out["recon_mrna_self"], mrna)

        a_to_b = modality_b_loss_fn(out["recon_mrna_cross"], mrna)  # CNV -> modality B
        b_to_a = F.mse_loss(out["recon_cnv_cross"], cnv)            # modality B -> CNV
        if cross_direction == "a_to_b":
            recon_cross = a_to_b
        elif cross_direction == "b_to_a":
            recon_cross = b_to_a
        elif cross_direction == "both":
            recon_cross = a_to_b + b_to_a
        else:
            raise ValueError(cross_direction)

        kl = kl_divergence(out["mu_cnv"], out["logvar_cnv"]) + kl_divergence(out["mu_mrna"], out["logvar_mrna"])
        # Pulls the two encoders' latents together directly, on top of what
        # cross-reconstruction already encourages -- cheap to add, and it's
        # the term most directly responsible for making mu_cnv alone (the
        # only thing available at BRCA inference time) resemble mu_mrna.
        # Zero this out (align_weight=0) to isolate cross-reconstruction
        # direction's own effect without this confound.
        align = F.mse_loss(out["mu_cnv"], out["mu_mrna"])

        loss = recon_self + cross_weight * recon_cross + align_weight * align + kl_weight * kl
        return loss, {
            "recon_self": recon_self.item(), "recon_cross": recon_cross.item(),
            "align": align.item(), "kl": kl.item(),
        }
