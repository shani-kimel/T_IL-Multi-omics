"""Shared Cox partial-likelihood loss, matching the convention used elsewhere
in this project (train_senmo_brca.py's CoxLoss): unstratified (pooled across
cancer types), which is also how SeNMo's own pancancer_combined training
works -- consistent with prior art in this codebase rather than a new
convention."""
import torch


def cox_ptl_loss(risk_pred, survtime, censor):
    """risk_pred, survtime, censor: 1D tensors, same length."""
    n = len(survtime)
    R_mat = (survtime.reshape(1, -1) >= survtime.reshape(-1, 1)).float()
    theta = risk_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    return -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor)
