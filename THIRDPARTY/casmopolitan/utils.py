def spearman(pred, target) -> float:
    """Compute the spearman correlation coefficient between prediction and target"""
    from scipy import stats
    coef_val, p_val = stats.spearmanr(pred, target)
    return coef_val


def pearson(pred, target) -> float:
    from scipy import stats
    coef_val, p_val = stats.pearsonr(pred, target)
    return coef_val


def negative_log_likelihood(pred, pred_std, target) -> float:
    """Compute the negative log-likelihood on the validation dataset"""
    from scipy.stats import norm
    import numpy as np
    n = pred.shape[0]
    res = 0.
    for i in range(n):
        res += (
            np.log(norm.pdf(target[i], pred[i], pred_std[i])).sum()
        )
    return -res


def get_dim_info(n_categories):
    dim_info = []
    offset = 0
    for i, cat in enumerate(n_categories):
        dim_info.append(list(range(offset, offset + cat)))
        offset += cat
    return dim_info
