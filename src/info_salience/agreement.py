import krippendorff
import numpy as np
from scipy import stats


def krippendorff_alpha(x1, x2, **krippendorff_args):
    if np.all(x1 == x2):
        return 1

    return krippendorff.alpha(reliability_data=np.vstack([x1, x2]), **krippendorff_args)


def spearman_rank_correlation(x1, x2):
    corr, _ = stats.spearmanr(x1, x2)
    return corr
