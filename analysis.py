
import numpy as np
import scipy.stats as sts
import sklearn.linear_model as sklm
import sklearn.decomposition as skd

import general.utility as u
import continuous_parallelism.composite_code as cp

def compute_code_metrics(code, metrics, train_pt, test_pt, distr_var,
                         n_pts=1000, with_noise=True,
                         distr=sts.multivariate_normal,
                         **lr_kwargs):
    train_stim = distr(train_pt, distr_var).rvs(n_pts)
    test_stim = distr(test_pt, distr_var).rvs(n_pts)
    train_reps = code.stim_resp(train_stim, add_noise=with_noise)
    test_reps = code.stim_resp(test_stim, add_noise=with_noise)
    out = list(m(train_stim, train_reps, test_stim, test_reps, code=code,
                 **lr_kwargs)
               for m in metrics)
    return out

def compute_cosine_parallelism(train_stim, train_reps, test_stim, test_reps,
                               code=None, **lr_kwargs):
    """
    Computes the cosine parallelism between representation in the neighborhood
    of two points. The contract of this function matches the contract for 
    metrics -- and should not be changed without changing the contract for all
    metrics. 

    Parameters
    ----------
    train_stim : array
       An array with stimuli sampled from around a single point in the stimulus
       space. 
    train_reps : array
       Corresponding array of the representations of the points in train_stim.
    test_stim : array
       Stimli as above, sampled from around the test point. 
    test_reps : array
       Representations as above. 
    code : CompositeCode, optional
       The code object giving rise to the points above. The code object itself
       is only required by some metrics.

    Returns
    -------
    out, float
       The mean parallelism score across dimensions for this code. 
    """
    lr = sklm.Ridge(**lr_kwargs)
    lr.fit(train_reps, train_stim)
    lr_test = sklm.Ridge(**lr_kwargs)
    lr_test.fit(test_reps, test_stim)
    para = u.cosine_similarity(lr.coef_, lr_test.coef_)
    return np.mean(para)

def compute_ccgp(train_stim, train_reps, test_stim, test_reps,
                 code=None, **lr_kwargs):
    lr = sklm.Ridge(**lr_kwargs)
    lr.fit(train_reps, train_stim)
    ccgp = lr.score(test_reps, test_stim)
    return ccgp

def compute_dimensionality(*args, code=None, n_reps=10000, **lr_kwargs):
    if code is None:
        raise IOError('cannot compute dimensionality without code')
    _, reps = code.sample_representations(n_reps, add_noise=False)
    p = skd.PCA()
    p.fit(reps)
    eigs = p.explained_variance_
    pr = np.sum(eigs)**2/np.sum(eigs**2)
    return pr

default_metrics = (compute_ccgp, compute_cosine_parallelism,
                   compute_dimensionality)
def compute_tradeoff(code_class, total_pwr, n_neurs, train_pt, test_pt,
                     samples_var, n_trades=100, metrics=default_metrics,
                     code_kwargs=None, **kwargs):
    if code_kwargs is None:
        code_kwargs = {}
    ln_tradeoff = np.linspace(0, 1, n_trades)
    outs = []
    for i, lnt in enumerate(ln_tradeoff):
        l_p = lnt*total_pwr
        n_p = (1 - lnt)*total_pwr
        code = cp.CombinedCode(n_neurs, l_p, n_p, **code_kwargs)
        out = compute_code_metrics(code, metrics, train_pt, test_pt,
                                   samples_var, **kwargs)
        outs.append(out)
    return ln_tradeoff, np.array(outs)
