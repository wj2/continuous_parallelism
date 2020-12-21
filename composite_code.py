
import numpy as np
import general.rf_models as rfm
import itertools as it
import scipy.stats as sts

def emp_variance(samps):
    return np.sum(np.var(samps, axis=0))

def l2_scaling(cp, p):
    return np.sqrt(cp/p)

def make_pwr_resp(inp_distrib, resp_func, target_pwr,
                  pwr_func=emp_variance, pwr_adjust=l2_scaling,
                  n_samples=10**5):
    samps = inp_distrib.rvs(n_samples)
    rep = resp_func(samps)
    pwr = pwr_func(rep)
    factor = pwr_adjust(target_pwr, pwr)
    new_resp_func = lambda x: factor*resp_func(x)
    return new_resp_func

def make_random_rotation(out_dim, in_dim):
    if out_dim < in_dim:
        raise Warning('information may be destroyed by combining transform '
                      'because the input dimension is greater than the output '
                      'dimension ({} > {})'.format(in_dim, out_dim))
    r = sts.multivariate_normal(0, 1).rvs((out_dim, in_dim))
    r_norm = np.expand_dims(np.sqrt(np.sum(r**2, axis=0)), 0)
    return r/r_norm

class CombinedCode(object):

    def __init__(self, n_neurs, linear_pwr, nonlinear_pwr, rf_width=None,
                 nonlin_components=10, first_rf_cent=-1, last_rf_cent=1,
                 input_distribution=None, rotation=True, inp_dim=2, rf_scale=1,
                 rf_baseline=0, noise_distrib=None, noise_variance=1):
        """ 
        A code with linear and nonlinear, receptive field components.

        Parameters
        ----------
        n_neurs : int
            The number of neurons to include in the combined representation
            only applied if rotation is True
        linear_pwr : float
            The power given to the linear representation -- power is quantified
            by pwr_quant and rescaling is done by pwr_rescale.
        nonlinear_pwr : float
            The power given to the nonlinear, RF representaton -- power is 
            quantified in the same way as above.
        rf_width : float or None, optional
            The width of receptive fields -- default is None. If None, then 
            receptive fields are given widths so that they contain roughly the
            same input distribution probability mass as each other. If a value
            is given, then they are equally spaced between the provided first 
            and last RF centers and all have the given width. 
        nonlin_components : int, optional
            The number of components along each dimension of the nonlinear 
            representation. The total number of nonlinear components is 
            nonline_components**inp_dim. 
        first_rf_cent : int, optional
            The position in stimulus space of the first RF center if width is 
            given.
        last_rf_cent : int, optional
            The position in stimulus space of the last RF center if width is 
            given.
        input_distribution : distribution object, optional
            If not given, distribution is an isotropic Gaussian. If given, must 
            be an iterable, and implement dim property and rvs method. 
        rotation : bool, optional
            If True, the representation will be random rotated, producing a 
            linear mixture of the linear and nonlinear components.
        inp_dim : int
            If input_distribution is None, this is the dimensionality of the
            isotropic Gaussian used as the input distribution. 
        """
        if input_distribution is None:
            input_distribution = sts.multivariate_normal(np.zeros(inp_dim), 1)
            id_list = (sts.norm(0, 1),)*inp_dim
        else:
            id_list = input_distribution
        self.inp_distrib = input_distribution
        self.inp_dim = input_distribution.dim
        self.p_l = linear_pwr
        self.p_n = nonlinear_pwr
        if rf_width is None:
            out = rfm.get_distribution_gaussian_resp_func(nonlin_components,
                                                          id_list,
                                                          scale=rf_scale,
                                                          baseline=rf_baseline)
            rfs, _, cents, wids = out
        else:
            c_i = np.linspace(first_rf_cent, last_rf_cent, nonlin_components)
            cents = list(it.product(c_i, repeat=inp_dim))
            wids = np.ones_like(cents)*rf_width
            out = rfm.make_gaussian_vector_rf(cents, wids, rf_scale,
                                              rf_baseline)
            rfs, _ = out
        self.rf_func = rfs
        self.rf_cents = cents
        self.rf_wids = wids
        self.rf_dim = len(cents)

        ident_func = lambda x: x
        self.nonlin_func = make_pwr_resp(input_distribution, rfs, self.p_n)
        self.lin_func = make_pwr_resp(input_distribution, ident_func, self.p_l)
        rep_dim = self.rf_dim + inp_dim
        if rotation:
            self.comb_transform = make_random_rotation(n_neurs, rep_dim)
        else:
            n_neurs = rep_dim
            self.comb_transform = None
        if noise_distrib is None:
            noise_distrib = sts.multivariate_normal(np.zeros(n_neurs),
                                                    noise_variance)
        self.noise_distrib = noise_distrib

    def stim_resp(self, x, add_noise=True):
        """
        Get representation of stimuli x in the code. 

        Parameters
        ----------
        x : ndarray
            Array of stimuli with shape (n, inp_dim) where n is the number of 
            stimuli. If shape of x has length 1, will be assumed to be a single
            stimulus.
        add_noise : bool, optional
            Whether or not to add noise sampled from the noise distribution of
            the model (default True). 

        Returns
        -------
        resp : ndarray
            Array of shape (n, n_neurs) -- the representations. 
        """
        x = np.array(x)
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        nl_resp = self.nonlin_func(x)
        l_resp = self.lin_func(x)
        resp_full = np.concatenate((nl_resp, l_resp), axis=1)
        if self.comb_transform is not None:
            resp_full = np.dot(resp_full, self.comb_transform.T)
        if add_noise:
            resp_full = resp_full + self.noise_distrib.rvs(x.shape[0])
        return resp_full

    def sample_representations(self, n, add_noise=True):
        """
        Sample n representations from the code. 

        Parameters
        ----------
        n : int
            The number of representations to sample.
        add_noise : bool, optional
            Whether or not to add noise sampled from the noise distribution of
            the model (default True). 

        Returns
        -------
        stim : ndarray
            Array of shape (n, inp_dim) -- the sampled stimuli.
        resp : ndarray
            Array of shape (n, n_neurs) -- the representations. 
        """
        stim = self.inp_distrib.rvs(n)
        resp = self.stim_resp(stim, add_noise=add_noise)
        return stim, resp
