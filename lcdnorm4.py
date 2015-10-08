"""Local Normalization Operations"""

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Giovani Chiachia <giovani.chiachia@gmail.com>
#
# License: BSD

__all__ = ['lcdnorm4']

import numpy as np
from skimage.util.shape import view_as_windows

EPSILON = 1e-4
DEFAULT_STRIDE = 1
DEFAULT_THRESHOLD = 1.0
DEFAULT_STRETCH = 1.0
DEFAULT_CONTRAST = True
DEFAULT_DIVISIVE = True


def lcdnorm4(arr_in, neighborhood,
             contrast=DEFAULT_CONTRAST,
             divisive=DEFAULT_DIVISIVE,
             stretch=DEFAULT_STRETCH,
             threshold=DEFAULT_THRESHOLD,
             stride=DEFAULT_STRIDE, arr_out=None):
    """4D Local Contrast Divisive Normalization

    XXX: docstring
    """

    assert arr_in.ndim == 4
    assert len(neighborhood) == 2
    assert isinstance(contrast, bool)
    assert isinstance(divisive, bool)
    assert contrast or divisive

    in_imgs, inh, inw, ind = arr_in.shape

    nbh, nbw = neighborhood
    assert nbh <= inh
    assert nbw <= inw

    nb_size = 1. * nbh * nbw * ind

    if arr_out is not None:
        assert arr_out.dtype == arr_in.dtype
        assert arr_out.shape == (in_imgs,
                                 1 + (inh - nbh) / stride,
                                 1 + (inw - nbw) / stride,
                                 ind)

    # -- prepare arr_out
    lys = nbh / 2
    lxs = nbw / 2
    rys = (nbh - 1) / 2
    rxs = (nbw - 1) / 2
    _arr_out = arr_in[:, lys:inh-rys, lxs:inw-rxs][::stride, ::stride]

    # -- Contrast Normalization
    if contrast:

        # -- local sums
        arr_sum = arr_in.sum(-1)
        arr_sum = view_as_windows(arr_sum, (1,
                                            1, nbw)).sum(-1)[:, :, ::stride,
                                                             0, 0]
        arr_sum = view_as_windows(arr_sum, (1,
                                            nbh, 1)).sum(-2)[:, ::stride, :, 0]

        # -- remove the mean
        _arr_out = _arr_out - arr_sum / nb_size

    # -- Divisive (gain) Normalization
    if divisive:

        # -- local sums of squares
        arr_ssq = (arr_in ** 2.0).sum(-1)
        arr_ssq = view_as_windows(arr_ssq, (1,
                                            1, nbw)).sum(-1)[:, :, ::stride,
                                                             0, 0]
        arr_ssq = view_as_windows(arr_ssq, (1,
                                            nbh, 1)).sum(-2)[:, ::stride, :, 0]

        # -- divide by the euclidean norm
        if contrast:
            l2norms = (arr_ssq - (arr_sum ** 2.0) / nb_size)
        else:
            l2norms = arr_ssq

        np.putmask(l2norms, l2norms < 0., 0.)
        l2norms = np.sqrt(l2norms) + EPSILON

        if stretch != 1:
            _arr_out *= stretch
            l2norms *= stretch

        np.putmask(l2norms, l2norms < (threshold + EPSILON), 1.0)

        _arr_out = _arr_out / l2norms

    if arr_out is not None:
        arr_out[:] = _arr_out
    else:
        arr_out = _arr_out

    assert arr_out.shape[0] == in_imgs

    return arr_out