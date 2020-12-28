import glob
import math
import os

import numpy as np
from tqdm import tqdm

from preproc import Spectrum


def final_grid(TGM, spectra):
    tu, gu, mu = np.unique(TGM[:, 0]), np.unique(TGM[:, 1]), np.unique(TGM[:, 2])
    tgm = []
    spc = []
    var = []

    for i in tqdm(range(len(tu))):
        for j in range(len(gu)):
            for k in range(len(mu)):
                try:
                    node = grid_diagonal_index(TGM, i, j, k)[0]
                    diag_indexes = grid_diagonal_index(TGM, i, j, k)
                    tgm.append(TGM[node])
                    spc.append(spectra[node])

                    variation = spectra[diag_indexes[1]] - spectra[diag_indexes[0]]
                    var.append(variation)

                except:
                    var = None
    tgm = np.array(tgm)
    spc = np.array(spc)
    var = np.array(var)

    return tgm, spc, var


def grid_index(TGM, meshgrid_index):
    """
    Args:
    meshgrid_index: the meshgrid index of given parameters,

    Return:
    idx: the corresponding index in TGM grid
    """
    tu, gu, mu = np.unique(TGM[:, 0]), np.unique(TGM[:, 1]), np.unique(TGM[:, 2])
    t_ax, g_ax, m_ax = np.meshgrid(tu, gu, mu, sparse=False, indexing='ij')

    i, j, k = meshgrid_index[0], meshgrid_index[1], meshgrid_index[2]
    idx = np.where(np.all(TGM == np.array((t_ax[i, j, k], g_ax[i, j, k], m_ax[i, j, k])), axis=-1))[0]

    return idx


def meshgrid_diagonal_index(i, j, k):
    """
    Returns the meshgrid index of all nearest diagonal of the given node

    """
    dia_idx = np.array(([i, j, k], [i + 1, j + 1, k + 1], [i + 1, j + 1, k - 1], [i + 1, j - 1, k + 1],
                        [i + 1, j - 1, k - 1], [i - 1, j + 1, k + 1], [i - 1, j + 1, k - 1], [i - 1, j - 1, k + 1],
                        [i - 1, j - 1, k - 1]))
    return dia_idx


def grid_diagonal_index(TGM, i, j, k):
    _idx = []
    mgd_idx = meshgrid_diagonal_index(i, j, k)
    for _mgd_idx in mgd_idx:
        if _mgd_idx[0] < 41 and _mgd_idx[1] < 13 and _mgd_idx[2] < 6:
            if np.all(_mgd_idx == mgd_idx[0]) and grid_index(TGM, _mgd_idx).size == 0:
                break
            if (_mgd_idx >= 0).all():
                idx = grid_index(TGM, _mgd_idx)
                if idx.size != 0:
                    _idx.append(idx[0])
    _idx = np.array(_idx)
    return _idx


class CreateGrid(object):
    def __init__(self, path_to_grid: str, R_fin: float = 3500.):
        """
        Args:
        path_to_grid: local path of the grid where all the fits files are stored,
        R_fin: the resolution of the spectra at 6000A in the grid 

        """
        #TODO change the default path
        wl_NGC = np.loadtxt('./data/wl_NGC.txt')
        R_int = 10000.
        const = (2 * np.sqrt(2 * math.log(2)))
        init_sigma = 6000. / (const * R_int)
        final_sigma = 6000. / (const * R_fin)
        convolution_sigma = math.sqrt(final_sigma ** 2 - init_sigma ** 2)

        # path_to_grid = '/home/nitesh/nitesh/spectral_library/GSL/PHOENIX_1/'
        params = []
        fluxs = []
        os.chdir(path_to_grid)
        # i=0
        for file in tqdm(glob.glob("*.fits")):
            _spfile = Spectrum.load(file)
            _spfile.mode(Spectrum.Mode.LAMBDA)

            _new_spfile = _spfile.resample(flux_conserve=True, wave_start=wl_NGC[0] - 1.25, wave_end=wl_NGC[-1] + 1.25,
                                           wave_step=wl_NGC[1] - wl_NGC[0])

            # degradation of the spectra
            _new_spfile.smooth(sigma=convolution_sigma)
            _new_spfile.norm_to_mean()
            params.append(_spfile.param)
            fluxs.append(_new_spfile.flux)
            # i = i + 1
            # if i == 105: break
        _params = np.array(params)
        _spectra = np.array(fluxs)[:, 1:-1]
        self.wl = _new_spfile.wave[1:-1]
        #os.chdir('/home/nitesh/nitesh/PhD/GSL_Interpolator/')

        self.tgm, self.spc, self.var = final_grid(_params, _spectra)

    @property
    def params(self) -> np.ndarray:
        return self.tgm

    @property
    def spectra(self) -> np.ndarray:
        return self.spc

    @property
    def wave(self) -> np.ndarray:
        return self.wl

    @property
    def var_spectra(self) -> np.ndarray:
        return self.var


__all__ = {'CreateGrid'}
