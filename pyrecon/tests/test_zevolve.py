import os
from pathlib import Path
import numpy as np

from pyrecon import IterativeFFTReconstruction, mpi, setup_logging
from pyrecon.utils import DistanceToRedshift, sky_to_cartesian, cartesian_to_sky
from utils import data_fn, randoms_fn, catalog_rec_fn, Catalog
from cosmoprimo.fiducial import DESI
from scipy.interpolate import InterpolatedUnivariateSpline


def get_clustering_positions_weights(data, distance):
    mask = (data['Z'] > 0.) & (data['Z'] < 10.)
    ra = data['RA'][mask]
    dec = data['DEC'][mask]
    dist = distance(data['Z'][mask])
    positions = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = data['Weight'][mask]
    return positions, weights, mask


def bias_evolution(z, tracer='QSO'):
    """
    Bias model fitted from DR1 unblinded data (the formula from Laurent et al. 2016 (1705.04718))
    """
    if tracer == 'QSO':
        alpha = 0.237
        beta = 2.328
    elif tracer == 'LRG':
        alpha = 0.209
        beta = 2.790
    elif tracer == 'ELG_LOPnotqso':
        alpha = 0.153
        beta = 1.541
    else:
        raise NotImplementedError(f'{tracer} not implemented.')
    return alpha * ((1+z)**2 - 6.565) + beta


def interpolate_f_bias(cosmo, tracer, zdependent=False):
    P0 = {'LRG': 8.9e3, 'QSO': 5.0e3}[tracer]
    if zdependent:
        z = np.linspace(0.0, 5.0, 10000)
        growth_rate = cosmo.growth_rate(z)
        bias = bias_evolution(z, tracer)
        distance = cosmo.comoving_radial_distance
        f_at_dist = InterpolatedUnivariateSpline(distance(z), growth_rate, k=3)
        bias_at_dist = InterpolatedUnivariateSpline(distance(z), bias, k=3)
        f_at_dist, bias_at_dist
    else:
        f_at_dist = {'LRG': 0.834, 'QSO': 0.928}[tracer]
        bias_at_dist = {'LRG': 2.0, 'QSO': 2.1}[tracer]
    return f_at_dist, bias_at_dist, P0


def interpolate_nbar(data, randoms, distance):
    from mockfactory import RedshiftDensityInterpolator
    alpha = data['Weight'].csum() / randoms['Weight'].csum()
    density = RedshiftDensityInterpolator(distance(randoms['Z']), weights=alpha * randoms.ones(), bins=distance(np.linspace(0., 3., 100)),
                                          fsky=0.01)
    return density


def test_ref(data_fn, randoms_fn, data_fn_rec, randoms_fn_rec, tracer='LRG', recon_weights=True, fmesh=True, bmesh=True):
    cosmo = DESI()
    data = Catalog.read(data_fn)
    randoms = Catalog.read(randoms_fn)
    f, bias, P0 = interpolate_f_bias(cosmo, tracer, zdependent=False)
    f_at_dist, bias_at_dist, P0 = interpolate_f_bias(cosmo, tracer, zdependent=True)
    nbar_at_dist =  interpolate_nbar(data, randoms, distance=cosmo.comoving_radial_distance)
    f = f_at_dist if fmesh else f
    bias = bias_at_dist if bmesh else bias
    for mode in ['std', 'fast']:
        if mode == 'std':
            recon = IterativeFFTReconstruction(f=f, bias=bias, positions=randoms['Position'], cellsize=7.,
                                            los='local', position_type='pos', dtype='f8')
            recon.assign_data(data['Position'], data['Weight'])
            recon.assign_randoms(randoms['Position'], randoms['Weight'])
            #recon.set_density_contrast(smoothing_radius=15.)
            #if recon_weights: recon.set_optimal_weights(**{'P0': P0, 'n_at_dist': nbar_at_dist})
            recon.set_density_contrast(smoothing_radius=15., kw_weights={'P0': P0, 'nbar': nbar_at_dist} if recon_weights else None)
            recon.run()
        else:
            recon = IterativeFFTReconstruction(f=f, bias=bias, data_positions=data['Position'],
                                            randoms_positions=randoms['Position'], cellsize=7.,
                                            smoothing_radius=15., kw_weights={'P0': P0, 'nbar': nbar_at_dist} if recon_weights else None,
                                            los='local', position_type='pos', dtype='f8')

        data['Position_rec'] = recon.read_shifted_positions(data['Position'])
        randoms['Position_rec'] = recon.read_shifted_positions(randoms['Position'])

        for cat, fn in zip([data, randoms], [data_fn_rec, randoms_fn_rec]):
            rec = recon.read_shifted_positions(cat['Position'])
            if 'Position_rec' in cat:
                if recon.mpicomm.rank == 0: print('Checking...')
                assert np.allclose(rec, cat['Position_rec'])
            else:
                cat['Position_rec'] = rec
            if fn is not None:
                cat.write(fn)


if __name__ == '__main__':

    setup_logging()

    data_fn_rec, randoms_fn_rec = [catalog_rec_fn(fn, 'zevolve') for fn in [data_fn, randoms_fn]]
    #test_ref(data_fn, randoms_fn, data_fn_rec, randoms_fn_rec)
    test_ref(data_fn_rec, randoms_fn_rec, None, None)