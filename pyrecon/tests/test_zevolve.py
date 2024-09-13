from pyrecon import IterativeFFTReconstruction, setup_logging
from pyrecon.utils import DistanceToRedshift, sky_to_cartesian, cartesian_to_sky
import mpytools as mpy
from cosmoprimo.fiducial import DESI
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import fitsio
from pathlib import Path


def read_data():
    data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version}'
    data_fn = Path(data_dir) / f'{tracer}_{region}_clustering.dat.fits'
    return fitsio.read(data_fn)

def read_randoms(idx=0):
    data_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/{version}'
    data_fn = Path(data_dir) / f'{tracer}_{region}_{idx}_clustering.ran.fits'
    return fitsio.read(data_fn)

def get_clustering_positions_weights(data):
    mask = (data['Z'] > zmin) & (data['Z'] < zmax)
    ra = data[mask]['RA']
    dec = data[mask]['DEC']
    dist = distance(data[mask]['Z'])
    pos = sky_to_cartesian(ra=ra, dec=dec, dist=dist)
    weights = data[mask]['WEIGHT']
    return pos, weights, mask

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

def interpolate_growth_bias():
    z = np.linspace(0.0, 5.0, 10000)
    growth_rate = cosmo.growth_rate(z)
    bias = bias_evolution(z)
    growth_at_dist = InterpolatedUnivariateSpline(distance(z), growth_rate, k=3)
    bias_at_dist = InterpolatedUnivariateSpline(distance(z), bias, k=3)
    return growth_at_dist, bias_at_dist

def interpolate_number_density():
    nz_dir = f'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/blinded/'
    nz_fn = Path(nz_dir) / f'{tracer}_{region}_nz.txt'
    data = np.genfromtxt(nz_fn)
    zmid = data[:, 0]
    nz = data[:, 3]
    n_at_dist = InterpolatedUnivariateSpline(distance(zmid), nz, k=3, ext=1)
    return n_at_dist

def run_recon(recon_weights=False, fmesh=False, bmesh=False):
    f = growth_at_dist if fmesh else growth_rate[tracer]
    bias = bias_at_dist if bmesh else bias
    if mpicomm.rank == 0:
        data = read_data()
        data_positions, data_weights, data_mask = get_clustering_positions_weights(data)
        data = data[data_mask]
    else:
        data_positions, data_weights = None, None
    recon = IterativeFFTReconstruction(f=f, bias=bias, positions=data_positions,
                                    los='local', cellsize=cellsize, boxpad=1.1,
                                    position_type='pos', dtype='f8', mpicomm=mpicomm,
                                    mpiroot=0, growth_at_dist=growth_at_dist)
    recon.assign_data(data_positions, data_weights)
    for i in range(nrand):
        if mpicomm.rank == 0:
            randoms = read_randoms(i)
            random_positions, random_weights, randoms_mask = get_clustering_positions_weights(randoms)
        else:
            random_positions, random_weights = None, None
        recon.assign_randoms(random_positions, random_weights)
    recon.set_density_contrast(smoothing_radius=smoothing_radius)
    if recon_weights:
        recon.set_optimal_weights(n_at_dist, P0)
    recon.run()
    data_positions_recon = recon.read_shifted_positions(data_positions)
    if mpicomm.rank == 0:
        dist, ra, dec = cartesian_to_sky(data_positions_recon)
        data['RA'], data['DEC'], data['Z'] = ra, dec, d2r(dist)
        if recon_weights and fmesh and bmesh:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/growth_bias_weights/{version}'
        elif fmesh and bmesh:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/growth_bias/{version}'
        elif recon_weights:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/weights/{version}'
        elif fmesh:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/growth/{version}'
        else:
            output_dir = f'/pscratch/sd/e/epaillas/recon_weights/vanilla/{version}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_fn = Path(output_dir) / f'{tracer}_{region}_clustering.IFTrecsym.dat.fits'
        fitsio.write(output_fn, data, clobber=True)
    for i in range(4):
        if mpicomm.rank == 0:
            randoms = read_randoms(i)
            randoms_positions, randoms_weights, randoms_mask = get_clustering_positions_weights(randoms)
            randoms = randoms[randoms_mask]
        else:
            randoms_positions, randoms_weights = None, None
        randoms_positions_recon = recon.read_shifted_positions(randoms_positions)
        if mpicomm.rank == 0:
            dist, ra, dec = cartesian_to_sky(randoms_positions_recon)
            randoms['RA'], randoms['DEC'], randoms['Z'] = ra, dec, d2r(dist)
            output_fn = Path(output_dir) / f'{tracer}_{region}_{i}_clustering.IFTrecsym.ran.fits'
            fitsio.write(output_fn, randoms, clobber=True)

setup_logging()
mpicomm = mpy.COMM_WORLD

bias = {'LRG': 2.0, 'QSO': 2.1}
P0 = {'LRG': 8.9e3, 'QSO': 5.0e3}
smoothing_radius = {'LRG': 15, 'QSO': 30}
growth_rate = {'LRG': 0.834, 'QSO': 0.928}

version = 'v1.2/unblinded'
tracer = 'QSO'
regions = ['NGC', 'SGC']
zmin, zmax = 0.8, 2.1
bias = bias[tracer]
P0 = P0[tracer]
smoothing_radius = smoothing_radius[tracer]
cellsize = 5.0
nrand = 18

cosmo = DESI()
distance = cosmo.comoving_radial_distance
d2r = DistanceToRedshift(distance)

growth_at_dist, bias_at_dist = interpolate_growth_bias()

for region in regions:
    n_at_dist = interpolate_number_density()

    # run_recon(recon_weights=False, fmesh=False)
    run_recon(recon_weights=False, fmesh=True, bmesh=True)
    # run_recon(recon_weights=True, fmesh=True, bmesh=True)
