import os

import numpy as np
import fitsio


catalog_dir = '_catalogs'
data_fn = os.path.join(catalog_dir,'data.fits')
randoms_fn = os.path.join(catalog_dir,'randoms.fits')
bias = 2.0


def mkdir(dirname):
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def save_lognormal_catalogs(data_fn, randoms_fn, seed=42):
    from nbodykit.lab import cosmology, LogNormalCatalog, UniformCatalog
    from nbodykit.transform import CartesianToSky
    from nbodykit.utils import GatherArray

    def save_fits(cat, fn):
        array = np.empty(cat.size,dtype=[(col,cat[col].dtype,cat[col].shape[1:]) for col in cat.columns])
        for col in cat.columns: array[col] = cat[col].compute()
        array = GatherArray(array,comm=cat.comm)
        if cat.comm.rank == 0:
            fitsio.write(fn,array,clobber=True)

    redshift = 0.7
    cosmo = cosmology.Planck15.match(Omega0_m=0.3)
    Plin = cosmology.LinearPower(cosmo,redshift,transfer='CLASS')
    nbar = 3e-4
    BoxSize = 800
    catalog = LogNormalCatalog(Plin=Plin,nbar=nbar,BoxSize=BoxSize,Nmesh=256,bias=bias,seed=seed)
    #print(redshift,cosmo.scale_independent_growth_rate(redshift),cosmo.comoving_distance(redshift))

    offset = cosmo.comoving_distance(redshift) - BoxSize/2.
    offset = np.array([offset,0,0])
    catalog['Position'] += offset
    distance = np.sum(catalog['Position']**2,axis=-1)**0.5
    los = catalog['Position']/distance[:,None]
    catalog['Position'] += (catalog['VelocityOffset']*los).sum(axis=-1)[:,None]*los
    #mask = (catalog['Position'] >= offset) & (catalog['Position'] < offset + BoxSize)
    #catalog = catalog[np.all(mask,axis=-1)]
    catalog['NZ'] = nbar*np.ones(catalog.size,dtype='f8')
    catalog['Weight'] = np.ones(catalog.size,dtype='f8')
    catalog['RA'],catalog['DEC'],catalog['Z'] = CartesianToSky(catalog['Position'],cosmo)
    save_fits(catalog,data_fn)

    catalog = UniformCatalog(BoxSize=BoxSize,nbar=10*nbar,seed=seed)
    catalog['Position'] += offset
    catalog['Weight'] = np.ones(catalog.size,dtype='f8')
    catalog['NZ'] = nbar*np.ones(catalog.size,dtype='f8')
    catalog['RA'],catalog['DEC'],catalog['Z'] = CartesianToSky(catalog['Position'],cosmo)
    save_fits(catalog,randoms_fn)


def setup():
    mkdir(catalog_dir)
    save_lognormal_catalogs(data_fn,randoms_fn,seed=42)


class MemoryMonitor(object):
    """
    Class that monitors memory usage and clock, useful to check for memory leaks.

    >>> with MemoryMonitor() as mem:
            '''do something'''
            mem()
            '''do something else'''
    """

    def __init__(self, pid=None, msg=''):
        """
        Initalize :class:`MemoryMonitor` and register current memory usage.

        Parameters
        ----------
        pid : int, default=None
            Process identifier. If ``None``, use the identifier of the current process.

        msg : string, default=''
            Additional message.
        """
        import psutil
        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.mem = self.proc.memory_info().rss / 1e6
        self.time = time.time()
        self.msg = msg
        msg = 'using {:.3f} [Mb]'.format(self.mem)
        if self.msg:
            msg = '[{}] {}'.format(self.msg,msg)
        print(msg)

    def __enter__(self):
        """Enter context."""

    def __call__(self):
        """Update memory usage."""
        mem = self.proc.memory_info().rss / 1e6
        t = time.time()
        msg = 'using {:.3f} [Mb] (increase of {:.3f} [Mb]) after {:.3f} [s]'.format(mem,mem-self.mem,t-self.time)
        if self.msg:
            msg = '[{}] {}'.format(self.msg,msg)
        print(msg)
        self.mem = mem
        self.time = t

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()
