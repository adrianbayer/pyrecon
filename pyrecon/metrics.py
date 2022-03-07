"""
Routines to estimate reconstruction efficiency:

  - :class:`MeshFFTCorrelation`: correlation
  - :class:`MeshFFTTransfer`: transfer
  - :class:`MeshFFTPropagator`: propagator

This requires the following packages:

  - pmesh
  - pypower, see https://github.com/adematti/pypower

"""
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline
from pypower import MeshFFTPower, CatalogMesh, ParticleMesh, ArrayMesh, PowerSpectrumWedges

from .utils import BaseClass
from . import utils


class BasePowerRatio(BaseClass):
    """
    Base template class to compute power ratios.
    Specific statistic should extend this class.
    """
    _coords_names = ['k', 'mu']
    _result_names = ['num', 'denom']
    _power_names = ['ratio']
    _attrs = []

    def get_ratio(self, complex=False, **kwargs):
        """
        Return power spectrum ratio, computed using various options.

        Parameters
        ----------
        complex : bool, default=False
            Whether (``True``) to return the ratio of complex power spectra,
            or (``False``) return the ratio of their real part only.

        kwargs : dict
            Optionally, arguments for :meth:`BasePowerSpectrumStatistics.get_power`.

        Results
        -------
        ratio : array
        """
        return self.num.get_power(complex=complex, **kwargs)/self.denom.get_power(complex=complex, **kwargs)

    @property
    def ratio(self):
        """Power spectrum ratio."""
        return self.get_ratio()

    def __call__(self, k=None, mu=None, return_k=False, return_mu=False, complex=False, **kwargs):
        r"""
        Return power spectrum ratio, optionally performing linear interpolation over :math:`k` and :math:`\mu`.

        Parameters
        ----------
        k : float, array, default=None
            :math:`k` where to interpolate the power spectrum.
            Values outside :attr:`kavg` are set to the first/last ratio value;
            outside :attr:`edges[0]` to nan.
            Defaults to :attr:`kavg`.

        mu : float, array, default=None
            :math:`\mu` where to interpolate the power spectrum.
            Values outside :attr:`muavg` are set to the first/last ratio value;
            outside :attr:`edges[1]` to nan.
            Defaults to :attr:`muavg`.

        return_k : bool, default=False
            Whether (``True``) to return :math:`k`-modes (see ``k``).
            If ``None``, return :math:`k`-modes if ``k`` is ``None``.

        return_mu : bool, default=False
            Whether (``True``) to return :math:`\mu`-modes (see ``mu``).
            If ``None``, return :math:`\mu`-modes if ``mu`` is ``None``.

        complex : bool, default=False
            Whether (``True``) to return the ratio of complex power spectra,
            or (``False``) return the ratio of their real part only.

        kwargs : dict
            Other arguments for :meth:`get_ratio`.

        Returns
        -------
        k : array
            Optionally, :math:`k`-modes.

        mu : array
            Optionally, :math:`\mu`-modes.

        ratio : array
            (Optionally interpolated) power spectrum ratio.
        """
        power = self.ratio
        kavg, muavg = self.kavg, self.muavg
        if return_k is None:
            return_k = k is None
        if return_mu is None:
            return_mu = mu is None
        if k is None and mu is None:
            if return_k:
                if return_mu:
                    return kavg, muavg, power
            return power
        if k is None: k = kavg
        if mu is None: mu = muavg
        mask_finite_k, mask_finite_mu = ~np.isnan(kavg), ~np.isnan(muavg)
        kavg, muavg, power = kavg[mask_finite_k], muavg[mask_finite_mu], power[np.ix_(mask_finite_k, mask_finite_mu)]
        k, mu = np.asarray(k), np.asarray(mu)
        is1d = k.ndim == 0 or mu.ndim == 0
        isscalar = k.ndim == mu.ndim == 0
        k, mu = np.atleast_1d(k), np.atleast_1d(mu)
        toret = np.nan * np.zeros((k.size, mu.size), dtype=power.dtype)
        mask_k = (k >= self.edges[0][0]) & (k <= self.edges[0][-1])
        mask_mu = (mu >= self.edges[1][0]) & (mu <= self.edges[1][-1])
        k, mu = k[mask_k], mu[mask_mu]
        if mask_k.any() and mask_mu.any():
            if muavg.size == 1:
                interp = lambda array: UnivariateSpline(kavg, array, k=1, s=0, ext='const')(k)[:, None]
            else:
                interp = lambda array: RectBivariateSpline(kavg, muavg, array, kx=1, ky=1, s=0)(k, mu, grid=True)
            toret[np.ix_(mask_k, mask_mu)] = interp(power.real)
            if complex and np.iscomplexobj(power):
                toret[np.ix_(mask_k, mask_mu)] += 1j * interp(power.imag)
        if is1d:
            toret = toret.ravel()
        if isscalar:
            toret = toret[0]
        if return_k:
            if return_mu:
                return k, mu, toret
            return k, toret
        return toret

    def __copy__(self):
        new = super(BasePowerRatio, self).__copy__()
        for name in self._result_names:
            setattr(new, name, getattr(self, name).__copy__())
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        for name in self._result_names:
            if hasattr(self, name):
                state[name] = getattr(self, name).__getstate__()
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set this class state."""
        self.__dict__.update(state)
        for name in self._result_names:
            if name in state:
                setattr(self, name, PowerSpectrumWedges.from_state(state[name]))

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        self.log_info('Saving {}.'.format(filename))
        utils.mkdir(os.path.dirname(filename))
        np.save(filename, self.__getstate__(), allow_pickle=True)

    def save_txt(self, filename, fmt='%.12e', delimiter=' ', header=None, comments='# ', **kwargs):
        """
        Save power spectrum ratio as txt file.

        Warning
        -------
        Attributes are not all saved, hence there is :meth:`load_txt` method.

        Parameters
        ----------
        filename : str
            File name.

        fmt : str, default='%.12e'
            Format for floating types.

        delimiter : str, default=' '
            String or character separating columns.

        header : str, list, default=None
            String that will be written at the beginning of the file.
            If multiple lines, provide a list of one-line strings.

        comments : str, default=' #'
            String that will be prepended to the header string.

        kwargs : dict
            Arguments for :meth:`get_power`.
        """
        self.log_info('Saving {}.'.format(filename))
        utils.mkdir(os.path.dirname(filename))
        formatter = {'int_kind': lambda x: '%d' % x, 'float_kind': lambda x: fmt % x, 'complex_kind': lambda x: '{}+{}j'.format(fmt % x.real, fmt % x.imag)}
        if header is None: header = []
        elif isinstance(header, str): header = [header]
        else: header = list(header)
        for name in ['los_type', 'los', 'nmesh', 'boxsize', 'boxcenter']:
            value = self.attrs.get(name, getattr(self, name, None))
            if value is None:
                value = 'None'
            elif any(name.startswith(key) for key in ['los_type']):
                value = str(value)
            else:
                value = np.array2string(np.array(value), separator=delimiter, formatter=formatter).replace('\n', '')
            header.append('{} = {}'.format(name, value))
        labels = ['nmodes']
        assert len(self._coords_names) == self.ndim
        for name in self._coords_names:
            labels += ['{}mid'.format(name), '{}avg'.format(name)]
        labels += self._power_names
        power = self.get_ratio(**kwargs)
        columns = [self.nmodes.flat]
        mids = np.meshgrid(*[(edges[:-1] + edges[1:])/2. for edges in self.edges], indexing='ij')
        for idim in range(self.ndim):
            columns += [mids[idim].flat, self.modes[idim].flat]
        for column in power.reshape((-1,)*(power.ndim == self.ndim) + power.shape):
            columns += [column.flat]
        columns = [[np.array2string(value, formatter=formatter) for value in column] for column in columns]
        widths = [max(max(map(len, column)) - len(comments) * (icol == 0), len(label)) for icol, (column, label) in enumerate(zip(columns, labels))]
        widths[-1] = 0 # no need to leave a space
        header.append((' '*len(delimiter)).join(['{:<{width}}'.format(label, width=width) for label, width in zip(labels, widths)]))
        widths[0] += len(comments)
        with open(filename, 'w') as file:
            for line in header:
                file.write(comments + line + '\n')
            for irow in range(len(columns[0])):
                file.write(delimiter.join(['{:<{width}}'.format(column[irow], width=width) for column, width in zip(columns, widths)]) + '\n')

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new

    def __getitem__(self, slices):
        """Call :meth:`slice`."""
        new = self.copy()
        if isinstance(slices, tuple):
            new.slice(*slices)
        else:
            new.slice(slices)
        return new

    def select(self, *xlims):
        """
        Restrict statistic to provided coordinate limits in place.

        For example:

        .. code-block:: python

            statistic.select((0, 0.3)) # restrict first axis to (0, 0.3)
            statistic.select(None, (0, 0.2)) # restrict second axis to (0, 0.2)

        """
        for name in self._result_names:
            getattr(self, name).select(*xlims)

    def slice(self, *slices):
        """
        Slice statistics in place. If slice step is not 1, use :meth:`rebin`.
        For example:

        .. code-block:: python

            statistic.slice(slice(0, 10, 2), slice(0, 6, 3)) # rebin by factor 2 (resp. 3) along axis 0 (resp. 1), up to index 10 (resp. 6)
            statistic[:10:2,:6:3] # same as above, but return new instance.

        """
        for name in self._result_names:
            getattr(self, name).slice(*slices)

    def rebin(self, factor=1):
        """
        Rebin statistic, by factor(s) ``factor``.
        A tuple must be provided in case :attr:`ndim` is greater than 1.
        Input factors must divide :attr:`shape`.
        """
        for name in self._result_names:
            getattr(self, name).rebin(factor=factor)


def _make_property(name):

    @property
    def func(self):
        return getattr(self.num, name)

    return func

for name in ['edges', 'shape', 'ndim', 'nmodes', 'modes', 'k', 'mu', 'kavg', 'muavg', 'attrs']:
    setattr(BasePowerRatio, name, _make_property(name))



class MeshFFTCorrelator(BasePowerRatio):
    r"""
    Estimate correlation between two meshes (reconstructed and initial fields), i.e.:

    .. math::

        r(k) = \frac{P_{\mathrm{rec},\mathrm{init}}}{\sqrt{P_{\mathrm{rec}}P_{\mathrm{init}}}}
    """
    _result_names = ['num', 'auto_reconstructed', 'auto_initial']
    _power_names = ['correlator']

    def __init__(self, mesh_reconstructed, mesh_initial, edges=None, los=None, compensations=None):
        r"""
        Initialize :class:`MeshFFTCorrelation`.

        Parameters
        ----------
        mesh_reconstructed : CatalogMesh, RealField
            Mesh with reconstructed density field.
            If ``RealField``, should be :math:`1 + \delta` or :math:`\bar{n} (1 + \delta)`.

        mesh_initial : CatalogMesh, RealField
            Mesh with initial density field (before structure formation).
            If ``RealField``, should be :math:`1 + \delta` or :math:`\bar{n} (1 + \delta)`.

        edges : tuple, array, default=None
            :math:`k`-edges for :attr:`poles`.
            One can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'step' (if not provided :func:`pypower.fft_power.find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').

        los : array, default=None
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        compensations : list, tuple, default=None
            Compensations to apply to mesh to (optionally) correct for particle-mesh assignment scheme;
            e.g. 'cic-sn' (resp. 'cic') for cic assignment scheme, with (resp. without) interlacing.
            Provide a list (or tuple) of two such strings (for ``mesh_reconstructed`` and `mesh_initial``, respectively).
            Used only if provided ``mesh_reconstructed`` or ``mesh_initial`` are not ``CatalogMesh``.
        """
        if los is None:
            raise ValueError('Provide a box axis or a vector as line-of-sight (no varying line-of-sight accepted)')
        num = MeshFFTPower(mesh_reconstructed, mesh2=mesh_initial, edges=edges, ells=None, los=los, compensations=compensations, shotnoise=0.)
        self.num = num.wedges
        # If compensations is a tuple, the code will anyway use the first element
        self.auto_reconstructed = MeshFFTPower(mesh_reconstructed, edges=edges, ells=None, los=los, compensations=num.compensations).wedges
        self.auto_initial = MeshFFTPower(mesh_initial, edges=edges, ells=None, los=los, compensations=num.compensations[::-1]).wedges

    def get_ratio(self, complex=False, **kwargs):
        """
        Return correlation, computed using various options.

        Parameters
        ----------
        complex : bool, default=False
            Whether (``True``) to return the ratio of complex power spectra,
            or (``False``) return the ratio of their real part only.

        kwargs : dict
            Optionally, arguments for :meth:`BasePowerSpectrumStatistics.get_power`.

        Results
        -------
        ratio : array
        """
        return self.num.get_power(complex=complex, **kwargs)/(self.auto_reconstructed.get_power(complex=complex, **kwargs) * self.auto_initial.get_power(complex=complex, **kwargs))**0.5

    def to_propagator(self, growth=1.):
        """
        Return propagator, using computed power spectra.

        Parameters
        ----------
        growth : float, default=1.
            Growth factor (and galaxy bias) to turn initial field to the linearly-evolved galaxy density field at the redshift of interest.

        Returns
        -------
        propagator : MeshFFTPropagator
        """
        new = MeshFFTPropagator.__new__(MeshFFTPropagator)
        new.num = self.num
        new.denom = self.auto_initial
        new.growth = growth
        return new

    def to_transfer(self, growth=1.):
        """
        Return transfer function, using computed power spectra.

        Parameters
        ----------
        growth : float, default=1.
            Growth factor (and galaxy bias) to turn initial field to the linearly-evolved galaxy density field at the redshift of interest.

        Returns
        -------
        transfer : MeshFFTTransfer
        """
        new = MeshFFTTransfer.__new__(MeshFFTTransfer)
        new.num = self.auto_reconstructed
        new.denom = self.auto_initial
        new.growth = growth
        return new


class MeshFFTTransfer(BasePowerRatio):
    r"""
    Estimate transfer function, i.e.:

    .. math::

        t(k) = \frac{P_{\mathrm{rec}}}{G^{2}(z) b^{2}(z) P_{\mathrm{init}}}
    """
    _power_names = ['transfer']
    _attrs = ['growth']

    def __init__(self, mesh_reconstructed, mesh_initial, edges=None, los=None, compensations=None, growth=1.):
        """
        Initialize :class:`MeshFFTTransfer`.

        Parameters
        ----------
        mesh_reconstructed : CatalogMesh, RealField
            Mesh with reconstructed density field.

        mesh_initial : CatalogMesh, RealField
            Mesh with initial density field (before structure formation).

        edges : tuple, array, default=None
            :math:`k`-edges for :attr:`poles`.
            One can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'step' (if not provided :func:`pypower.fft_power.find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').

        los : array, default=None
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        compensations : list, tuple, default=None
            Compensations to apply to mesh to (optionally) correct for particle-mesh assignment scheme;
            e.g. 'cic-sn' (resp. 'cic') for cic assignment scheme, with (resp. without) interlacing.
            Provide a list (or tuple) of two such strings (for ``mesh_reconstructed`` and `mesh_initial``, respectively).
            Used only if provided ``mesh_reconstructed`` or ``mesh_initial`` are not ``CatalogMesh``.

        growth : float, default=1.
            Growth factor (and galaxy bias) to turn initial field to the linearly-evolved galaxy density field at the redshift of interest.
        """
        num = MeshFFTPower(mesh_reconstructed, edges=edges, ells=None, los=los, compensations=compensations)
        self.num = num.wedges
        self.denom = MeshFFTPower(mesh_initial, edges=edges, ells=None, los=los, compensations=num.compensations[::-1]).wedges
        self.growth = growth

    def get_ratio(self, complex=False, **kwargs):
        """
        Return transfer function, computed using various options.

        Parameters
        ----------
        complex : bool, default=False
            Whether (``True``) to return the ratio of complex power spectra,
            or (``False``) return the ratio of their real part only.

        kwargs : dict
            Optionally, arguments for :meth:`BasePowerSpectrumStatistics.get_power`.

        Results
        -------
        ratio : array
        """
        return super(MeshFFTTransfer, self).get_ratio(complex=complex, **kwargs)**0.5/self.growth


class MeshFFTPropagator(BasePowerRatio):
    r"""
    Estimate propagator, i.e.:

    .. math::

        g(k) = \frac{P_{\mathrm{rec},\mathrm{init}}}{G(z) b(z) P_{\mathrm{init}}}
    """
    _power_names = ['propagator']
    _attrs = ['growth']

    def __init__(self, mesh_reconstructed, mesh_initial, edges=None, los=None, compensations=None, growth=1.):
        """
        Initialize :class:`MeshFFTPropagator`.

        Parameters
        ----------
        mesh_reconstructed : CatalogMesh, RealField
            Mesh with reconstructed density field.

        mesh_initial : CatalogMesh, RealField
            Mesh with initial density field (before structure formation).

        edges : tuple, array, default=None
            :math:`k`-edges for :attr:`poles`.
            One can also provide :math:`\mu-edges` (hence a tuple ``(kedges, muedges)``) for :attr:`wedges`.
            ``kedges`` may be a dictionary, with keys 'min' (minimum :math:`k`, defaults to 0), 'max' (maximum :math:`k`, defaults to ``np.pi/(boxsize/nmesh)``),
            'step' (if not provided :func:`pypower.fft_power.find_unique_edges` is used to find unique :math:`k` (norm) values between 'min' and 'max').

        los : array, default=None
            May be 'x', 'y' or 'z', for one of the Cartesian axes.
            Else, a 3-vector.

        compensations : list, tuple, default=None
            Compensations to apply to mesh to (optionally) correct for particle-mesh assignment scheme;
            e.g. 'cic-sn' (resp. 'cic') for cic assignment scheme, with (resp. without) interlacing.
            Provide a list (or tuple) of two such strings (for ``mesh_reconstructed`` and `mesh_initial``, respectively).
            Used only if provided ``mesh_reconstructed`` or ``mesh_initial`` are not ``CatalogMesh``.

        growth : float, default=1.
            Growth factor (and galaxy bias) to turn initial field to the linearly-evolved galaxy density field at the redshift of interest.
        """
        num = MeshFFTPower(mesh_reconstructed, mesh2=mesh_initial, edges=edges, ells=None, los=los, compensations=compensations, shotnoise=0.)
        self.num = num.wedges
        self.denom = MeshFFTPower(mesh_initial, edges=edges, ells=None, los=los, compensations=num.compensations[::-1]).wedges
        self.growth = growth

    def get_ratio(self, complex=False, **kwargs):
        """
        Return propagator, computed using various options.

        Parameters
        ----------
        complex : bool, default=False
            Whether (``True``) to return the ratio of complex power spectra,
            or (``False``) return the ratio of their real part only.

        kwargs : dict
            Optionally, arguments for :meth:`BasePowerSpectrumStatistics.get_power`.

        Results
        -------
        ratio : array
        """
        return super(MeshFFTPropagator, self).get_ratio(complex=complex, **kwargs)/self.growth
