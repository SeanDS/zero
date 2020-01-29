"""Component library parser"""

import logging
import numpy as np

from .base import BaseConfig
from ..format import Quantity
from ..misc import db_to_mag

LOGGER = logging.getLogger(__name__)


class OpAmpLibrary(BaseConfig):
    """Op-amp library"""
    # User config filename.
    USER_CONFIG_FILENAME = "components.yaml"
    # Default config copied to user directory if requested.
    DEFAULT_USER_CONFIG_FILENAME = USER_CONFIG_FILENAME + ".dist"
    # Config into which others are merged.
    BASE_CONFIG_FILENAME = USER_CONFIG_FILENAME + ".dist.default"

    def __init__(self, *args, **kwargs):
        """Instantiate a new op-amp library."""
        super().__init__(*args, **kwargs)
        # Defaults.
        self.data = {}
        self.loaded = False
        # Load and parse op-amp data from config file.
        self.populate_library()

    def populate_library(self):
        """Load and parse op-amp data from config file."""
        count = 0
        # Each section is a new op-amp.
        if "op-amps" in self and self["op-amps"] is not None:
            for opamp, data in self["op-amps"].items():
                self._parse_lib_data(opamp, data)
                count += 1
        LOGGER.debug("found %i op-amps", count)
        self.loaded = True

    @classmethod
    def format_name(cls, name):
        """Format op-amp name for use as a key in the data dict.

        Parameters
        ----------
        name : :class:`str`
            The name to format.

        Returns
        -------
        :class:`str`
            The formatted name.
        """
        return str(name).upper()

    def get_opamp(self, model):
        """Get op-amp by model.

        Parameters
        ----------
        model : :class:`str`
            The op-amp model.

        Returns
        -------
        :class:`.LibraryOpAmp`
            The op-amp.
        """
        for opamp in self.opamps:
            if opamp.model.upper() == model.upper():
                return opamp
        raise ValueError(f"op-amp model '{model}' not found in library.")

    def get_data(self, name):
        """Get op-amp data.

        Parameters
        ----------
        name : :class:`str`
            The op-amp name.

        Returns
        -------
        :class:`dict`
            The op-amp data.

        Raises
        ------
        ValueError
            If the specified op-amp name is not found in the library.
        """
        model = self.format_name(name)
        try:
            return self.data[model]
        except KeyError:
            raise ValueError(f"op-amp model '{name}' not found in library")

    def has_data(self, name):
        """Check if op-amp data exists in library.

        Parameters
        ----------
        name : :class:`str`
            The op-amp name.

        Returns
        -------
        :class:`bool`
            Whether the op-amp exists in the library.
        """
        return self.format_name(name) in self.data.keys()

    def match(self, opamp):
        """Get model name of library op-amp given a specified op-amp.

        Parameters
        ----------
        opamp : :class:`.OpAmp`
            The op-amp object to match.

        Returns
        -------
        :class:`str`
            The op-amp's name as specified in the library, or None if not found.
        """
        for model in self.data:
            if opamp.params == self.data[model]:
                return model
        return None

    def _parse_lib_data(self, name, data):
        """Parse op-amp data from config file."""
        poles = []
        zeros = []
        if "poles" in data and data["poles"] is not None:
            for freq in data["poles"]:
                poles.extend(self._parse_freq_str(freq))
        if "zeros" in data and data["zeros"] is not None:
            for freq in data["zeros"]:
                zeros.extend(self._parse_freq_str(freq))
        data["poles"] = np.array(poles)
        data["zeros"] = np.array(zeros)
        # Check if there are aliases.
        aliases = []
        if "aliases" in data:
            aliases.extend([alias.strip() for alias in data["aliases"].split(",")])
        # Remove unused op-amp fields.
        for field in ["aliases", "comment", "description"]:
            if field in data:
                del data[field]
        # Add data to library.
        self.add_data(name, data)
        # Create new op-amps for each alias using identical data.
        for alias in aliases:
            self.add_data(alias, data)

    def add_data(self, name, data):
        """Add op-amp data to library.

        Parameters
        ----------
        name : :class:`str`
            The op-amp name.
        data : :class:`dict`
            The op-amp data.

        Raises
        ------
        ValueError
            If the op-amp is already in library.
        """
        name = self.format_name(name)
        if name in self.opamp_names:
            raise ValueError(f"Duplicate op-amp type: '{name}'")
        self.data[name] = data

    @property
    def opamp_names(self):
        """Get names of op-amps in library (including alises)."""
        return self.data.keys()

    def _parse_freq_str(self, token):
        """Parse token as complex frequency/frequencies.

        The frequency may include an optional q-factor, which results in this
        method returning a pair of equal and opposite complex frequencies. The
        one or two returned frequencies are always contained in a list.

        Parameters
        ----------
        token : :class:`str`
            The frequency and optional q-factor.

        Returns
        -------
        :class:`list`
            The list of frequencies.

        Raises
        ------
        Exception
            If the frequency list is malformed.
        """
        frequencies = []
        # Split frequency and optional q-factor into list entries.
        try:
            parts = token.split()
        except AttributeError:
            # Assume number.
            parts = [token]
        # Frequency is always first in the list.
        frequency = Quantity(parts[0], "Hz")
        # Q-factor is second, if present.
        if len(parts) == 1:
            frequencies.append(frequency)
        elif len(parts) == 2:
            # Calculate complex frequency using q-factor.
            qfactor = Quantity(parts[1])
            # Cast to complex to avoid issues with arccos.
            qfactor = complex(qfactor)
            theta = np.arccos(1 / (2 * qfactor))
            # Add negative/positive pair of poles/zeros.
            frequencies.append(frequency * np.exp(-1j * theta))
            frequencies.append(frequency * np.exp(1j * theta))
        else:
            raise Exception("invalid frequency list")
        return frequencies

    @property
    def opamps(self):
        opdata = self.data.items()
        return [LibraryOpAmp(model=self.format_name(model), **data) for model, data in opdata]


class LibraryOpAmp:
    """Represents a library op-amp.

    Some of the default parameter values are based on the OP27.

    Parameters
    ----------
    model : :class:`str`
        Model name.
    a0 : :class:`float`, optional
        Open loop gain.
    gbw : :class:`float`, optional
        Gain-bandwidth product.
    delay : :class:`float`, optional
        Delay.
    zeros : sequence, optional
        Zeros.
    poles : sequence, optional
        Poles.
    vnoise : :class:`float`, optional
        Flat voltage noise.
    vcorner : :class:`float`, optional
        Voltage noise corner frequency.
    inoise : :class:`float`, optional
        Flat current noise.
    icorner : :class:`float`, optional
        Current noise corner frequency.
    vmax : :class:`float`, optional
        Maximum input voltage.
    imax : :class:`float`, optional
        Maximum output current.
    sr : :class:`float`, optional
        Slew rate.
    """
    def __init__(self, model="OP00", a0=1.5e6, gbw=8e6, delay=0, zeros=np.array([]),
                 poles=np.array([]), vnoise=3.2e-9, vcorner=2.7, inoise=0.4e-12, icorner=140,
                 vmax=12, imax=0.06, sr=1e6, **kwargs):
        super().__init__(**kwargs)
        # Default properties.
        self._model = "None"
        self.params = {}

        # Op-amp parameters.
        self.model = model
        self.a0 = a0
        self.gbw = gbw
        self.delay = delay
        self.zeros = zeros
        self.poles = poles
        self.vnoise = vnoise
        self.vcorner = vcorner
        self.inoise = inoise
        self.icorner = icorner
        self.vmax = vmax
        self.imax = imax
        self.sr = sr

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = str(model).upper()

    @property
    def a0(self):
        """Gain"""
        return self.params["a0"]

    @a0.setter
    def a0(self, a0):
        try:
            a0 = a0.strip()
            if a0[-2:].lower() == "db":
                # Convert decibels to absolute magnitude.
                a0 = db_to_mag(float(a0[:-2].strip()))
        except AttributeError:
            pass
        self.params["a0"] = Quantity(a0, "V/V")

    @property
    def gbw(self):
        """Gain-bandwidth product"""
        return self.params["gbw"]

    @gbw.setter
    def gbw(self, gbw):
        self.params["gbw"] = Quantity(gbw, "Hz")

    @property
    def delay(self):
        """Delay"""
        return self.params["delay"]

    @delay.setter
    def delay(self, delay):
        self.params["delay"] = Quantity(delay, "s")

    @property
    def zeros(self):
        """Additional zeros"""
        return self.params["zeros"]

    @zeros.setter
    def zeros(self, zeros):
        self.params["zeros"] = np.array(zeros)

    @property
    def zeros_mag_q(self):
        """Additional zeros, in tuples containing magnitude and Q-factor"""
        return self._mag_q_pairs(self.zeros)

    @property
    def poles(self):
        """Additional poles"""
        return self.params["poles"]

    @poles.setter
    def poles(self, poles):
        self.params["poles"] = np.array(poles)

    @property
    def poles_mag_q(self):
        """Additional poles, in tuples containing magnitude and Q-factor"""
        return self._mag_q_pairs(self.poles)

    @property
    def vnoise(self):
        """Voltage noise"""
        return self.params["vnoise"]

    @vnoise.setter
    def vnoise(self, vnoise):
        self.params["vnoise"] = Quantity(vnoise, "V/sqrt(Hz)")

    @property
    def vcorner(self):
        """Voltage noise corner frequency"""
        return self.params["vcorner"]

    @vcorner.setter
    def vcorner(self, vcorner):
        self.params["vcorner"] = Quantity(vcorner, "Hz")

    @property
    def inoise(self):
        """Current noise"""
        return self.params["inoise"]

    @inoise.setter
    def inoise(self, inoise):
        self.params["inoise"] = Quantity(inoise, "A/sqrt(Hz)")

    @property
    def icorner(self):
        """Current noise corner frequency"""
        return self.params["icorner"]

    @icorner.setter
    def icorner(self, icorner):
        self.params["icorner"] = Quantity(icorner, "Hz")

    @property
    def vmax(self):
        """Maximum output voltage"""
        return self.params["vmax"]

    @vmax.setter
    def vmax(self, vmax):
        self.params["vmax"] = Quantity(vmax, "V")

    @property
    def imax(self):
        """Maximum output current"""
        return self.params["imax"]

    @imax.setter
    def imax(self, imax):
        self.params["imax"] = Quantity(imax, "A")

    @property
    def sr(self):
        """Slew rate"""
        return self.params["sr"]

    @sr.setter
    def sr(self, sr):
        self.params["sr"] = Quantity(sr, "V/s")

    def gain(self, frequency):
        """Get op-amp voltage gain at the specified frequency.

        Parameters
        ----------
        frequency : :class:`float`
            Frequency to compute gain at.

        Returns
        -------
        :class:`float`
            Op-amp gain at specified frequency.
        """
        return (self.a0
                / (1 + self.a0 * 1j * frequency / self.gbw)
                * np.exp(-2j * np.pi * self.delay * frequency)
                * np.prod(1 + 1j * frequency / self.zeros)
                / np.prod(1 + 1j * frequency / self.poles))

    def inverse_gain(self, *args, **kwargs):
        """Op-amp inverse gain.

        Note that the inverse gain may be modified by the analysis, e.g. in the
        case of a voltage follower (see :meth:`zero.analysis.ac.BaseAcAnalysis.component_equation`).
        """
        return 1 / self.gain(*args, **kwargs)

    def _mag_q_pairs(self, complex_freqs):
        complex_freqs = list(complex_freqs)
        pairs = []
        for freq in complex_freqs:
            fabs = np.absolute(freq)
            freq_conj = np.conj(freq)
            # Find conjugate.
            if freq_conj in complex_freqs:
                complex_freqs.remove(freq_conj)
            qfactor = fabs / (2 * np.real(freq)) # = 0.5 if real pole
            pairs.append((fabs, qfactor))
        return pairs

    def __str__(self):
        return f"{self.model}(a0={self.a0}, gbw={self.gbw}, delay={self.delay})"

    def __repr__(self):
        def format_poles(poles):
            formatted_poles = []
            for mag, q in poles:
                frequency = Quantity(mag, units="Hz")
                if q == 0.5:
                    q = "real"
                else:
                    q = f"q={q}"
                formatted_poles.append(f"{frequency} ({q})")
            return ", ".join(formatted_poles)

        if self.poles.size:
            poles = format_poles(self.poles_mag_q)
        else:
            poles = "--"
        if self.zeros.size:
            zeros = format_poles(self.zeros_mag_q)
        else:
            zeros = "--"
        return f"""{self.model}
    a0: {self.a0}
    gbw: {self.gbw}
    delay: {self.delay}
    vnoise: {self.vnoise}
    vcorner: {self.vcorner}
    inoise: {self.inoise}
    icorner: {self.icorner}
    vmax: {self.vmax}
    imax: {self.imax}
    sr: {self.sr}
    poles: {poles}
    zeros: {zeros}
"""
