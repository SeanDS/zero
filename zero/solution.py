"""Plotting functions for solutions to simulations"""

import logging
from collections import defaultdict
import datetime
import numpy as np

from .config import ZeroConfig
from .data import (Response, NoiseDensity, MultiNoiseDensity, ReferenceResponse, ReferenceNoise,
                   frequencies_match)
from .components import BaseElement
from .noise import Noise
from .format import Quantity
from .display import BodePlotter, SpectralDensityPlotter

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()


class Solution:
    """Represents a solution to the simulated circuit"""
    # Function filter flags.
    RESPONSE_SOURCES_ALL = "all"
    RESPONSE_SINKS_ALL = "all"
    RESPONSE_GROUPS_ALL = "all"
    RESPONSE_LABELS_ALL = "all"
    NOISE_SOURCES_ALL = "all"
    NOISE_SINKS_ALL = "all"
    NOISE_GROUPS_ALL = "all"
    NOISE_LABELS_ALL = "all"
    NOISE_TYPES_ALL = "all"

    # Default group names (reserved).
    DEFAULT_GROUP_NAME = "__default__"
    DEFAULT_REF_GROUP_NAME = "reference"

    def __init__(self, frequencies, name=None):
        """Instantiate a new solution

        :param frequencies: sequence of frequencies this solution contains results for
        :type frequencies: :class:`~np.ndarray`
        """
        # Functions by group. The order of functions in their groups, and the groups themselves,
        # determine plotting order.
        self.functions = defaultdict(list)
        # Map of functions to their groups, for quick look-ups.
        self._function_groups = {}

        # Default functions in each group.
        self.default_responses = defaultdict(list)
        self.default_noise = defaultdict(list)
        self.default_noise_sums = defaultdict(list)

        # Reference functions.
        self.response_references = []
        self.noise_references = []

        # Default plotters.
        self.response_plotter = BodePlotter
        self.noise_plotter = SpectralDensityPlotter
        # Last created plotter.
        self._last_plotter = None

        self._name = None

        # Creation date.
        self._creation_date = datetime.datetime.now()
        # Solution name.
        self.name = name

        self.frequencies = frequencies

    @property
    def groups(self):
        return list(self.functions) + [self.DEFAULT_REF_GROUP_NAME]

    def get_group_functions(self, group=None):
        """Get functions by group"""
        if group is None:
            group = self.DEFAULT_GROUP_NAME
        elif group not in self.functions:
            raise ValueError(f"group '{group}' does not exist")

        return self.functions[group]

    def function_group(self, function):
        """Get function group"""
        return self._function_groups[function]

    def sort_functions(self, key_function, default_only=False):
        """Sort functions using specified callback.

        Parameters
        ----------
        key_function : callable
            Function that yields a key given a :class:`.BaseFunction`.
        default_only : bool, optional
            Whether to sort only the default functions.
        """
        groups = defaultdict(list)

        if default_only:
            functions = self.default_functions
        else:
            functions = self.functions

        for group, functions in functions.items():
            groups[group] = sorted(functions, key=key_function)

        self.functions = groups

    def _merge_groupsets(self, *groupsets):
        """Merge grouped functions into one dict"""
        combined = defaultdict(list)

        for groups in groupsets:
            for group, functions in groups.items():
                combined[group].extend(functions)

        return combined

    @property
    def name(self):
        name = self._name

        if self._name is None:
            # use creation date
            name = str(self._creation_date)

        return name

    @name.setter
    def name(self, name):
        self._name = name

    def add_response(self, response, default=False, group=None):
        """Add a response to the solution.

        Parameters
        ----------
        response : :class:`.Response`
            The response to add.
        default : `bool`, optional
            Whether this response is a default.
        group : `str`, optional
            Group name.

        Raises
        ------
        ValueError
            If the specified response is incompatible with this solution.
        """
        # Dimension sanity checks.
        if not np.all(response.frequencies == self.frequencies):
            raise ValueError(f"response '{response}' doesn't fit this solution")

        self._add_function(response, group=group)

        if default:
            self.set_response_as_default(response, group)

    def is_default_response(self, response, group=None):
        if group is None:
            group = self.DEFAULT_GROUP_NAME

        return response in self.default_responses[group]

    def set_response_as_default(self, response, group=None):
        """Set the specified response as a default.

        Parameters
        ----------
        response : :class:`.Response`
            The response to set as default.
        group : str, optional
            The function group. If None, the default is used.

        Raises
        ------
        ValueError
            If `response` is not part of this solution or already set as a default.
        """
        if group is None:
            group = self.DEFAULT_GROUP_NAME

        if response not in self.responses[group]:
            raise ValueError(f"response '{response}' is not in the solution")

        if self.is_default_response(response, group):
            raise ValueError(f"response '{response}' is already default")

        self.default_responses[group].append(response)

    def add_noise(self, spectral_density, default=False, group=None):
        """Add a noise spectral density to the solution.

        Parameters
        ----------
        spectral_density : :class:`.NoiseDensity`
            The noise spectral density to add.
        default : :class:`bool`, optional
            Whether this noise spectral density is a default.
        group : :class:`str`, optional
            The function group. If None, the default is used.

        Raises
        ------
        ValueError
            If the specified noise spectral density is incompatible with this solution or not
            single-source or single-sink.
        """
        if not isinstance(spectral_density, NoiseDensity):
            raise ValueError(f"noise density '{spectral_density}' is not single-source and -sink "
                             "type")

        # dimension sanity checks
        if not np.all(spectral_density.frequencies == self.frequencies):
            raise ValueError(f"noise density '{spectral_density}' doesn't fit this solution")

        self._add_function(spectral_density, group=group)

        if default:
            self.set_noise_as_default(spectral_density, group)

    def is_default_noise(self, noise, group=None):
        if group is None:
            group = self.DEFAULT_GROUP_NAME

        return noise in self.default_noise[group]

    def set_noise_as_default(self, spectral_density, group=None):
        """Set the specified noise spectral density as a default"""
        if group is None:
            group = self.DEFAULT_GROUP_NAME

        if spectral_density not in self.noise[group]:
            raise ValueError(f"noise density '{spectral_density}' is not in the solution")

        if self.is_default_noise(spectral_density, group):
            raise ValueError(f"noise density '{spectral_density}' is already default")

        self.default_noise[group].append(spectral_density)

    def add_noise_sum(self, noise_sum, default=False, group=None):
        """Add a noise sum to the solution.

        Parameters
        ----------
        noise_sum : :class:`.MultiNoiseDensity`
            The noise sum to add.
        default : :class:`bool`, optional
            Whether this noise sum is a default.
        group : :class:`str`, optional
            The function group. If None, the default is used.

        Raises
        ------
        ValueError
            If the specified noise sum is incompatible with this solution or not multi-source.
        """
        if not isinstance(noise_sum, MultiNoiseDensity):
            raise ValueError(f"noise sum '{noise_sum}' is not multi-source type")

        # dimension sanity checks
        if not np.all(noise_sum.frequencies == self.frequencies):
            raise ValueError(f"noise sum '{noise_sum}' doesn't fit this solution")

        self._add_function(noise_sum, group=group)

        if default:
            self.set_noise_sum_as_default(noise_sum, group)

    def is_default_noise_sum(self, noise_sum, group=None):
        if group is None:
            group = self.DEFAULT_GROUP_NAME

        return noise_sum in self.default_noise_sums[group]

    def set_noise_sum_as_default(self, noise_sum, group=None):
        """Set the specified noise sum as a default"""
        if group is None:
            group = self.DEFAULT_GROUP_NAME

        if noise_sum not in self.noise_sums[group]:
            raise ValueError(f"noise sum '{noise_sum}' is not in the solution")

        if self.is_default_noise_sum(noise_sum):
            raise ValueError(f"noise sum '{noise_sum}' is already default")

        self.default_noise_sums[group].append(noise_sum)

    def add_response_reference(self, *args, reference=None, **kwargs):
        if reference is None:
            reference = ReferenceResponse(*args, **kwargs)
        if reference in self.response_references:
            raise ValueError("response reference is already present in the solution")
        self.response_references.append(reference)

    def add_noise_reference(self, *args, reference=None, **kwargs):
        if reference is None:
            reference = ReferenceNoise(*args, **kwargs)
        if reference in self.noise_references:
            raise ValueError("noise reference is already present in the solution")
        self.noise_references.append(reference)

    def _add_function(self, function, group=None):
        if group is None:
            group = self.DEFAULT_GROUP_NAME
        elif group in (self.DEFAULT_GROUP_NAME, self.DEFAULT_REF_GROUP_NAME):
            raise ValueError(f"group '{group}' is a reserved keyword")

        group = str(group)

        if function in self.functions[group]:
            raise ValueError(f"duplicate function '{function}' in group '{group}'")

        self.functions[group].append(function)
        self._function_groups[function] = group

    @classmethod
    def __group_params(cls, sparam, mparam, sparam_name, mparam_name, default=None, allmstr=None):
        """Create list with either the singular or multiple valued parameter's value(s).

        This method allows only one or the other parameter to be defined, and returns a list
        containing whatever it finds. If neither parameter is defined, the specified default is
        instead returned.
        """
        if sparam is None and mparam is None:
            return default
        if sparam is not None:
            if mparam is not None:
                raise ValueError(f"{sparam_name} and {mparam_name} cannot both be defined")
            return [sparam]
        if allmstr is not None and mparam == allmstr:
            # The "all" specifier was given for the multi-specifier; let this though unimpeded.
            return mparam
        return list(mparam)

    def filter_responses(self, group=None, groups=None, source=None, sources=None, sink=None,
                         sinks=None, label=None, labels=None):
        groups = self.__group_params(group, groups, "group", "groups",
                                     default=[self.DEFAULT_GROUP_NAME],
                                     allmstr=self.RESPONSE_GROUPS_ALL)
        sources = self.__group_params(source, sources, "source", "sources",
                                      allmstr=self.RESPONSE_SOURCES_ALL)
        sinks = self.__group_params(sink, sinks, "sink", "sinks", allmstr=self.RESPONSE_SINKS_ALL)
        labels = self.__group_params(label, labels, "label", "labels",
                                     allmstr=self.RESPONSE_LABELS_ALL)
        return self._apply_response_filters(self.responses, groups=groups, sources=sources,
                                            sinks=sinks, labels=labels)

    def _apply_response_filters(self, responses, groups=None, sources=None, sinks=None,
                                labels=None):
        filter_sources = []
        filter_sinks = []

        if groups is None:
            groups = self.RESPONSE_GROUPS_ALL
        if sources is None:
            sources = self.RESPONSE_SOURCES_ALL
        if sinks is None:
            sinks = self.RESPONSE_SINKS_ALL
        if labels is None:
            labels = self.RESPONSE_LABELS_ALL

        if groups != self.RESPONSE_GROUPS_ALL:
            # Filter by group.
            for group in list(responses):
                if group not in groups:
                    del responses[group]

        if sources != self.RESPONSE_SOURCES_ALL:
            if isinstance(sources, str):
                sources = [sources]

            for source in sources:
                if isinstance(source, str):
                    source = self.get_response_source(source)
                if not isinstance(source, BaseElement):
                    raise ValueError(f"signal source '{source}' invalid")
                filter_sources.append(source)

            # Filter by source.
            for group, group_responses in responses.items():
                responses[group] = [response for response in group_responses
                                    if response.source in filter_sources]

        if sinks != self.RESPONSE_SINKS_ALL:
            if isinstance(sinks, str):
                sinks = [sinks]

            for sink in sinks:
                if isinstance(sink, str):
                    sink = self.get_response_sink(sink)
                if not isinstance(sink, BaseElement):
                    raise ValueError(f"signal sink '{sink}' invalid")
                filter_sinks.append(sink)

            # Filter by sink.
            for group, group_responses in responses.items():
                responses[group] = [response for response in group_responses
                                    if response.sink in filter_sinks]

        if labels != self.RESPONSE_LABELS_ALL:
            # Filter by label.
            if isinstance(labels, str):
                labels = [labels]
            for group, group_responses in responses.items():
                for response in list(group_responses): # List required to allow item removal.
                    if response.label not in labels:
                        # No match.
                        group_responses.remove(response)
                responses[group] = group_responses

        return responses

    def get_response(self, source=None, sink=None, group=None, label=None):
        """Get response from specified source to specified sink.

        This is a convenience method for :meth:`.filter_responses` for when only a single response
        is required.

        Parameters
        ----------
        source : :class:`str` or :class:`.Node` or :class:`.Component`, optional
            The response source element.
        sink : :class:`str` or :class:`.Node` or :class:`.Component`, optional
            The response sink element.
        group : :class:`str`, optional
            The response group. If `None`, the default group is assumed.
        label : :class:`str`, optional
            The response label.

        Returns
        -------
        :class:`.Response`
            The matched response.

        Raises
        ------
        ValueError
            If no response is found, or if more than one matching response is found.

        Examples
        --------
        Get response from node `nin` to node `nout` using string specifiers:

        >>> get_response("nin", "nout")

        Get response from node `nin` to component `op1` using objects:

        >>> get_response(Node("nin"), op1)

        Get response from node `nin` to node `nout`, searching only in group `b`:

        >>> get_response("nin", "nout", group="b")
        """
        response_groups = self.filter_responses(source=source, sink=sink, group=group, label=label)
        if not response_groups:
            raise ValueError("no response found")
        responses = list(response_groups.values())[0]
        if not responses:
            raise ValueError("no response found")
        if len(response_groups) > 1 or len(responses) > 1:
            raise ValueError("degenerate responses for the specified source, sink, and group")
        return responses[0]

    def filter_noise(self, group=None, groups=None, source=None, sources=None, sink=None,
                     sinks=None, label=None, labels=None, type=None, types=None):
        """Filter for noise spectra.

        This does not include sums.
        """
        groups = self.__group_params(group, groups, "group", "groups",
                                     default=[self.DEFAULT_GROUP_NAME],
                                     allmstr=self.NOISE_GROUPS_ALL)
        sources = self.__group_params(source, sources, "source", "sources",
                                      allmstr=self.NOISE_SOURCES_ALL)
        sinks = self.__group_params(sink, sinks, "sink", "sinks", allmstr=self.NOISE_SINKS_ALL)
        labels = self.__group_params(label, labels, "label", "labels",
                                     allmstr=self.NOISE_LABELS_ALL)
        types = self.__group_params(type, types, "type", "types", allmstr=self.NOISE_TYPES_ALL)
        return self._apply_noise_filters(self.noise, groups=groups, sources=sources, sinks=sinks,
                                         labels=labels, types=types)

    def _filter_default_noise(self, **kwargs):
        """Special filter for default noise spectra.

        This does not include default sums.
        """
        return self._apply_noise_filters(self.default_noise, **kwargs)

    def filter_noise_sums(self, group=None, groups=None, sink=None, sinks=None, label=None,
                          labels=None, type=None, types=None):
        """Filter for noise sums."""
        groups = self.__group_params(group, groups, "group", "groups",
                                     default=[self.DEFAULT_GROUP_NAME])
        sinks = self.__group_params(sink, sinks, "sink", "sinks")
        labels = self.__group_params(label, labels, "label", "labels")
        types = self.__group_params(type, types, "type", "types")
        return self._apply_noise_filters(self.noise_sums, groups=groups, sinks=sinks, labels=labels,
                                         types=types)

    def _scale_functions(self, scale_function, functions):
        for group, group_functions in functions.items():
            for function in group_functions:
                self.replace(function, function * scale_function, group=group)

    def scale_responses(self, scale, **kwargs):
        """Apply a scaling to responses matching the specified filters.

        Supports the keyword arguments of :meth:`._apply_response_filters`.

        Parameters
        ----------
        scale : number or :class:`.BaseFunction`
            The scaling to apply to the matched responses.
        """
        self._scale_functions(scale, self.filter_responses(**kwargs))

    def scale_noise(self, scale, include_singular=True, include_sums=True, **kwargs):
        """Apply a scaling to noise matching the specified filters.

        Supports the keyword arguments of :meth:`.filter_noise`.

        Parameters
        ----------
        scale : number or :class:`.BaseFunction`
            The scaling to apply to the matched noise.
        include_singular : :class:`bool`, optional
            Scale single noise functions.
        include_sums : :class:`bool`, optional
            Scale noise sums.
        """
        if include_singular:
            self._scale_functions(scale, self.filter_noise(**kwargs))
        if include_sums:
            # Remove source filters, since sums don't have single sources to match against.
            if "source" in kwargs:
                del kwargs["source"]
            if "sources" in kwargs:
                del kwargs["sources"]
            self._scale_functions(scale, self.filter_noise_sums(**kwargs))

    def replace(self, current_function, new_function, group=None):
        """Replace existing function with the specified function."""
        if group is None:
            group = self.DEFAULT_GROUP_NAME
        index = self.functions[group].index(current_function)
        self.functions[group][index] = new_function

    def rename_group(self, source_group, new_group):
        """Rename the specified group, moving all of its functions to the new group.

        The new group must not already exist. If it does, :meth:`.merge_group` should be used
        instead.

        Raises
        ------
        ValueError
            If the new group already exists.
        """
        if new_group in self.groups:
            raise ValueError("use merge_group to move functions into an existing group")
        # Touch the new function group to create its key.
        self.functions[new_group]
        # Merge source functions into new group.
        self.merge_group(source_group, new_group)

    def move_default_group_functions(self, new_group):
        """Move the default group's functions to a new group.

        The default group will still be used for any new functions added to the solution with no
        explicit group, as usual.
        """
        return self.merge_group(self.DEFAULT_GROUP_NAME, new_group)

    def merge_group(self, source_group, target_group):
        """Merge functions from source group into target group.

        The source group cannot contain any functions in the target group.

        Raises
        ------
        ValueError
            If the source or target group is the reference group.
        ValueError
            If a function in the source group matches one already present in the target group.
        """
        if source_group not in self.groups:
            raise ValueError("source group does not exist")
        if self.DEFAULT_REF_GROUP_NAME in (source_group, target_group):
            raise ValueError("neither source nor target group can be the reference group")
        if target_group == self.DEFAULT_GROUP_NAME:
            target_group_name = None
        for function in self.functions[source_group]:
            self._add_function(function, group=target_group_name)
        # Remove source group.
        del self.functions[source_group]
        # Move default settings.
        self.default_responses[target_group] = self.default_responses[source_group]
        self.default_noise[target_group] = self.default_noise[source_group]
        self.default_noise_sums[target_group] = self.default_noise_sums[source_group]
        del self.default_responses[source_group]
        del self.default_noise[source_group]
        del self.default_noise_sums[source_group]

    def _apply_noise_filters(self, spectra, groups=None, sources=None, sinks=None, labels=None,
                             types=None):
        filter_sources = []
        filter_sinks = []

        if groups is None:
            groups = self.NOISE_GROUPS_ALL
        if sources is None:
            sources = self.NOISE_SOURCES_ALL
        if sinks is None:
            sinks = self.NOISE_SINKS_ALL
        if labels is None:
            labels = self.NOISE_LABELS_ALL
        if types is None:
            types = self.NOISE_TYPES_ALL

        if groups != self.NOISE_GROUPS_ALL:
            # Filter by group.
            for group in list(spectra):
                if group not in groups:
                    del spectra[group]

        if sources != self.NOISE_SOURCES_ALL:
            if isinstance(sources, str):
                sources = [sources]

            for source in sources:
                if isinstance(source, str):
                    source = self.get_noise_source(source)
                if not isinstance(source, Noise):
                    raise ValueError(f"noise source '{source}' is not a noise source")
                filter_sources.append(source)

            # Filter by source.
            for group, group_spectra in spectra.items():
                spectra[group] = [spectral_density for spectral_density in group_spectra
                                  if spectral_density.source in filter_sources]

        if sinks != self.NOISE_SINKS_ALL:
            if isinstance(sinks, str):
                sinks = [sinks]

            for sink in sinks:
                if isinstance(sink, str):
                    sink = self.get_noise_sink(sink)
                if not isinstance(sink, BaseElement):
                    raise ValueError(f"noise sink '{sink}' invalid")
                filter_sinks.append(sink)

            # Filter by sink.
            for group, group_spectra in spectra.items():
                spectra[group] = [spectral_density for spectral_density in group_spectra
                                  if spectral_density.sink in filter_sinks]

        if labels != self.NOISE_LABELS_ALL:
            # Filter by label.
            if isinstance(labels, str):
                labels = [labels]
            for group, group_spectra in spectra.items():
                for noise in list(group_spectra): # List required to allow item removal.
                    if noise.label not in labels:
                        # No match.
                        group_spectra.remove(noise)
                spectra[group] = group_spectra

        if types != self.NOISE_TYPES_ALL:
            # Filter by noise type.
            for group, group_spectra in spectra.items():
                for noise in list(group_spectra): # List required to allow item removal.
                    if noise.element_type not in types and noise.noise_type not in types:
                        # No match.
                        group_spectra.remove(noise)
                spectra[group] = group_spectra

        return spectra

    def get_noise(self, source=None, sink=None, group=None, label=None):
        """Get noise spectral density from specified source to specified sink.

        This is a convenience method for :meth:`.filter_noise` for when only a single noise spectral
        density is required.

        Parameters
        ----------
        source : :class:`str` or :class:`~.components.Noise`, optional
            The noise source element.
        sink : :class:`str` or :class:`.Node` or :class:`.Component`, optional
            The noise sink element.
        group : :class:`str`, optional
            The noise group. If `None`, the default group is assumed.
        label : :class:`str`, optional
            The noise label.

        Returns
        -------
        :class:`.NoiseDensity`
            The matched noise spectral density.

        Raises
        ------
        ValueError
            If no noise spectral density is found, or if more than one matching noise spectral
            density is found.

        Examples
        --------
        Get noise arising from op-amp `op1`'s voltage noise at node `nout` using string specifiers:

        >>> get_noise("V(op1)", "nout")

        Get noise arising from op-amp `op1`'s voltage noise at component `op2` using objects:

        >>> get_noise(op1.voltage_noise, op2)

        Get noise arising from op-amp `op1`'s voltage noise at node `nout`, searching only in group
        `b`:

        >>> get_noise("V(op1)", "nout", group="b")
        """
        noise_groups = self.filter_noise(source=source, sink=sink, group=group, label=label)
        if not noise_groups:
            raise ValueError("no noise found")
        noise_densities = list(noise_groups.values())[0]
        if not noise_densities:
            raise ValueError("no noise found")
        if len(noise_groups) > 1 or len(noise_densities) > 1:
            raise ValueError("degenerate noise spectral densities for the specified source, sink, "
                             "group and label")
        return noise_densities[0]

    def get_noise_sum(self, sink=None, group=None, label=None):
        """Get noise sum with the specified label.

        Parameters
        ----------
        sink : :class:`str` or :class:`.Node` or :class:`.Component`, optional
            The noise sink element.
        group : :class:`str`, optional
            The noise group. If `None`, the default group is assumed.
        label : :class:`str`, optional
            The noise label.

        Returns
        -------
        :class:`.MultiNoiseDensity`
            The matched noise sum.

        Raises
        ------
        ValueError
            If no noise sum is found.
        """
        noise_groups = self.filter_noise_sums(sink=sink, group=group, label=label)
        if not noise_groups:
            raise ValueError("no noise sums found")
        noise_densities = list(noise_groups.values())[0]
        if not noise_densities:
            raise ValueError("no noise sums found")
        if len(noise_groups) > 1 or len(noise_densities) > 1:
            raise ValueError("degenerate noise sums for the specified sink, group and label")
        return noise_densities[0]

    @property
    def responses(self):
        return {group: [function for function in functions if isinstance(function, Response)]
                for group, functions in self.functions.items()}

    @property
    def noise(self):
        return {group: [function for function in functions if isinstance(function, NoiseDensity)]
                for group, functions in self.functions.items()}

    @property
    def component_noise(self):
        return {group: [function for function in functions if function.element_type == "component"]
                for group, functions in self.noise.items()}

    @property
    def opamp_noise(self):
        return {group: [function for function in functions
                        if function.source.component_type == "op-amp"]
                for group, functions in self.component_noise.items()}

    @property
    def resistor_noise(self):
        return {group: [function for function in functions
                        if function.source.component_type == "resistor"]
                for group, functions in self.component_noise.items()}

    @property
    def noise_sums(self):
        return {group: [function for function in functions if isinstance(function, MultiNoiseDensity)]
                for group, functions in self.functions.items()}

    @property
    def has_responses(self):
        for responses in self.responses.values():
            if responses:
                return True
        return False

    @property
    def has_noise(self):
        for spectra in self.noise.values():
            if spectra:
                return True
        return False

    @property
    def has_noise_sums(self):
        for sum_spectra in self.noise_sums.values():
            if sum_spectra:
                return True
        return False

    @property
    def default_functions(self):
        """Default responses and noise spectra"""
        return self._merge_groupsets(self.default_responses, self.default_noise,
                                     self.default_noise_sums)

    @property
    def response_sources(self):
        sources = set()
        for responses in self.responses.values():
            sources.update([response.source for response in responses])
        return sources

    def get_response_source(self, source_name):
        source_name = source_name.lower()

        for source in self.response_sources:
            if source_name == source.name.lower():
                return source

        raise ValueError(f"signal source '{source_name}' not found")

    @property
    def response_sinks(self):
        sinks = set()
        for responses in self.responses.values():
            sinks.update([response.sink for response in responses])
        return sinks

    def get_response_sink(self, sink_name):
        sink_name = sink_name.lower()

        for sink in self.response_sinks:
            if sink_name == sink.name.lower():
                return sink

        raise ValueError(f"signal sink '{sink_name}' not found")

    @property
    def noise_sources(self):
        """Get noise sources.

        :return: noise sources
        """
        sources = set()
        for spectra in self.noise.values():
            sources.update([spectral_density.source for spectral_density in spectra])
        for sum_spectra in self.noise_sums.values():
            for sum_spectral_density in sum_spectra:
                sources.update(sum_spectral_density.sources)
        return sources

    def get_noise_source(self, source_name):
        source_name = source_name.lower()

        for source in self.noise_sources:
            if source_name == source.label.lower():
                return source

        raise ValueError(f"noise source '{source_name}' not found")

    @property
    def noise_sinks(self):
        sinks = set()
        for spectra in list(self.noise.values()) + list(self.noise_sums.values()):
            sinks.update([spectral_density.sink for spectral_density in spectra])
        return sinks

    def get_noise_sink(self, sink_name):
        sink_name = sink_name.lower()

        for sink in self.noise_sinks:
            if sink_name == sink.name.lower():
                return sink

        raise ValueError(f"noise sink '{sink_name}' not found")

    @property
    def n_frequencies(self):
        return len(list(self.frequencies))

    def plot(self):
        if self.has_responses:
            self._last_plotter = self.plot_responses()
        if self.has_noise:
            self._last_plotter = self.plot_noise()

    def plot_responses(self, figure=None, group=None, groups=None, source=None, sources=None,
                       sink=None, sinks=None, xlabel=None, ylabel_mag=None, ylabel_phase=None,
                       scale_db=True, **kwargs):
        """Plot responses.

        Note: if only one of "sources" or "sinks" is specified, the other defaults to "all" as per
        the behaviour of :meth:`.filter_responses`.

        Parameters
        ----------
        figure : :class:`~matplotlib.figure.Figure`, optional
            Figure to plot to. If not specified, a new figure is created.
        group, groups : :class:`str` or list of :class:`str`, optional
            The response group(s) to plot. If None, the default group is assumed.
        source, sources, sink, sinks : :class:`str` or list of :class:`str`, :class:`.Component`
                                        or :class:`.Node`
            The source(s) and sink(s) to plot responses between. If None, all matched sources and
            sinks are plotted.
        xlabel, ylabel_mag, ylabel_phase : :class:`str`, optional
            The x- and y-axis labels for the magnitude and phase plots.
        scale_db : :class:`bool`, optional
            Scale the magnitude y-axis values in decibels. If False, absolute scaling is used.

        Other Parameters
        ----------------
        legend : :class:`bool`, optional
            Display legend.
        legend_loc : :class:`str`, optional
            Legend display location. Defaults to "best".
        legend_groups : :class:`bool`, optional
            Display function group names in legends, if the group is not the default.
        title : :class:`str`, optional
            The plot title.
        xlim, mag_ylim, phase_ylim : sequence of :class:`float`, optional
            The lower and upper limits for the x- and y-axes for the magnitude and phase plots.
        db_tick_major_step, db_tick_minor_step : :class:`float`, optional
            The magnitude y axis tick step sizes when ``scale_db`` is enabled. Defaults to 20 and 10
            for the major and minor steps, respectively.
        phase_tick_major_step, phase_tick_minor_step : :class:`float`, optional
            The phase y axis tick step sizes when ``scale_db`` is enabled. Defaults to 30 and 15 for
            the major and minor steps, respectively.

        Returns
        -------
        :class:`.BasePlotter`
            The plotter object.
        """
        # Plot default noise if no filters are being applied.
        default_none_param_names = ("group", "groups", "source", "sources", "sink", "sinks")
        default_none_params = (group, groups, source, sources, sink, sinks)
        if all([param is None for param in default_none_params]):
            # No filters have been applied.
            groups = self.default_responses
        else:
            groups = self.filter_responses(source=source, sources=sources, sink=sink, sinks=sinks,
                                           group=group, groups=groups)
        if not groups:
            filters = ", ".join([f"'{param}'" for param in default_none_param_names])
            raise NoDataException(f"No responses found. Consider setting one of {filters}.")
        # Add reference functions.
        groups[self.DEFAULT_REF_GROUP_NAME] = self.response_references
        # Draw plot.
        plotter = self.response_plotter(figure=figure, xlabel=xlabel, ylabel_mag=ylabel_mag,
                                        ylabel_phase=ylabel_phase, scale_db=scale_db,
                                        hidden_group_names=[self.DEFAULT_GROUP_NAME], **kwargs)
        plotter.plot_groups(groups)
        self._last_plotter = plotter
        return plotter

    def plot_noise(self, figure=None, group=None, groups=None, source=None, sources=None, sink=None,
                   sinks=None, type=None, types=None, show_individual=True, show_sums=True,
                   xlabel=None, ylabel=None, **kwargs):
        """Plot noise.

        Note: if only some of "groups", "sources", "sinks", "types" are specified, the others
        default to "all" as per the behaviour of :meth:`.filter_noise`.

        Parameters
        ----------
        figure : :class:`~matplotlib.figure.Figure`, optional
            Figure to plot to. If not specified, a new figure is created.
        group, groups : :class:`str` or list of :class:`str`, optional
            The noise group(s) to plot. If None, the default group is assumed.
        source, sources : :class:`str` or list of :class:`str` or :class:`.Noise`, optional
            The noise source(s) to plot at the specified ``sinks``. If None, all matched sources are
            plotted.
        sink, sinks : :class:`str` or list of :class:`str`, :class:`.Component` or :class:`.Node`, optional
            The sink(s) to plot noise at. If None, all matched sinks are plotted.
        type, types : :class:`str` or list of :class:`str`, optional
            The noise type(s) to plot. If None, all matched noise types are plotted.
        show_individual : :class:`bool`, optional
            Plot any individual noise spectra contained in this solution.
        show_sums : :class:`bool`, optional
            Plot any sums contained in this solution.
        xlabel, ylabel : :class:`str`, optional
            The x- and y-axis labels.

        Other Parameters
        ----------------
        legend : :class:`bool`, optional
            Display legend.
        legend_loc : :class:`str`, optional
            Legend display location. Defaults to "best".
        legend_groups : :class:`bool`, optional
            Display function group names in legends, if the group is not the default.
        title : :class:`str`, optional
            The plot title.
        xlim, ylim : sequence of :class:`float`, optional
            The lower and upper limits for the x- and y-axes.

        Returns
        -------
        :class:`.BasePlotter`
            The plotter object.
        """
        # Plot default noise if no filters are being applied.
        default_none_param_names = ("group", "groups", "source", "sources", "sink", "sinks", "type",
                                    "types")
        default_none_params = (group, groups, source, sources, sink, sinks, type, types)

        grouped_noise = {}
        if all([param is None for param in default_none_params]):
            # No filters have been applied.
            if show_individual:
                grouped_noise = self._merge_groupsets(grouped_noise, self._filter_default_noise())
            if show_sums:
                grouped_noise = self._merge_groupsets(grouped_noise, self.default_noise_sums)
        else:
            if show_individual:
                individual = self.filter_noise(source=source, sources=sources, sink=sink,
                                               sinks=sinks, group=group, groups=groups, type=type,
                                               types=types)
                grouped_noise = self._merge_groupsets(grouped_noise, individual)
            if show_sums:
                sums = self.filter_noise_sums(sink=sink, sinks=sinks, group=group, groups=groups,
                                              type=type, types=types)
                grouped_noise = self._merge_groupsets(grouped_noise, sums)
        if not grouped_noise:
            filters = ", ".join([f"'{param}'" for param in default_none_param_names])
            raise NoDataException(f"No noise spectra found. Consider setting one of {filters}.")
        if ylabel is None:
            # Show all plotted noise units.
            unit_tex = []
            noise_units = set()
            for spectra in grouped_noise.values():
                noise_units.update([spectral_density.sink_unit for spectral_density in spectra])
            if noise_units:
                unit_tex = [r"$\frac{\mathrm{" + unit + r"}}{\sqrt{\mathrm{Hz}}}$"
                            for unit in sorted(noise_units)]
            unit = ", ".join(unit_tex)
            ylabel = r"$\bf{Noise}$" + f" ({unit})"
        # Add reference functions.
        grouped_noise[self.DEFAULT_REF_GROUP_NAME] = self.noise_references
        # Draw plot.
        plotter = self.noise_plotter(figure=figure, xlabel=xlabel, ylabel=ylabel,
                                     hidden_group_names=[self.DEFAULT_GROUP_NAME], **kwargs)
        plotter.plot_groups(grouped_noise)
        self._last_plotter = plotter
        return plotter

    def show(self):
        """Show plot(s)."""
        if self._last_plotter is not None:
            self._last_plotter.show()

    def __str__(self):
        return self.name

    def __repr__(self):
        data = f"Solution '{self}'"

        if self.has_responses:
            data += "\n\tResponses:"
            for group, responses in self.responses.items():
                if group == self.DEFAULT_GROUP_NAME:
                    data += f"\n\t\tDefault group:"
                else:
                    data += f"\n\t\tGroup '{group}':"
                for response in responses:
                    data += f"\n\t\t\t{response}"
                    if self.is_default_response(response):
                        data += " (default)"

        if self.has_noise:
            data += "\n\tNoise spectra:"
            for group, noises in self.noise.items():
                if group == self.DEFAULT_GROUP_NAME:
                    data += f"\n\t\tDefault group:"
                else:
                    data += f"\n\t\tGroup '{group}':"
                for noise in noises:
                    data += f"\n\t\t\t{noise}"
                    if self.is_default_noise(noise):
                        data += " (default)"

        if self.has_noise_sums:
            data += "\n\tNoise sums:"
            for group, noise_sums in self.noise_sums.items():
                if group == self.DEFAULT_GROUP_NAME:
                    data += f"\n\t\tDefault group:"
                else:
                    data += f"\n\t\tGroup '{group}':"
                for noise_sum in noise_sums:
                    data += f"\n\t\t\t{noise_sum}"
                    if self.is_default_noise_sum(noise_sum):
                        data += " (default)"

        return data

    def __add__(self, other):
        return self.combine(other)

    def combine(self, *others, name=None, merge_groups=False):
        """Combine this solution with the specified other solution(s).

        The groups of each solution are copied to a new, combined solution. By default, the group
        names have the source solution's name appended as a suffix in the form "group name (solution
        name)", unless the group is the default group, in which case the functions are placed in a
        group with the solution name only. When the `merge_groups` flag is True, in cases where
        groups from each solution have the same name, their functions are combined into a single new
        group as long as none of the functions are present in both source groups.

        To be able to be combined, the two solutions must have equivalent frequency vectors, and
        cannot have the same name.

        Parameters
        ----------
        *others : sequence of :class:`.Solution`
            The solution(s) to combine.
        name : :class:`str`, optional
            The name to give to the combined solution. Defaults to the "A + B + ..." where "A", "B",
            etc. are the source solutions.

        Returns
        -------
        :class:`.Solution`
            The combined solution.

        Raises
        ------
        ValueError
            If the solutions have the same name.
        ValueError
            If the solutions have different frequency vectors.
        ValueError
            If an identical function with an identical group is present in both solutions.
        """
        other_names = ", ".join([str(other) for other in others])
        LOGGER.info(f"combining {self} solution with {other_names}")
        for other in others:
            if str(self) == str(other):
                raise ValueError("cannot combined groups with the same name")
            # Check frequencies match.
            if not frequencies_match(self.frequencies, other.frequencies):
                raise ValueError(f"specified other solution '{other}' has incompatible frequencies")

        if name is None:
            name = " + ".join([str(solution) for solution in [self] + list(others)])

        # Resultant solution.
        result = self.__class__(self.frequencies, name)

        def new_group_name(group, solution):
            if merge_groups:
                new_group = group if group != self.DEFAULT_GROUP_NAME else None
            else:
                if group == solution.DEFAULT_GROUP_NAME:
                    # Use the solution name as the group name.
                    new_group = solution.name
                else:
                    # Append the solution name to the group name.
                    new_group = f"{group} ({solution.name})"
            return new_group

        def merge_into_result(solution):
            for group, responses in solution.responses.items():
                new_group = new_group_name(group, solution)
                for response in responses:
                    is_default = solution.is_default_response(response, group)
                    result.add_response(response, default=is_default, group=new_group)
            for group, spectra in solution.noise.items():
                new_group = new_group_name(group, solution)
                for spectral_density in spectra:
                    is_default = solution.is_default_noise(spectral_density, group)
                    result.add_noise(spectral_density, default=is_default, group=new_group)
            for group, noise_sums in solution.noise_sums.items():
                new_group = new_group_name(group, solution)
                for noise_sum in noise_sums:
                    is_default = solution.is_default_noise_sum(noise_sum, group)
                    result.add_noise_sum(noise_sum, default=is_default, group=new_group)
            for response_ref in solution.response_references:
                result.add_response_reference(reference=response_ref)
            for noise_ref in solution.noise_references:
                result.add_noise_reference(reference=noise_ref)

        merge_into_result(self)
        for other in others:
            merge_into_result(other)

        return result

    def equivalent_to(self, other, **kwargs):
        """Checks if the specified other solution has equivalent, identical functions to this one.

        Parameters
        ----------
        other : :class:`.Solution`
            The other solution to compare to.

        Other Parameters
        ----------------
        defaults_only : :class:`bool`, optional
            Whether to check only the default functions, or everything. Defaults to everything.
        meta_only : :class:`bool`, optional
            Whether to check only meta data, not function data, when comparing. Defaults to False.

        Returns
        -------
        :class:`bool`
            True if equivalent, False otherwise.
        """
        # check frequencies match
        if np.all(self.frequencies != other.frequencies):
            return False

        # get unmatched functions
        _, residuals_a, residuals_b = matches_between(self, other, **kwargs)

        if residuals_a or residuals_b:
            if residuals_a:
                LOGGER.info("function(s) in %s but not %s: %s", self, other,
                            ", ".join([str(n) for n in residuals_a]))

            if residuals_b:
                LOGGER.info("function(s) in %s but not %s: %s", other, self,
                            ", ".join([str(n) for n in residuals_b]))

            return False

        return True

    def difference(self, other, **kwargs):
        """Get table containing the difference between this solution and the specified one."""
        # find matching functions
        matches, residuals_a, residuals_b = matches_between(self, other, **kwargs)

        header = ["", "Worst difference (absolute)", "Worst difference (relative)"]
        rows = []

        for func_a, func_b in matches:
            # relevant data
            frequencies = func_a.frequencies
            data_a = func_a.series.y
            data_b = func_b.series.y

            # absolute and relative worst indices
            iworst = np.argmax(np.abs(data_a - data_b))
            worst = np.abs(data_a[iworst] - data_b[iworst])
            fworst = Quantity(frequencies[iworst], units="Hz")
            irelworst = np.argmax(np.abs((data_a - data_b) / data_b))
            relworst = np.abs((data_a[irelworst] - data_b[irelworst]) / data_b[irelworst])
            frelworst = Quantity(frequencies[irelworst], units="Hz")

            if worst != 0:
                # descriptions of worst
                fmt = "%.2e (f = %s)"
                strworst = fmt % (worst, fworst.format())
                strrelworst = fmt % (relworst, frelworst.format())
            else:
                strworst = "n/a"
                strrelworst = "n/a"

            rows.append([str(func_a), strworst, strrelworst])

        for residual in residuals_a + residuals_b:
            data = residual.series.y[0]
            rows.append([str(residual), "-", "-"])

        return header, rows


class NoDataException(Exception):
    pass


def matches_between(sol_a, sol_b, defaults_only=False, meta_only=False):
    """Finds matching functions in the specified solutions. Ignores groups.

    Parameters
    ----------
    sol_a, sol_b : :class:`.Solution`
        The solutions to compare.
    defaults_only : :class:`bool`, optional
        Whether to check only the default functions, or everything. Defaults to everything.
    meta_only : :class:`bool`, optional
        Whether to check only meta data, not function data, when comparing. Defaults to False.

    Returns
    -------
    matches : :class:`list`
        Matching pairs from each solution.
    residuals_a, residuals_b : :class:`list`
        Functions only present in the first or second solutions, respectively.
    """
    if defaults_only:
        sol_a_functions = sol_a.default_functions
        sol_b_functions = sol_b.default_functions
    else:
        sol_a_functions = sol_a.functions
        sol_b_functions = sol_b.functions

    def function_in_dict(check_item, search_dict):
        """Check for match depending on type of equivalence specified."""
        for items in search_dict.values():
            for item in items:
                if meta_only:
                    match = check_item.meta_equivalent(item)
                else:
                    match = check_item.equivalent(item)

                if match:
                    return item

        return None

    # Lists to hold matching pairs and those remaining.
    matches = []
    residuals_a = []
    residuals_b = []

    # Functions in list b but not a (will be refined in next step).
    residuals_b = [function for functions in sol_b_functions.values() for function in functions]

    # Get difference between functions.
    # Can't use sets here because Series is not hashable (because data can match not exactly
    # but within tolerance).
    for items_a in sol_a_functions.values():
        for item_a in items_a:
            item_b = function_in_dict(item_a, sol_b_functions)

            if item_b is None:
                # Not matched in solution b.
                residuals_a.append(item_a)
            else:
                matches.append((item_a, item_b))

                # Remove from residuals.
                residuals_b.remove(item_b)

    return matches, residuals_a, residuals_b
