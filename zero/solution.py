"""Plotting functions for solutions to simulations"""

import logging
from collections import defaultdict
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import cycler

from .config import ZeroConfig
from .data import Response, NoiseDensity, MultiNoiseDensity, frequencies_match
from .components import Component, Node, Noise
from .format import Quantity
from .misc import lighten_colours

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()


class Solution:
    """Represents a solution to the simulated circuit"""
    # Function filter flags.
    RESPONSE_SOURCES_ALL = "all"
    RESPONSE_SINKS_ALL = "all"
    RESPONSE_GROUPS_ALL = "all"
    NOISE_SOURCES_ALL = "all"
    NOISE_SINKS_ALL = "all"
    NOISE_GROUPS_ALL = "all"
    NOISE_TYPES_ALL = "all"

    # Default group name (reserved).
    DEFAULT_GROUP_NAME = "__default__"

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
        self._default_responses = defaultdict(list)
        self._default_noise = defaultdict(list)
        self._default_noise_sums = defaultdict(list)

        self._name = None

        # creation date
        self._creation_date = datetime.datetime.now()
        # solution name
        self.name = name

        # line style group cycle
        self._linestyles = ["-", "--", "-.", ":"]

        # default colour cycle
        self._default_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # cycles by group
        self._group_colours = {}

        self.frequencies = frequencies

    @property
    def groups(self):
        return list(self.functions)

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
            Function that yields a key given a :class:`.Function`.
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

    def _merge_groups(self, *groupsets):
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

        return response in self._default_responses[group]

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

        self._default_responses[group].append(response)

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

        return noise in self._default_noise[group]

    def set_noise_as_default(self, spectral_density, group=None):
        """Set the specified noise spectral density as a default"""
        if group is None:
            group = self.DEFAULT_GROUP_NAME

        if spectral_density not in self.noise[group]:
            raise ValueError(f"noise density '{spectral_density}' is not in the solution")

        if self.is_default_noise(spectral_density, group):
            raise ValueError(f"noise density '{spectral_density}' is already default")

        self._default_noise[group].append(spectral_density)

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

        return noise_sum in self._default_noise_sums[group]

    def set_noise_sum_as_default(self, noise_sum, group=None):
        """Set the specified noise sum as a default"""
        if group is None:
            group = self.DEFAULT_GROUP_NAME

        if noise_sum not in self.noise_sums[group]:
            raise ValueError(f"noise sum '{noise_sum}' is not in the solution")

        if self.is_default_noise_sum(noise_sum):
            raise ValueError(f"noise sum '{noise_sum}' is already default")

        self._default_noise_sums[group].append(noise_sum)

    def _add_function(self, function, group=None):
        if group is None:
            group = self.DEFAULT_GROUP_NAME
        elif group == self.DEFAULT_GROUP_NAME:
            raise ValueError(f"group '{self.DEFAULT_GROUP_NAME}' is a reserved keyword")

        group = str(group)

        if function in self.functions[group]:
            raise ValueError(f"duplicate function '{function}' in group '{group}'")

        self.functions[group].append(function)
        self._function_groups[function] = group

    def filter_responses(self, **kwargs):
        return self._apply_response_filters(self.responses, **kwargs)

    def _apply_response_filters(self, responses, groups=None, sources=None, sinks=None):
        filter_sources = []
        filter_sinks = []

        if groups is None:
            groups = self.RESPONSE_GROUPS_ALL
        if sources is None:
            sources = self.RESPONSE_SOURCES_ALL
        if sinks is None:
            sinks = self.RESPONSE_SINKS_ALL

        if groups != self.RESPONSE_GROUPS_ALL:
            # Filter by group.
            for group in responses:
                if group not in groups:
                    del responses[groups]

        if sources != self.RESPONSE_SOURCES_ALL:
            if isinstance(sources, str):
                sources = [sources]

            for source in sources:
                if isinstance(source, str):
                    source = self.get_response_source(source)

                if not isinstance(source, (Component, Node)):
                    raise ValueError(f"signal source '{source}' is not a component or node")

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

                if not isinstance(sink, (Component, Node)):
                    raise ValueError(f"signal sink '{sink}' is not a component or node")

                filter_sinks.append(sink)

            # Filter by sink.
            for group, group_responses in responses.items():
                responses[group] = [response for response in group_responses
                                    if response.sink in filter_sinks]

        return responses

    def filter_noise(self, **kwargs):
        """Filter for noise spectra.

        This does not include sums.
        """
        return self._apply_noise_filters(self.noise, **kwargs)

    def _filter_default_noise(self, **kwargs):
        """Special filter for default noise spectra.

        This does not include default sums.
        """
        return self._apply_noise_filters(self._default_noise, **kwargs)

    def _apply_noise_filters(self, spectra, groups=None, sources=None, sinks=None, types=None):
        filter_sources = []
        filter_sinks = []

        if groups is None:
            groups = self.NOISE_GROUPS_ALL
        if sources is None:
            sources = self.NOISE_SOURCES_ALL
        if sinks is None:
            sinks = self.NOISE_SINKS_ALL
        if types is None:
            types = self.NOISE_TYPES_ALL

        if groups != self.NOISE_GROUPS_ALL:
            # Filter by group.
            for group in spectra:
                if group not in groups:
                    del spectra[groups]

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

                if not isinstance(sink, Node):
                    raise ValueError(f"noise sink '{sink}' is not a node")

                filter_sinks.append(sink)

            # Filter by sink.
            for group, group_spectra in spectra.items():
                spectra[group] = [spectral_density for spectral_density in group_spectra
                                  if spectral_density.sink in filter_sinks]

        if types != self.NOISE_TYPES_ALL:
            # Filter by noise type.
            for group, group_spectra in spectra.items():
                for noise in group_spectra:
                    if noise.noise_type not in types and noise.noise_subtype not in types:
                        # No match.
                        group_spectra.remove(noise)
                spectra[group] = group_spectra

        return spectra

    @property
    def responses(self):
        return {group: [function for function in functions if isinstance(function, Response)]
                for group, functions in self.functions.items()}

    @property
    def noise(self):
        return {group: [function for function in functions if isinstance(function, NoiseDensity)]
                for group, functions in self.functions.items()}

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
        return self._merge_groups(self._default_responses, self._default_noise,
                                  self._default_noise_sums)

    @property
    def response_source_nodes(self):
        """Get response input nodes.

        :return: input nodes
        :rtype: Set[:class:`Node`]
        """
        nodes = set()
        for responses in self.responses.values():
            nodes.update([response.source for response in responses
                          if isinstance(response.source, Node)])
        return nodes

    @property
    def response_source_components(self):
        """Get response input components.

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        components = set()
        for responses in self.responses.values():
            components.update([response.source for response in responses
                               if isinstance(response.source, Component)])
        return components

    @property
    def response_sources(self):
        return self.response_source_nodes | self.response_source_components

    def get_response_source(self, source_name):
        source_name = source_name.lower()

        for source in self.response_sources:
            if source_name == source.name.lower():
                return source

        raise ValueError(f"signal source '{source_name}' not found")

    @property
    def response_sink_nodes(self):
        """Get output nodes in solution

        :return: output nodes
        :rtype: Set[:class:`Node`]
        """
        nodes = set()
        for responses in self.responses.values():
            nodes.update([response.sink for response in responses
                          if isinstance(response.sink, Node)])
        return nodes

    @property
    def response_sink_components(self):
        """Get output components in solution

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        components = set()
        for responses in self.responses.values():
            components.update([response.sink for response in responses
                               if isinstance(response.sink, Component)])
        return components

    @property
    def response_sinks(self):
        return self.response_sink_nodes | self.response_sink_components

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
        return sources

    def get_noise_source(self, source_name):
        source_name = source_name.lower()

        for source in self.noise_sources:
            if source_name == source.label().lower():
                return source

        raise ValueError(f"noise source '{source_name}' not found")

    @property
    def noise_sink_nodes(self):
        """Get noise nodes in solution

        :return: noise nodes
        :rtype: Set[:class:`Node`]
        """
        nodes = set()
        for spectra in self.noise.values():
            nodes.update([spectral_density.sink for spectral_density in spectra
                          if isinstance(spectral_density.sink, Node)])
        return nodes

    @property
    def noise_sink_components(self):
        """Get output components in solution

        :return: output components
        :rtype: Sequence[:class:`Component`]
        """
        components = set()
        for spectra in self.noise.values():
            components.update([spectral_density.sink for spectral_density in spectra
                               if isinstance(spectral_density.sink, Component)])
        return components

    @property
    def noise_sinks(self):
        return self.noise_sink_nodes | self.noise_sink_components

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
            self.plot_responses()
        if self.has_noise:
            self.plot_noise()

    def plot_responses(self, figure=None, sources=None, sinks=None, **kwargs):
        """Plot responses.

        Note: if only one of "sources" or "sinks" is specified, the other defaults to "all" as per
        the behaviour of :meth:`.filter_responses`.
        """
        if sources is None and sinks is None:
            responses = self._default_responses
        else:
            responses = self.filter_responses(sources=sources, sinks=sinks)

        if not responses:
            raise NoDataException("no responses found")

        # Draw plot.
        figure = self._plot_bode(responses, figure=figure, **kwargs)
        LOGGER.info("response(s) plotted on %s", figure.canvas.get_window_title())

        return figure

    def plot_noise(self, figure=None, groups=None, sources=None, sinks=None, types=None,
                   show_sums=True, title=None):
        """Plot noise.

        Note: if only some of "groups", "sources", "sinks", "types" are specified, the others
        default to "all" as per the behaviour of :meth:`.filter_noise`.
        """
        if groups is None and sources is None and sinks is None and types is None:
            # Filter against sum flag.
            noise = self._filter_default_noise()

            if show_sums:
                noise = self._merge_groups(noise, self._default_noise_sums)
        else:
            noise = self.filter_noise(sources=sources, sinks=sinks, groups=groups, types=types)

            if show_sums:
                noise = self._merge_groups(noise, self.noise_sums)

        if not noise:
            raise NoDataException("no noise spectra found from specified sources and/or sum(s)")

        figure = self._plot_spectral_density(noise, figure=figure, title=title)
        LOGGER.info("noise plotted on %s", figure.canvas.get_window_title())

        return figure

    def figure(self):
        return plt.figure(figsize=(float(CONF["plot"]["size_x"]),
                                   float(CONF["plot"]["size_y"])))

    def bode_figure(self):
        figure = self.figure()
        ax1 = figure.add_subplot(211)
        _ = figure.add_subplot(212, sharex=ax1)

        return figure

    def noise_figure(self):
        figure = self.figure()
        _ = figure.add_subplot(111)

        return figure

    def _plot_bode(self, responses, figure=None, legend=True, legend_loc="best", legend_groups=True,
                   title=None, xlim=None, mag_ylim=None, phase_ylim=None,
                   xlabel=r"$\bf{Frequency}$ (Hz)", ylabel_mag=r"$\bf{Magnitude}$ (dB)",
                   ylabel_phase=r"$\bf{Phase}$ ($\degree$)", mag_tick_major_step=20,
                   mag_tick_minor_step=10, phase_tick_major_step=30, phase_tick_minor_step=15):
        if figure is None:
            # create figure
            figure = self.bode_figure()

        if len(figure.axes) != 2:
            raise ValueError("specified figure must contain two axes")

        ax1, ax2 = figure.axes

        for group, group_responses in responses.items():
            if not group_responses:
                # Skip empty group.
                continue

            with self._figure_style_context(group):
                # reset axes colour wheels
                ax1.set_prop_cycle(plt.rcParams["axes.prop_cycle"])
                ax2.set_prop_cycle(plt.rcParams["axes.prop_cycle"])

                if legend_groups and group != self.DEFAULT_GROUP_NAME:
                    # Show group.
                    legend_group = "(%s)" % group
                else:
                    legend_group = None

                for response in group_responses:
                    response.draw(ax1, ax2, label_suffix=legend_group)

                # overall figure title
                if title:
                    figure.suptitle(title)

                # legend
                if legend:
                    ax1.legend(loc=legend_loc)

                # limits
                if xlim:
                    ax1.set_xlim(xlim)
                    ax2.set_xlim(xlim)
                if mag_ylim:
                    ax1.set_ylim(mag_ylim)
                if phase_ylim:
                    ax2.set_ylim(phase_ylim)

                # set other axis properties
                ax2.set_xlabel(xlabel)
                ax1.set_ylabel(ylabel_mag)
                ax2.set_ylabel(ylabel_phase)
                ax1.grid(True)
                ax2.grid(True)

                # magnitude and phase tick locators
                ax1.yaxis.set_major_locator(MultipleLocator(base=mag_tick_major_step))
                ax1.yaxis.set_minor_locator(MultipleLocator(base=mag_tick_minor_step))
                ax2.yaxis.set_major_locator(MultipleLocator(base=phase_tick_major_step))
                ax2.yaxis.set_minor_locator(MultipleLocator(base=phase_tick_minor_step))

        return figure

    def _plot_spectral_density(self, noise, figure=None, legend=True, legend_loc="best",
                               legend_groups=True, title=None, xlim=None, ylim=None,
                               xlabel=r"$\bf{Frequency}$ (Hz)", ylabel=None):
        if figure is None:
            # create figure
            figure = self.noise_figure()

        if len(figure.axes) != 1:
            raise ValueError("specified figure must contain one axis")

        ax = figure.axes[0]

        if ylabel is None:
            unit_tex = []
            has_volts = False
            has_amps = False

            # Check which noise units to use.
            for spectra in noise.values():
                if any([spectral_density.sink_unit == "V" for spectral_density in spectra]):
                    has_volts = True
                if any([spectral_density.sink_unit == "A" for spectral_density in spectra]):
                    has_amps = True

            if has_volts:
                unit_tex.append(r"$\frac{\mathrm{V}}{\sqrt{\mathrm{Hz}}}$")
            if has_amps:
                unit_tex.append(r"$\frac{\mathrm{A}}{\sqrt{\mathrm{Hz}}}$")

            unit = ", ".join(unit_tex)
            ylabel = r"$\bf{Noise}$" + f" ({unit})"

        for group, spectra in noise.items():
            if not spectra:
                # Skip empty group.
                continue

            # reset axis colour wheel
            ax.set_prop_cycle(None)

            with self._figure_style_context(group):
                # reset axes colour wheels
                ax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])

                if legend_groups and group != self.DEFAULT_GROUP_NAME:
                    # Show group.
                    legend_group = "(%s)" % group
                else:
                    legend_group = None

                for spectral_density in spectra:
                    spectral_density.draw(ax, label_suffix=legend_group)

                # overall figure title
                if title:
                    figure.suptitle(title)

                # legend
                if legend:
                    ax.legend(loc=legend_loc)

                # limits
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                # set other axis properties
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True)

        return figure

    def _figure_style_context(self, group):
        """Figure style context manager.

        Used to override the default style for a figure.
        """
        # Find group index.
        group_index = self.groups.index(group)
        # get line style according to group index
        index = group_index % len(self._linestyles)

        if group not in self._group_colours:
            # brighten new cycle
            cycle = lighten_colours(self._default_color_cycle, 0.5 ** group_index)
            self._group_colours[group] = cycle

        prop_cycler = cycler(color=self._group_colours[group])

        settings = {"lines.linestyle": self._linestyles[index],
                    "axes.prop_cycle": prop_cycler}

        return plt.rc_context(settings)

    @staticmethod
    def save_figure(figure, path, **kwargs):
        """Save specified figure to specified path.

        (path can be file object or string path)
        """
        # set figure as current figure
        plt.figure(figure.number)

        # squeeze things together
        figure.tight_layout()

        plt.savefig(path, **kwargs)

    @staticmethod
    def show():
        """Show plots"""
        plt.tight_layout()
        plt.show()

    def __str__(self):
        return self.name

    def __repr__(self):
        data = f"Solution '{self}'"

        if self.has_responses:
            data += "\n\tResponses:"
            for response in self.responses:
                data += f"\n\t\t{response}"

                if self.is_default_response(response):
                    data += " (default)"

        if self.has_noise:
            data += "\n\tNoise spectra:"
            for noise in self.noise:
                data += f"\n\t\t{noise}"

                if self.is_default_noise(noise):
                    data += " (default)"

        if self.has_noise_sums:
            data += "\n\tNoise sums:"
            for noise_sum in self.noise_sums:
                data += f"\n\t\t{noise_sum}"

                if self.is_default_noise_sum(noise_sum):
                    data += " (default)"

        return data

    def __add__(self, other):
        return self.combine(other)

    def combine(self, other):
        """Combine this solution with the specified other solution.

        To be able to be combined, the two solutions must have equivalent frequency vectors.
        """
        LOGGER.info("combining %s solution with %s", self, other)

        if str(self) == str(other):
            raise ValueError("cannot combined groups with the same name")

        # check frequencies match
        if not frequencies_match(self.frequencies, other.frequencies):
            raise ValueError(f"specified other solution '{other}' is incompatible with this one")

        for group, functions in self.functions.items():
            if group in other.functions:
                for function in functions:
                    if function in other.functions[group]:
                        LOGGER.debug("function '%s' appears in both solutions (group '%s')",
                                     function, group)

        # Resultant name.
        name = f"{self} + {other}"
        # Resultant solution.
        result = self.__class__(self.frequencies, name)

        def flag_functions(solution):
            # Use solution name as group name.
            new_group = str(solution)

            for group, responses in solution.responses.items():
                for response in responses:
                    is_default = solution.is_default_response(response, group)
                    result.add_response(response, default=is_default, group=new_group)
            for group, spectra in solution.noise.items():
                for spectral_density in spectra:
                    is_default = solution.is_default_noise(spectral_density, group)
                    result.add_noise(spectral_density, default=is_default, group=new_group)
            for group, noise_sums in solution.noise_sums.items():
                for noise_sum in noise_sums:
                    is_default = solution.is_default_noise_sum(noise_sum, group)
                    result.add_noise_sum(noise_sum, default=is_default, group=new_group)

        flag_functions(self)
        flag_functions(other)

        return result

    def equivalent_to(self, other, **kwargs):
        """Checks if the specified other solution has equivalent, identical functions to this one.

        Parameters
        ----------
        other : :class:`.Solution`
            The other solution to compare to.

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
            fworst = Quantity(frequencies[iworst], unit="Hz")
            irelworst = np.argmax(np.abs((data_a - data_b) / data_b))
            relworst = np.abs((data_a[irelworst] - data_b[irelworst]) / data_b[irelworst])
            frelworst = Quantity(frequencies[irelworst], unit="Hz")

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
        Whether to check only the default functions, or everything.
    meta_only : :class:`bool`, optional
        Whether to check only meta data when comparing.

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
