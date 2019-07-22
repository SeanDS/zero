"""LISO output file parser"""

import logging
import numpy as np

from ..solution import Solution
from ..data import Series, Response, NoiseDensity, MultiNoiseDensity
from ..format import Quantity
from ..components import OpAmp, Input, Node
from .base import (LisoParser, LisoParserError, LisoOutputVoltage, LisoOutputCurrent,
                   LisoNoisyElement)

LOGGER = logging.getLogger(__name__)


class LisoOutputParser(LisoParser):
    """LISO output file parser

    This implements a lexer to identify appropriate definitions in a LISO output file,
    and a parser to build a solution and circuit from what is found.

    The parsing of the LISO output file is more complicated than that for the input file.
    The data is first parsed. For this, the lexer is initially in its default state and
    it simply looks for numbers matching a certain pattern (`DATUM` tokens). In the
    parser, these are combined together in a list until a `NEWLINE` token is identified,
    at which point the list representing a line of the data file is added to the list
    representing the whole data set.

    With the data parsed, the next step is to parse the circuit definition which is
    included in the output file. This is not only necessary in order to simulate the
    circuit again, natively, but also for understanding the meaning of the data columns
    identified in the last step. The parsing of this metadata is handled by identifying
    in turn the sections in the commented block below the data that correspond to the
    various parts of the circuit definition. As these have different formats, the lexer
    enters into different states once it identifies each section, with special lexing
    rules. The identified tokens are passed to the parser, which pieces them together
    in set patterns. Once a particular line is parsed (or lines, in the case of op-amps),
    the particular combination of tokens used to create the line is used to create the
    circuit.
    """
    # text to ignore in op-amp list
    OPAMP_IGNORE_STRINGS = [
        "*OVR*", # overridden parameter flag
        "s***DEFAULT", # default parameter used
        "***DEFAULT"
    ]

    # additional states
    # avoid using underscores here to stop PLY sharing rules across states
    states = (
        ('resistors', 'inclusive'),
        ('capacitors', 'inclusive'),
        ('inductors', 'inclusive'),
        ('mutualinductances', 'inclusive'),
        ('opamps', 'inclusive'),
        ('nodes', 'inclusive'),
        ('voltageoutputnodes', 'inclusive'),
        ('currentoutputcomponents', 'inclusive'),
        ('noisevoltageoutputs', 'inclusive'),      # node voltage noise and plotted noise
        ('noisecurrentoutputs', 'inclusive'),      # component current noise and plotted noise
        ('noisysources', 'inclusive'),             # calculated noise
        ('gnuplotoptions', 'inclusive'),           # used to prevent mis-parsing of gnuplot options
                                                   # as something else
    )

    # data lexer tokens
    tokens = [
        'DATUM',
        'NEWLINE',
        # circuit elements
        'RESISTOR',
        'CAPACITOR',
        'INDUCTOR',
        'MUTUAL_INDUCTANCE',
        'OPAMP_CHUNK_1', # op-amps are split across up to 4 lines
        'OPAMP_CHUNK_2',
        'OPAMP_CHUNK_3',
        'OPAMP_CHUNK_4',
        'NODE',
        # inputs/outputs
        'VOLTAGE_OUTPUT_NODE',
        'CURRENT_OUTPUT_COMPONENT',
        'NOISE_OUTPUTS',             # plotted noise
        'NOISY_SOURCES',             # calculated noise
    ]

    # data point (scientific notation float, or +/- inf)
    t_DATUM = r'-?(inf|(\d+\.\d*|\d*\.\d+|\d+)([eE]-?\d*\.?\d*)?)'

    # ignore comments (sometimes)
    # this is overridden by methods below; some do parse comments
    t_ignore_COMMENT = r'\#.*'

    # ignore spaces and tabs (always)
    t_ignore = ' \t'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Circuit input component, created using the output file, necessary for building solution
        # functions.
        self._input_component = None

    @property
    def _default_circuit_properties(self):
        extra = {"n_resistors": None,
                 "n_capacitors": None,
                 "n_inductors": None,
                 "n_mutual_inductances": None,
                 "n_opamps": None,
                 "n_nodes": None,
                 "n_voltage_outputs": None,
                 "n_current_outputs": None,
                 "n_noise_sources": None,
                 "n_noise": None,
                 "n_noisy": None,
                 # Data lists from parsed file.
                 "raw_data": [],
                 # Index of noise source sum column.
                 "source_sum_index": None}

        return {**super()._default_circuit_properties, **extra}

    @property
    def n_resistors(self):
        return self._circuit_properties["n_resistors"]

    @n_resistors.setter
    def n_resistors(self, n_resistors):
        self._circuit_properties["n_resistors"] = int(n_resistors)

    @property
    def n_capacitors(self):
        return self._circuit_properties["n_capacitors"]

    @n_capacitors.setter
    def n_capacitors(self, n_capacitors):
        self._circuit_properties["n_capacitors"] = int(n_capacitors)

    @property
    def n_inductors(self):
        return self._circuit_properties["n_inductors"]

    @n_inductors.setter
    def n_inductors(self, n_inductors):
        self._circuit_properties["n_inductors"] = int(n_inductors)

    @property
    def n_mutual_inductances(self):
        return self._circuit_properties["n_mutual_inductances"]

    @n_mutual_inductances.setter
    def n_mutual_inductances(self, n_mutual_inductances):
        self._circuit_properties["n_mutual_inductances"] = int(n_mutual_inductances)

    @property
    def n_opamps(self):
        return self._circuit_properties["n_opamps"]

    @n_opamps.setter
    def n_opamps(self, n_opamps):
        self._circuit_properties["n_opamps"] = int(n_opamps)

    @property
    def n_nodes(self):
        return self._circuit_properties["n_nodes"]

    @n_nodes.setter
    def n_nodes(self, n_nodes):
        self._circuit_properties["n_nodes"] = int(n_nodes)

    @property
    def n_voltage_outputs(self):
        return self._circuit_properties["n_voltage_outputs"]

    @n_voltage_outputs.setter
    def n_voltage_outputs(self, n_voltage_outputs):
        self._circuit_properties["n_voltage_outputs"] = int(n_voltage_outputs)

    @property
    def n_current_outputs(self):
        return self._circuit_properties["n_current_outputs"]

    @n_current_outputs.setter
    def n_current_outputs(self, n_current_outputs):
        self._circuit_properties["n_current_outputs"] = int(n_current_outputs)

    @property
    def n_noise_sources(self):
        return self._circuit_properties["n_noise_sources"]

    @n_noise_sources.setter
    def n_noise_sources(self, n_noise_sources):
        self._circuit_properties["n_noise_sources"] = int(n_noise_sources)

    @property
    def n_noise(self):
        return self._circuit_properties["n_noise"]

    @n_noise.setter
    def n_noise(self, n_noise):
        self._circuit_properties["n_noise"] = int(n_noise)

    @property
    def n_noisy(self):
        return self._circuit_properties["n_noisy"]

    @n_noisy.setter
    def n_noisy(self, n_noisy):
        self._circuit_properties["n_noisy"] = int(n_noisy)

    @property
    def source_sum_index(self):
        return self._circuit_properties["source_sum_index"]

    @source_sum_index.setter
    def source_sum_index(self, source_sum_index):
        self._circuit_properties["source_sum_index"] = int(source_sum_index)

    def _do_build(self):
        super()._do_build()

        # Parse data.
        data = np.array(self._circuit_properties["raw_data"])

        # Frequencies are the first data column.
        self.frequencies = data[:, 0]

        # The rest is data.
        data = data[:, 1:]

        # Create input component.
        if self.input_node_n is None:
            # Grounded input.
            nodes = [Node("gnd"), self.input_node_p]
        else:
            # Floating input.
            nodes = [self.input_node_n, self.input_node_p]
        self._input_component = Input(nodes, input_type=self.input_type)

        # Create solution.
        self._build_solution(data)

    def _build_solution(self, data):
        self._solution = Solution(self.frequencies)

        if self.output_type == "response":
            self._build_responses(data)
        elif self.output_type == "noise":
            self._build_noise(data)
        else:
            raise ValueError(f"unrecognised output type '{self.output_type}'")

    def _build_responses(self, data):
        # column offset
        offset = 0

        for response_output in self.response_outputs:
            # get data
            if response_output.has_real and response_output.has_imag:
                real_index, _ = response_output.real_index
                imag_index, _ = response_output.imag_index

                # get data
                real_data = data[:, offset + real_index]
                imag_data = data[:, offset + imag_index]

                # create data series
                series = Series.from_re_im(x=self.frequencies, re=real_data, im=imag_data)
            elif response_output.has_magnitude or response_output.has_phase:
                # dict to contain Series arguments
                series_data = {}

                if response_output.has_magnitude:
                    mag_index, mag_scale = response_output.magnitude_index

                    # get magnitude data
                    series_data["magnitude"] = data[:, offset + mag_index]
                    series_data["mag_scale"] = mag_scale

                if response_output.has_phase:
                    phase_index, phase_scale = response_output.phase_index

                    # get phase data
                    series_data["phase"] = data[:, offset + phase_index]
                    series_data["phase_scale"] = phase_scale

                # create data series
                series = Series.from_mag_phase(x=self.frequencies, **series_data)
            else:
                raise ValueError("cannot build solution without either magnitude or phase, or "
                                 "both real and imaginary data columns present")

            if self.input_type == "voltage":
                source = self.input_node_p
            elif self.input_type == "current":
                source = self._input_component
            else:
                raise ValueError("invalid input type")

            if response_output.OUTPUT_TYPE == "voltage":
                sink = self.circuit.get_node(response_output.node)
            elif response_output.OUTPUT_TYPE == "current":
                sink = self.circuit[response_output.component]
            else:
                raise ValueError("invalid output type")

            function = Response(series=series, source=source, sink=sink)

            self._solution.add_response(function)

            # increment offset
            offset += response_output.n_scales

    def _build_noise(self, data):
        """Build noise outputs"""
        if self.input_refer:
            if self.input_type == "voltage":
                sink = self.input_node_p
            elif self.input_type == "current":
                sink = self._input_component
            else:
                raise ValueError("invalid input type")
        else:
            # The data sink is the noise output element.
            sink = self.circuit[self.noise_output_element]

        # Now that we have all the noise sources, create noise outputs.
        for index, noisy_element in enumerate(self.noisy_elements):
            # Get component.
            component = self.circuit[noisy_element.component]

            # Get data.
            series = Series(x=self.frequencies, y=data[:, index])

            if isinstance(component, OpAmp) and noisy_element.has_suffix:
                if noisy_element.has_opamp_voltage_noise:
                    # Voltage noise.
                    noise = component.voltage_noise
                elif noisy_element.has_opamp_non_inv_current_noise:
                    # Non-inverting input current noise.
                    noise = component.non_inv_current_noise
                elif noisy_element.has_opamp_inv_current_noise:
                    # Inverting input current noise.
                    noise = component.inv_current_noise
                else:
                    self.p_error("unrecognised op-amp noise type")
            else:
                # Must be a resistor.
                noise = component.johnson_noise

            # Create noise spectral density.
            spectral_density = NoiseDensity(source=noise, sink=sink, series=series)

            self._solution.add_noise(spectral_density)

        # Generate sum if present.
        if self.source_sum_index is not None:
            # Get sources contributing to sum.
            sources = self.summed_noise_objects

            # Get data.
            series = Series(x=self.frequencies, y=data[:, self.source_sum_index])

            # Create and store sum noise.
            sum_noise = MultiNoiseDensity(sources=sources, sink=sink, series=series)
            self._solution.add_noise_sum(sum_noise, default=True)

            # Flag that noise sum must be generated for any future native runs of this circuit.
            self._circuit_properties["noise_sum_to_be_computed"] = True

    def t_ANY_resistors(self, t):
        # match start of resistor section
        r'\#(?P<n>\d+)\sresistors?:'
        self.n_resistors = t.lexer.lexmatch.group('n')
        t.lexer.begin('resistors')

    def t_ANY_capacitors(self, t):
        # match start of capacitor section
        r'\#(?P<n>\d+)\scapacitors?:'
        self.n_capacitors = t.lexer.lexmatch.group('n')
        t.lexer.begin('capacitors')

    def t_ANY_inductors(self, t):
        # match start of inductor section
        r'\#(?P<n>\d+)\scoils?:'
        self.n_inductors = t.lexer.lexmatch.group('n')
        t.lexer.begin('inductors')

    def t_ANY_mutualinductances(self, t):
        # match start of mutual inductance section
        r'\#(?P<n>\d+)\smutual\sinductances?:'
        self.n_mutual_indutances = t.lexer.lexmatch.groups('n')
        t.lexer.begin('mutualinductances')

    def t_ANY_opamps(self, t):
        # match start of op-amp section
        r'\#(?P<n>\d+)\sop-amps?:'
        self.n_opamps = t.lexer.lexmatch.group('n')
        t.lexer.begin('opamps')

    def t_ANY_nodes(self, t):
        # match start of node section
        r'\#(?P<n>\d+)\snodes?:'
        self.n_nodes = t.lexer.lexmatch.group('n')
        t.lexer.begin('nodes')

    def t_ANY_inputrefer(self, t):
        r'\#Noise\sis\sINPUT-REFERRED\sto\s(.*)\sinput\sat\snode\s(.*).'
        self.input_refer = True

    def t_ANY_noisysources(self, t):
        # match start of noise source section
        r'\#Noise\sis\scomputed\s(?P<ntype>at\snode|through\scomponent)\s(?:.+\:)?(?P<element>.+)\sfor\s\(nnoise=(?P<nnoise>\d+),\snnoisy=(?P<nnoisy>\d+)\)\s:'
        self.output_type = "noise"
        self.n_noise = t.lexer.lexmatch.group('nnoise')
        self.n_noisy = t.lexer.lexmatch.group('nnoisy')
        self.noise_output_element = t.lexer.lexmatch.group('element')
        t.lexer.begin('noisysources')

    def t_ANY_voltageinput(self, t):
        r'\#Voltage\sinput\sat\snode\s(?P<node>.+),\simpedance\s(?P<impedance>.+)Ohm'
        self.input_type = "voltage"
        self.input_node_p = t.lexer.lexmatch.group('node')
        self.input_impedance = t.lexer.lexmatch.group('impedance')

    def t_ANY_floatingvoltageinput(self, t):
        r'\#Floating\svoltage\sinput\sbetween\snodes\s(?P<node_p>.+)\sand\s(?P<node_n>.+),\simpedance\s(?P<impedance>.+)Ohm'
        self.input_type = "voltage"
        self.input_node_p = t.lexer.lexmatch.group('node_p')
        self.input_node_n = t.lexer.lexmatch.group('node_n')
        self.input_impedance = t.lexer.lexmatch.group('impedance')

    def t_ANY_currentinput(self, t):
        r'\#Current\sinput\sinto\snode\s(?P<node>.+),\simpedance\s(?P<impedance>.+)Ohm'
        self.input_type = "current"
        self.input_node_p = t.lexer.lexmatch.group('node')
        self.input_impedance = t.lexer.lexmatch.group('impedance')

    def t_ANY_voltageoutputnodes(self, t):
        # match start of voltage output section
        r'\#OUTPUT\s(?P<nout>\d+)\svoltage\soutputs?:'
        self.output_type = "response"
        self.n_voltage_outputs = t.lexer.lexmatch.group('nout')
        t.lexer.begin('voltageoutputnodes')

    def t_ANY_currentoutputcomponents(self, t):
        # match start of current output section
        r'\#OUTPUT\s(?P<nout>\d+)\scurrent\soutputs?:'
        self.output_type = "response"
        self.n_current_outputs = t.lexer.lexmatch.group('nout')
        t.lexer.begin('currentoutputcomponents')

    def t_ANY_noisevoltageoutputs(self, t):
        # match start of noise voltage output section
        r'\#OUTPUT\s(?P<nsource>\d+)\snoise\svoltages?\scaused\sby:'
        self.n_noise_sources = t.lexer.lexmatch.group('nsource')
        t.lexer.begin('noisevoltageoutputs')

    def t_ANY_noisecurrentoutputs(self, t):
        # match start of noise current output section
        r'\#OUTPUT\s(?P<nsource>\d+)\snoise\scurrents?\scaused\sby:'
        self.n_noise_sources = t.lexer.lexmatch.group('nsource')
        t.lexer.begin('noisecurrentoutputs')

    def t_ANY_noiseinputreferred(self, t):
        r'\#\sNoise\sis\sREFERRED\sTO\sTHE\sINPUT.'
        pass

    def t_ANY_gnuplotoptions(self, t):
        # match start of gnuplot section
        r'\#\d+\sGNUPLOT.*'
        t.lexer.begin('gnuplotoptions')

    def t_resistors_RESISTOR(self, t):
        r'\#\s+\d+\s+(?P<resistor>.*)'
        t.type = "RESISTOR"
        t.value = t.lexer.lexmatch.group('resistor')
        return t

    def t_capacitors_CAPACITOR(self, t):
        r'\#\s+\d+\s+(?P<capacitor>.*)'
        t.type = "CAPACITOR"
        t.value = t.lexer.lexmatch.group('capacitor')
        return t

    def t_inductors_INDUCTOR(self, t):
        r'\#\s+\d+\s+(?P<inductor>.*)'
        t.type = "INDUCTOR"
        t.value = t.lexer.lexmatch.group('inductor')
        return t

    def t_mutualinductances_MUTUAL_INDUCTANCE(self, t):
        r'\#\s+\d+\s+(?P<mutual_inductance>.*)'
        t.type = "MUTUAL_INDUCTANCE"
        t.value = t.lexer.lexmatch.group('mutual_inductance')
        return t

    def t_opamps_OPAMP_CHUNK_1(self, t):
        r'\#\s+\d+\s+(?P<opamp1>.*)'
        t.type = "OPAMP_CHUNK_1"
        t.value = t.lexer.lexmatch.group('opamp1')
        return t

    def t_opamps_OPAMP_CHUNK_2(self, t):
        r'\#\s+(?P<opamp2>un=.*)'
        t.type = "OPAMP_CHUNK_2"
        t.value = t.lexer.lexmatch.group('opamp2')
        return t

    def t_opamps_OPAMP_CHUNK_3(self, t):
        r'\#\s+(?P<opamp3>umax=.*)'
        t.type = "OPAMP_CHUNK_3"
        t.value = t.lexer.lexmatch.group('opamp3')
        return t

    def t_opamps_OPAMP_CHUNK_4(self, t):
        r'\#\s+(?P<opamp4>pole.*)'
        t.type = "OPAMP_CHUNK_4"
        t.value = t.lexer.lexmatch.group('opamp4')
        return t

    def t_nodes_NODE(self, t):
        r'\#\s+\d+\s+(?P<node>.*)'
        t.type = "NODE"
        t.value = t.lexer.lexmatch.group('node')
        return t

    def t_voltageoutputnodes_VOLTAGE_OUTPUT_NODE(self, t):
        r'\#\s+(?P<index>\d+)\snode:\s(?P<node>.*)'
        t.type = "VOLTAGE_OUTPUT_NODE"
        t.value = (t.lexer.lexmatch.group('index'), t.lexer.lexmatch.group('node'))
        return t

    def t_currentoutputcomponents_CURRENT_OUTPUT_COMPONENT(self, t):
        r'\#\s+(?P<index>\d+)\s(?P<component>.*)'
        t.type = "CURRENT_OUTPUT_COMPONENT"
        t.value = (t.lexer.lexmatch.group('index'), t.lexer.lexmatch.group('component'))
        return t

    def t_noisevoltageoutputs_noisecurrentoutputs_NOISE_OUTPUTS(self, t):
        r'\#\s*(?P<components>.*)'
        t.type = "NOISE_OUTPUTS"
        t.value = t.lexer.lexmatch.group('components')
        return t

    def t_noisysources_NOISY_SOURCES(self, t):
        r'\#\s+(?P<components>.*)'
        t.type = "NOISY_SOURCES"
        t.value = t.lexer.lexmatch.group('components')
        return t

    def t_gnuplotoptions(self, t):
        r'\#.*'
        # ignore

    # detect new lines
    def t_newline(self, t):
        r'\n+'
        self.lineno += len(t.value)
        self._previous_newline_position = t.lexer.lexpos
        # generate newline token
        t.type = "NEWLINE"
        return t

    def p_file_contents(self, p):
        '''file_contents : file_line
                         | file_contents file_line'''
        # do nothing
        pass

    def p_file_line(self, p):
        # a line of data or a comment line
        '''file_line : data_line
                     | metadata_line'''

    def p_data_line(self, p):
        # list of measurements on a line of its own
        '''data_line : data NEWLINE
                     | NEWLINE'''
        # only when we find tokens
        if len(p) == 2:
            # skip empty line
            return

        # add new row to data
        self._circuit_properties["raw_data"].append(p[1])

    def p_data(self, p):
        # list of measurements
        '''data : data datum
                | datum'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_datum(self, p):
        '''datum : DATUM'''
        # convert to float (LISO always converts to %g, i.e. shortest between %f and %e)
        p[0] = float(p[1])

    def p_metadata_line(self, p):
        # metadata on its own line, e.g. comment or resistor definition
        '''metadata_line : resistor
                         | capacitor
                         | inductor
                         | mutual_inductance
                         | opamp
                         | node
                         | voltage_output_node
                         | current_output_component
                         | noise_outputs
                         | noisy_sources'''

        instruction = p[1]
        p[0] = instruction

    def p_resistor(self, p):
        '''resistor : RESISTOR NEWLINE'''
        resistor_str = p[1]
        p[0] = resistor_str

        self._parse_passive("r", resistor_str)

    def p_capacitor(self, p):
        '''capacitor : CAPACITOR NEWLINE'''
        capacitor_str = p[1]
        p[0] = capacitor_str

        self._parse_passive("c", capacitor_str)

    def p_inductor(self, p):
        '''inductor : INDUCTOR NEWLINE'''
        inductor_str = p[1]
        p[0] = inductor_str

        self._parse_passive("l", inductor_str)

    def p_mutual_inductance(self, p):
        '''mutual_inductance : MUTUAL_INDUCTANCE NEWLINE'''
        mutual_inductance_str = p[1]
        p[0] = mutual_inductance_str

        self._parse_mutual_inductance(mutual_inductance_str)

    def p_opamp(self, p):
        # join lines of op-amp definition together
        '''opamp : OPAMP_CHUNK_1 NEWLINE OPAMP_CHUNK_2 NEWLINE OPAMP_CHUNK_3 NEWLINE OPAMP_CHUNK_4 NEWLINE
                 | OPAMP_CHUNK_1 NEWLINE OPAMP_CHUNK_2 NEWLINE OPAMP_CHUNK_3 NEWLINE'''
        # join without newlines
        opamp_str = " ".join(p[1::2])
        p[0] = opamp_str

        self._parse_opamp(opamp_str)

    def p_node(self, p):
        '''node : NODE NEWLINE'''
        p[0] = p[1]

    def p_voltage_output_node(self, p):
        '''voltage_output_node : VOLTAGE_OUTPUT_NODE NEWLINE'''
        output = p[1]
        p[0] = output

        self._parse_voltage_output(output)

    def p_current_output_component(self, p):
        '''current_output_component : CURRENT_OUTPUT_COMPONENT NEWLINE'''
        output_str = p[1]
        p[0] = output_str

        self._parse_current_output(output_str)

    def p_noise_outputs(self, p):
        '''noise_outputs : NOISE_OUTPUTS NEWLINE'''
        source_str = p[1]
        p[0] = source_str

        self._parse_noise_outputs(source_str)

    def p_noisy_sources(self, p):
        '''noisy_sources : NOISY_SOURCES NEWLINE'''
        source_str = p[1]
        p[0] = source_str

        self._parse_noisy_sources(source_str)

    def p_error(self, p):
        lineno = self.lineno

        if p:
            if hasattr(p, 'value'):
                # parser object
                # check for unexpected new line
                if p.value == "\n":
                    message = "unexpected end of line"
                    # compensate for mistaken newline
                    lineno -= 1
                else:
                    message = f"'{p.value}'"
            else:
                # error message thrown by production
                message = str(p)

                if lineno is not None:
                    # error while parsing
                    # productions always end with newlines, so errors in productions are on previous
                    # lines
                    lineno -= 1
        else:
            message = "unexpected end of file"

        raise LisoParserError(message, line=lineno)

    def _parse_passive(self, passive_type, component_str):
        # Split by whitespace.
        tokens = component_str.split()

        ntokens = len(tokens)
        if ntokens != 5:
            self.p_error(f"unexpected parameter count ({ntokens})")

        # Splice together value and unit.
        tokens[1:3] = [''.join(tokens[1:3])]

        arg_names = ["name", "value", "node1", "node2"]
        kwargs = {name: value for name, value in zip(arg_names, tokens)}

        if passive_type == "r":
            self.circuit.add_resistor(**kwargs)
        elif passive_type == "c":
            self.circuit.add_capacitor(**kwargs)
        elif passive_type == "l":
            self.circuit.add_inductor(**kwargs)
        else:
            self.p_error(f"unrecognised passive component '{passive_type}'")

    def _parse_mutual_inductance(self, mutual_indutance_str):
        # Split by whitespace.
        tokens = mutual_indutance_str.split()

        ntoken = len(tokens)
        if ntoken != 4:
            self.p_error(f"unexpected parameter count ({ntoken})")

        # Pack tokens.
        couplings = tuple(tokens)

        self._circuit_properties["inductor_couplings"].append(couplings)

    def _parse_opamp(self, opamp_str):
        # remove ignored strings
        opamp_str = self._remove_ignored_opamp_strings(opamp_str)

        # split by whitespace
        params = iter(opamp_str.split())

        # op-amp name and model
        name = next(params)
        model = next(params)
        # in+, in- and out nodes, with first characters stripped out
        node1 = next(params).lstrip("'+'=")
        node2 = next(params).lstrip("'-'=")
        node3 = next(params).lstrip("'out'=")

        # default op-amp constructor keywords
        kwargs = {"poles": [],
                  "zeros": []}

        for param in params:
            # get rid of any remaining whitespace
            param = param.strip()

            if not param.startswith("pole") and not param.startswith("zero"):
                prop, value = param.split("=")
            else:
                prop = param

            if prop.startswith("a0"):
                kwargs["a0"] = value
            elif prop.startswith("gbw"):
                # Combine number with unit since LISO often (always?) specifies combinations of
                # scientific notation and SI prefixes.
                kwargs["gbw"] = Quantity(str(float(value)) + next(params))
            elif prop.startswith("un"):
                units = next(params)
                # split off "/sqrt(Hz)"
                units = units.rstrip("/sqrt(Hz)")
                # parse as V
                kwargs["vnoise"] = Quantity(str(float(value)) + units)
            elif prop.startswith("uc"):
                kwargs["vcorner"] = Quantity(str(float(value)) + next(params))
            elif prop.startswith("in"):
                units = next(params)
                # split off "/sqrt(Hz)"
                units = units.rstrip("/sqrt(Hz)")
                # parse as A
                kwargs["inoise"] = Quantity(str(float(value)) + units)
            elif prop.startswith("ic"):
                kwargs["icorner"] = Quantity(str(float(value)) + next(params))
            elif prop.startswith("umax"):
                kwargs["vmax"] = Quantity(str(float(value)) + next(params))
            elif prop.startswith("imax"):
                kwargs["imax"] = Quantity(str(float(value)) + next(params))
            elif prop.startswith("sr"):
                next(params)
                # parse without unit to avoid warning
                slew_rate = Quantity(str(float(value)), "V/s")
                # convert from V/us to V/s
                slew_rate *= 1e6
                kwargs["sr"] = slew_rate
            elif prop.startswith("delay"):
                if value != "0":
                    units = next(params)
                else:
                    units = ""
                kwargs["delay"] = Quantity(str(float(value)) + units)
            elif prop.startswith("pole"):
                # skip "at"
                next(params)

                # frequency and its unit is next two params
                frequency = next(params) + next(params)

                # plane is next
                plane = next(params)

                kwargs["poles"].extend(self._parse_opamp_root(frequency, plane))
            elif prop.startswith("zero"):
                # skip "at"
                next(params)

                # frequency and its unit is next two params
                frequency = next(params) + next(params)

                # plane is next
                plane = next(params)

                kwargs["zeros"].extend(self._parse_opamp_root(frequency, plane))
            else:
                self.p_error(f"unknown op-amp override parameter '{prop}'")

        self.circuit.add_opamp(name=name, model=model, node1=node1, node2=node2, node3=node3,
                               **kwargs)

    @classmethod
    def _remove_ignored_opamp_strings(cls, opamp_str):
        for ignore in cls.OPAMP_IGNORE_STRINGS:
            opamp_str = opamp_str.replace(ignore, "")

        return opamp_str

    def _parse_opamp_root(self, frequency, plane):
        # parse frequency
        frequency = Quantity(frequency, "Hz")

        plane = plane.lstrip("(").rstrip(")")

        roots = []

        if plane == "real":
            roots.append(frequency)
        else:
            q_factor = plane.split("=")[1]

            # calculate complex frequency using q-factor
            q_factor = Quantity(q_factor)
            theta = np.arccos(1 / (2 * q_factor))

            # add negative/positive pair of poles/zeros
            roots.append(frequency * np.exp(-1j * theta))
            roots.append(frequency * np.exp(1j * theta))

        return sorted(roots)

    def _parse_voltage_output(self, output):
        self._add_voltage_output(output)

    def _add_voltage_output(self, output):
        index, output_str = output

        # split by colon
        params = output_str.split()

        node = params[0]
        scales = params[1:]

        sink = LisoOutputVoltage(node=node, scales=scales, index=index)

        try:
            self.add_response_output(sink)
        except ValueError:
            self.p_error(f"voltage output '{sink}' already specified")

    def _parse_current_output(self, output):
        self._add_current_output(output)

    def _add_current_output(self, output):
        index, output_str = output

        # split by colon
        params = output_str.split()

        # get rid of component type in first param
        component = params[0].split(":")[1]

        scales = params[1:]
        sink = LisoOutputCurrent(component=component, scales=scales, index=index)

        try:
            self.add_response_output(sink)
        except ValueError:
            self.p_error(f"current output '{sink}' already specified")

    def _parse_noise_outputs(self, outputs_line):
        """Parse noise outputs representing columns of the data file"""
        # split by whitespace
        noise_output_strs = outputs_line.split()

        for data_index, noise_output_str in enumerate(noise_output_strs):
            self._parse_noise_output(noise_output_str, data_index)

    def _parse_noise_output(self, output, data_index):
        # strip any remaining whitespace
        output = output.strip()

        # look for bracket
        output_pieces = output.split("(")

        # component name is first piece
        component_name = output_pieces[0]

        if component_name == "sum":
            # this is a sum column
            # (don't set self._source_sum = True as this regenerates the sum)
            self.source_sum_index = data_index

            # nothing more to do
            return

        if len(output_pieces) > 1:
            # Extract op-amp noise type.
            port = output_pieces[1].rstrip(")")
        else:
            port = None

        try:
            self.add_noisy_element(LisoNoisyElement(component=component_name, suffix=port,
                                                    index=data_index))
        except ValueError:
            self.p_error(f"noise source '{component_name}' already specified")

    def _parse_noisy_sources(self, sources_line):
        """Parse noise sources used to calculate the noise outputs."""
        # split by whitespace
        noise_source_strs = sources_line.split()

        for noise_source_str in noise_source_strs:
            self._parse_noisy_source(noise_source_str)

    def _parse_noisy_source(self, source):
        """Get the noise definition for a given source."""
        # strip any remaining whitespace
        source = source.strip()

        # look for bracket
        source_pieces = source.split("(")

        # component name is first piece
        component_name = source_pieces[0]

        if len(source_pieces) > 1:
            # Extract op-amp noise type.
            port = source_pieces[1].rstrip(")")
        else:
            port = None

        try:
            self.add_noisy_sum_element(LisoNoisyElement(component=component_name, suffix=port))
        except ValueError:
            self.p_error(f"noise sum source '{component_name}' already specified")
