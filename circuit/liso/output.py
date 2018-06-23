import numpy as np
import logging
import re

from ..components import CurrentNoise, VoltageNoise, JohnsonNoise
from ..solution import Solution
from ..data import (Series, ComplexSeries, VoltageVoltageTF, VoltageCurrentTF, CurrentVoltageTF,
                    CurrentCurrentTF, NoiseSpectrum, SumNoiseSpectrum)
from ..format import Quantity
from .base import LisoParser, LisoOutputVoltage, LisoOutputCurrent, LisoNoiseSource

LOGGER = logging.getLogger("liso")

class LisoOutputFormatException(Exception):
    pass


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
        ('opamps', 'inclusive'),
        ('nodes', 'inclusive'),
        ('voltageoutputnodes', 'inclusive'),
        ('currentoutputcomponents', 'inclusive'),
        ('noiseoutputs', 'inclusive'),             # plotted noise
        ('noisysources', 'inclusive'),             # calculated noise
        ('gnuplotoptions', 'inclusive'),           # used to prevent mis-parsing of gnuplot options as something else
    )

    # data lexer tokens
    tokens = [
        'DATUM',
        'NEWLINE',
        # circuit elements
        'RESISTOR',
        'CAPACITOR',
        'INDUCTOR',
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
        # raw data lists from parsed file
        self._raw_data = []

        # data in float array
        self._data = None

        # number of each element as reported by the output file
        self.nresistors = None
        self.ncapacitors = None
        self.ninductors = None
        self.nopamps = None
        self.nnodes = None
        self.nvoutputs = None
        self.nioutputs = None
        self.nnoisesources = None

        # raw noise source lists
        self._noise_defs = [] # displayed noise
        self._noisy_defs = [] # computed noise, including extra sources included in "sum"

        super().__init__(*args, **kwargs)

    def _do_build(self, *args, **kwargs):
        # call parent
        super()._do_build(*args, **kwargs)

        # parse data
        data = np.array(self._raw_data)

        # frequencies are the first data column
        self.frequencies = data[:, 0]

        # the rest is data
        self._data = data[:, 1:]

        # create solution
        self.build_solution()
        
        # clear data
        self._data = None

    def build_solution(self):
        self._solution = Solution(self.circuit, self.frequencies)

        if self.output_type == "tf":
            self._build_tfs()
        elif self.output_type == "noise":
            self._build_noise()
        else:
            raise ValueError("unrecognised output type")

    def _build_tfs(self):
        # column offset
        offset = 0

        for tf_output in self.tf_outputs:
            # get data
            if tf_output.has_real and tf_output.has_imag:
                real_index, real_scale = tf_output.real_index
                imag_index, imag_scale = tf_output.imag_index

                raise NotImplementedError("cannot handle real and imaginary data yet")
            elif tf_output.has_magnitude or tf_output.has_phase:
                if tf_output.has_magnitude:
                    mag_index, mag_scale = tf_output.magnitude_index

                    # get magnitude data
                    mag_data = self._data[:, offset + mag_index]
                
                if tf_output.has_phase:
                    phase_index, phase_scale = tf_output.phase_index

                    # get phase data
                    phase_data = self._data[:, offset + phase_index]
            else:
                raise ValueError("cannot build solution without either magnitude or phase, or "
                                 "both real and imaginary data columns present")

            # create data series
            series = ComplexSeries(x=self.frequencies, magnitude=mag_data,
                                   phase=phase_data, magnitude_scale=mag_scale,
                                   phase_scale=phase_scale)

            # create appropriate transfer function depending on analysis
            if self.input_type == "voltage":
                source = self.input_node_p

                if tf_output.output_type == "voltage":
                    sink = self.circuit.get_node(tf_output.node)
                    function = VoltageVoltageTF(series=series, source=source, sink=sink)
                elif tf_output.output_type == "current":
                    sink = self.circuit.get_component(tf_output.component)
                    function = VoltageCurrentTF(series=series, source=source, sink=sink)
                else:
                    raise ValueError("invalid output type")
            elif self.input_type == "current":
                source = self.circuit.get_component("input")

                if tf_output.output_type == "voltage":
                    sink = self.circuit.get_node(tf_output.node)
                    function = CurrentVoltageTF(series=series, source=source, sink=sink)
                elif tf_output.output_type == "current":
                    sink = self.circuit.get_component(tf_output.component)
                    function = CurrentCurrentTF(series=series, source=source, sink=sink)
                else:
                    raise ValueError("invalid output type")
            else:
                raise ValueError("invalid input type")

            self._solution.add_tf(function)

            # increment offset
            offset += tf_output.n_scales

    def _build_noise(self):
        """Build noise outputs"""
        # now that we have all the noise sources, create noise outputs
        for index, definition in enumerate(self._noise_defs):
            component = self.circuit.get_component(definition[0])

            # get data
            series = Series(x=self.frequencies, y=self._data[:, index])

            # data sink is the noise output node
            sink = self.noise_output_node

            if component == "sum":
                # get source spectra
                sources = self.noise_sources

                # create sum noise
                spectrum = SumNoiseSpectrum(sources=sources, sink=sink, series=series)
            else:
                if len(definition) > 1:
                    # op-amp noise type specified
                    noise_type_id = int(definition[1])

                    if noise_type_id == 0:
                        noise = component.voltage_noise
                    elif noise_type_id == 1:
                        # non-inverting input current noise
                        noise = component.non_inv_current_noise
                    elif noise_type_id == 2:
                        # inverting input current noise
                        noise = component.inv_current_noise
                    else:
                        raise SyntaxError("unrecognised op-amp noise id '%s'" % noise_type_id)
                else:
                    # must be a resistor
                    noise = component.johnson_noise
            
                # noise should always be in the noise source list
                #assert noise in self.noise_sources

                # create noise spectrum
                spectrum = NoiseSpectrum(source=noise, sink=sink, series=series)

            self._solution.add_noise(spectrum)

    @LisoParser.noise_sources.getter
    def noise_sources(self):
        """Noise sources to be plotted"""
        return set(self._get_noise_sources(self._noise_defs))

    @LisoParser.noise_sum_sources.getter
    def noise_sum_sources(self):
        """Noise sources included in the sum column"""
        return set(self._get_noise_sources(self._noisy_defs))

    def t_ANY_resistors(self, t):
        # match start of resistor section
        r'\#(?P<n>\d+)\sresistors?:'
        self.nresistors = t.lexer.lexmatch.group('n')
        t.lexer.begin('resistors')

    def t_ANY_capacitors(self, t):
        # match start of capacitor section
        r'\#(?P<n>\d+)\scapacitors?:'
        self.ncapacitors = t.lexer.lexmatch.group('n')
        t.lexer.begin('capacitors')

    def t_ANY_inductors(self, t):
        # match start of inductor section
        r'\#(?P<n>\d+)\scoils?:'
        self.ninductors = t.lexer.lexmatch.group('n')
        t.lexer.begin('inductors')

    def t_ANY_opamps(self, t):
        # match start of op-amp section
        r'\#(?P<n>\d+)\sop-amps?:'
        self.nopamps = t.lexer.lexmatch.group('n')
        t.lexer.begin('opamps')

    def t_ANY_nodes(self, t):
        # match start of node section
        r'\#(?P<n>\d+)\snodes?:'
        self.nnodes = t.lexer.lexmatch.group('n')
        t.lexer.begin('nodes')

    def t_ANY_noisysources(self, t):
        # match start of noise node section
        r'\#Noise\sis\scomputed\sat\snode\s(?P<node>.+)\sfor\s\(nnoise=(?P<nnoise>\d+),\snnoisy=(?P<nnoisy>\d+)\)\s:'
        self.output_type = "noise"
        self.nnoise = t.lexer.lexmatch.group('nnoise')
        self.nnoisy = t.lexer.lexmatch.group('nnoisy')
        self.noise_output_node = t.lexer.lexmatch.group('node')
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
        self.output_type = "tf"
        self.nvoutputs = t.lexer.lexmatch.group('nout')
        t.lexer.begin('voltageoutputnodes')

    def t_ANY_currentoutputcomponents(self, t):
        # match start of current output section
        r'\#OUTPUT\s(?P<nout>\d+)\scurrent\soutputs?:'
        self.output_type = "tf"
        self.nioutputs = t.lexer.lexmatch.group('nout')
        t.lexer.begin('currentoutputcomponents')

    def t_ANY_noiseoutputs(self, t):
        # match start of noise output section
        r'\#OUTPUT\s(?P<nsource>\d+)\snoise\svoltages?\scaused\sby:'
        self.nnoisesources = t.lexer.lexmatch.group('nsource')
        t.lexer.begin('noiseoutputs')

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

    def t_noiseoutputs_NOISE_OUTPUTS(self, t):
        r'\#\s+(?P<components>.*)'
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

    # error handling
    def t_error(self, t):
        # anything that gets past the other filters
        raise SyntaxError("Illegal character '%s' on line %i at position %i" %
                          (t.value[0], self.lineno, t.lexer.lexpos - self._previous_newline_position))
    
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
        self._raw_data.append(p[1])

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

        self.parse_passive("r", resistor_str)

    def p_capacitor(self, p):
        '''capacitor : CAPACITOR NEWLINE'''
        capacitor_str = p[1]
        p[0] = capacitor_str

        self.parse_passive("c", capacitor_str)

    def p_inductor(self, p):
        '''inductor : INDUCTOR NEWLINE'''
        inductor_str = p[1]
        p[0] = inductor_str

        self.parse_passive("l", inductor_str)

    def p_opamp(self, p):
        # join lines of op-amp definition together
        '''opamp : OPAMP_CHUNK_1 NEWLINE OPAMP_CHUNK_2 NEWLINE OPAMP_CHUNK_3 NEWLINE OPAMP_CHUNK_4 NEWLINE
                 | OPAMP_CHUNK_1 NEWLINE OPAMP_CHUNK_2 NEWLINE OPAMP_CHUNK_3 NEWLINE'''
        # join without newlines
        opamp_str = " ".join(p[1::2])
        p[0] = opamp_str

        self.parse_opamp(opamp_str)

    def p_node(self, p):
        '''node : NODE NEWLINE'''
        p[0] = p[1]

    def p_voltage_output_node(self, p):
        '''voltage_output_node : VOLTAGE_OUTPUT_NODE NEWLINE'''
        output = p[1]
        p[0] = output

        self.parse_voltage_output(output)

    def p_current_output_component(self, p):
        '''current_output_component : CURRENT_OUTPUT_COMPONENT NEWLINE'''
        output_str = p[1]
        p[0] = output_str

        self.parse_current_output(output_str)

    def p_noise_outputs(self, p):
        '''noise_outputs : NOISE_OUTPUTS NEWLINE'''
        source_str = p[1]
        p[0] = source_str

        self.parse_noise_outputs(source_str)

    def p_noisy_sources(self, p):
        '''noisy_sources : NOISY_SOURCES NEWLINE'''
        source_str = p[1]
        p[0] = source_str

        self.parse_noisy_sources(source_str)

    def p_error(self, p):
        if p:
            error_msg = "LISO syntax error '%s' at line %i" % (p.value, self.lineno)
        else:
            error_msg = "LISO syntax error at end of file"
        
        raise SyntaxError(error_msg)

    def parse_passive(self, passive_type, component_str):
        # split by whitespace
        tokens = component_str.split()

        if len(tokens) != 5:
            # TODO: add the error
            raise SyntaxError("LISO syntax error")

        # splice together value and unit
        tokens[1:3] = [''.join(tokens[1:3])]

        arg_names = ["name", "value", "node1", "node2"]
        kwargs = {name: value for name, value in zip(arg_names, tokens)}

        if passive_type == "r":
            return self.circuit.add_resistor(**kwargs)
        elif passive_type == "c":
            return self.circuit.add_capacitor(**kwargs)
        elif passive_type == "l":
            return self.circuit.add_inductor(**kwargs)
        
        raise SyntaxError

    def parse_opamp(self, opamp_str):
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
                unit = next(params)
                kwargs["gbw"] = value + unit
            elif prop.startswith("un"):
                unit = next(params)
                # split off "/sqrt(Hz)"
                unit = unit.rstrip("/sqrt(Hz)")
                kwargs["v_noise"] = value + unit
            elif prop.startswith("uc"):
                unit = next(params)
                kwargs["v_corner"] = value + unit
            elif prop.startswith("in"):
                unit = next(params)
                # split off "/sqrt(Hz)"
                unit = unit.rstrip("/sqrt(Hz)")
                kwargs["i_noise"] = value + unit
            elif prop.startswith("ic"):
                unit = next(params)
                kwargs["i_corner"] = value + unit
            elif prop.startswith("umax"):
                unit = next(params)
                kwargs["v_max"] = value + unit
            elif prop.startswith("imax"):
                unit = next(params)
                kwargs["i_max"] = value + unit
            elif prop.startswith("sr"):
                unit = next(params)
                # parse V/us and convert to V/s
                slew_rate = Quantity(value + unit, "V/s")
                slew_rate *= 1e6
                kwargs["slew_rate"] = slew_rate
            elif prop.startswith("delay"):
                if value != "0":
                    unit = next(params)
                else:
                    unit = ""
                
                kwargs["delay"] = value + unit
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
                raise SyntaxError("unrecognised op-amp parameter")

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

    def parse_voltage_output(self, output):
        self.add_voltage_output(output)

    def add_voltage_output(self, output):
        index, output_str = output

        # split by colon
        params = output_str.split()

        node = params[0]
        scales = params[1:]

        # TODO: can "all" be set here?
        output = LisoOutputVoltage(node=node, scales=scales, index=index)

        if output in self.tf_outputs:
            raise SyntaxError("output already specified")
            
        self.tf_outputs.append(output)

    def parse_current_output(self, output):
        self.add_current_output(output)

    def add_current_output(self, output):
        index, output_str = output

        # split by colon
        params = output_str.split()

        # get rid of component type in first param
        component = params[0].split(":")[1]
        
        scales = params[1:]

        # TODO: can "all" be set here?
        output = LisoOutputCurrent(component=component, scales=scales, index=index)

        if output in self.tf_outputs:
            raise SyntaxError("output component '%s' already specified" % output)
        
        self.tf_outputs.append(output)

    def parse_noise_outputs(self, outputs_line):
        """Parse noise outputs representing columns of the data file"""
        # split by whitespace
        noise_output_strs = outputs_line.split()

        for noise_output_str in noise_output_strs:
            self._parse_noise_output(noise_output_str)

    def _parse_noise_output(self, output):
        # strip any remaining whitespace
        output = output.strip()

        # look for bracket
        output_pieces = output.split("(")

        # component name is first piece
        component_name = output_pieces[0]

        # individual component
        definition = [component_name]
        
        if len(output_pieces) > 1:
            # remove trailing bracket
            noise_type_id = output_pieces[1].rstrip(")")

            # add op-amp noise type
            definition.append(noise_type_id)
        
        self._noise_defs.append(definition)

    def parse_noisy_sources(self, sources_line):
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

        # individual component
        definition = [component_name]
        
        if len(source_pieces) > 1:
            # remove trailing bracket
            type_str = source_pieces[1].rstrip(")")

            # add op-amp noise type
            definition.append(type_str)
        
        self._noisy_defs.append(definition)