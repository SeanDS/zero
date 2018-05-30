import numpy as np
import logging
import re

from ..components import CurrentNoise, VoltageNoise, JohnsonNoise
from ..solution import Solution
from ..format import SIFormatter
from .base import LisoParser, LisoOutputElement

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

    # op-amp roots
    OPAMP_ROOT_REGEX = re.compile(r"(pole|zero) at ([\w\d\s\.]+) "
                                  r"\((real|Q=([\w\d\s\.]+))\)")

    # noise components
    # "o1(I+)"
    NOISE_COMPONENT_REGEX = re.compile(r"^([\w\d]+)(\(([\w\d\-\+]*)\))?$")

    # additional states
    # avoid using underscores here to stop PLY sharing rules across states
    states = (
        ('resistors', 'inclusive'),
        ('capacitors', 'inclusive'),
        ('inductors', 'inclusive'),
        ('opamps', 'inclusive'),
        ('nodes', 'inclusive'),
        ('noisesourcecomponents', 'inclusive'),
        ('voltageoutputnodes', 'inclusive'),
        ('currentoutputcomponents', 'inclusive'),
        ('noisesourcenodes', 'inclusive'),
        ('gnuplotoptions', 'inclusive'), # used to prevent mis-parsing of gnuplot options as something else
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
        'NOISE_SOURCE_COMPONENTS',
        'VOLTAGE_OUTPUT_NODE',
        'CURRENT_OUTPUT_COMPONENT',
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

        # data in numpy array format
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

        super(LisoOutputParser, self).__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        # add input component, if not yet present
        self._set_circuit_input()
        
        # parse data
        self._data = np.array(self._raw_data)

        # frequencies are the first data column
        self.frequencies = self._data[:, 0]

        # create solution
        #self.build_solution()
        
        # clear data
        self._data = None

    def build_solution(self):
        self._solution = Solution(self.circuit, self.frequencies)

        # add functions
        for function in self.functions:
            solution.add_function(function)

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
        r'\#(?P<n>\d+)\sinductors?:'
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

    def t_ANY_noiseinputs(self, t):
        # match start of noise node section
        r'\#Noise\sis\scomputed\sat\snode\s(?P<node>.+)\sfor\s\(nnoise=(?P<nnoise>\d+),\snnoisy=(?P<nnoisy>\d+)\)\s:'
        self.output_type = "noise"
        self.nnoise = t.lexer.lexmatch.group('nnoise')
        self.nnoisy = t.lexer.lexmatch.group('nnoisy')
        self.noise_output_node = t.lexer.lexmatch.group('node')
        t.lexer.begin('noisesourcecomponents')

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

    def t_ANY_noisesourcenodes(self, t):
        # match start of noise output section
        r'\#OUTPUT\s(?P<nsource>\d+)\snoise\svoltages?\scaused\sby:'
        self.nnoisesources = t.lexer.lexmatch.group('nsource')
        t.lexer.begin('noisesourcenodes')

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

    def t_noisesourcecomponents_NOISE_SOURCE_COMPONENTS(self, t):
        r'\#\s+(?P<components>.*)'
        t.type = "NOISE_SOURCE_COMPONENTS"
        t.value = t.lexer.lexmatch.group('components')
        return t

    def t_voltageoutputnodes_VOLTAGE_OUTPUT_NODE(self, t):
        r'\#\s+\d+\snode:\s(?P<node>.*)'
        t.type = "VOLTAGE_OUTPUT_NODE"
        t.value = t.lexer.lexmatch.group('node')
        return t

    def t_currentoutputcomponents_CURRENT_OUTPUT_COMPONENT(self, t):
        r'\#\s+\d+\s(?P<component>.*)'
        t.type = "CURRENT_OUTPUT_COMPONENT"
        t.value = t.lexer.lexmatch.group('component')
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
                         | noise_source_components
                         | voltage_output_node
                         | current_output_component'''
        
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

    def p_noise_source_components(self, p):
        '''noise_source_components : NOISE_SOURCE_COMPONENTS NEWLINE'''
        source_str = p[1]
        p[0] = source_str

        self.parse_noise_sources(source_str)

    def p_voltage_output_node(self, p):
        '''voltage_output_node : VOLTAGE_OUTPUT_NODE NEWLINE'''
        output_str = p[1]
        p[0] = output_str

        self.parse_voltage_output(output_str)

    def p_current_output_component(self, p):
        '''current_output_component : CURRENT_OUTPUT_COMPONENT NEWLINE'''
        output_str = p[1]
        p[0] = output_str

        self.parse_current_output(output_str)

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

        arg_names = ["name", "value", "node1", "node2", "node3"]
        kwargs = {name: value for name, value in zip(arg_names, tokens)}

        if passive_type == "r":
            return self.circuit.add_resistor(**kwargs)
        elif passive_type == "c":
            return self.circuit.add_capacitor(**kwargs)
        elif passive_type == "l":
            return self.circuit.add_inductor(**kwargs)
        
        raise SyntaxError

    def parse_opamp(self, opamp_str):
        # split by whitespace
        params = opamp_str.split()

        # remove ignored strings
        params = self._remove_ignored_opamp_strings(params)

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
                if value != "0":
                    unit = next(params)
                else:
                    unit = ""

                kwargs["gbw"] = value + unit
            elif prop.startswith("un"):
                if value != "0":
                    unit = next(params)
                    # split off "/sqrt(Hz)"
                    unit = unit.rstrip("/sqrt(Hz)")
                else:
                    unit = ""

                kwargs["v_noise"] = value + unit
            elif prop.startswith("uc"):
                if value != "0":
                    unit = next(params)
                else:
                    unit = ""

                kwargs["v_corner"] = value + unit
            elif prop.startswith("in"):
                if value != "0":
                    unit = next(params)
                    # split off "/sqrt(Hz)"
                    unit = unit.rstrip("/sqrt(Hz)")
                else:
                    unit = ""

                kwargs["i_noise"] = value + unit
            elif prop.startswith("ic"):
                if value != "0":
                    unit = next(params)
                else:
                    unit = ""

                kwargs["i_corner"] = value + unit
            elif prop.startswith("umax"):
                if value != "0":
                    unit = next(params)
                else:
                    unit = ""

                kwargs["v_max"] = value + unit
            elif prop.startswith("imax"):
                if value != "0":
                    unit = next(params)
                else:
                    unit = ""

                kwargs["i_max"] = value + unit
            elif prop.startswith("sr"):
                if value != "0":
                    unit = next(params)

                    # parse V/us and convert to V/s
                    # FIXME: parse inverse units directly
                    slew_rate, _ = SIFormatter.parse(value + unit)
                    slew_rate *= 1e6
                else:
                    slew_rate = 0

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
    def _remove_ignored_opamp_strings(cls, params):
        for param in params:
            for ignore in cls.OPAMP_IGNORE_STRINGS:
                param = param.replace(ignore, "")
            
            # only return non-empty strings
            if param:
                yield param

    def _parse_opamp_root(self, frequency, plane):
        # parse frequency
        frequency, _ = SIFormatter.parse(frequency)

        plane = plane.lstrip("(").rstrip(")")

        roots = []

        if plane == "real":
            roots.append(frequency)
        else:
            q_factor = plane.split("=")[1]

            # calculate complex frequency using q-factor
            q_factor, _ = SIFormatter.parse(q_factor)
            theta = np.arccos(1 / (2 * q_factor))

            # add negative/positive pair of poles/zeros
            roots.append(frequency * np.exp(-1j * theta))
            roots.append(frequency * np.exp(1j * theta))

        return sorted(roots)

    def parse_noise_sources(self, sources_line):
        if self.noise_sources:
            raise SyntaxError("noise sources already set")
        
        # split by whitespace
        source_strs = sources_line.split()

        sources = []

        for source_str in source_strs:
            sources.append(self._parse_noise_source(source_str))
        
        self.noise_sources = sources

    def _parse_noise_source(self, source_str):
        # split by whitespace
        sources = source_str.split()

        for source in sources:
            # strip any remaining whitespace
            source = source.strip()

        # look for component name and brackets
        match = re.match(self.NOISE_COMPONENT_REGEX, source_str)

        # find extracted component
        component = self.circuit.get_component(match.group(1))

        # component noise type, e.g. I+ (or empty, for resistors)
        noise_str = match.group(3)

        # group 2 is not empty if noise type is specified
        if match.group(2):
            # op-amp noise; check first character
            if noise_str[0] == "U":
                noise_source = VoltageNoise(component=component)
            elif noise_str[0] == "I":
                # work out node
                if noise_str[1] == "+":
                    # non-inverting node
                    node = component.node1
                elif noise_str[1] == "-":
                    # inverting node
                    node = component.node2
                else:
                    raise SyntaxError("unexpected current noise node")

                noise_source = CurrentNoise(node=node, component=component)
            else:
                raise SyntaxError("unrecognised noise source")
        else:
            noise_source = JohnsonNoise(component=component,
                                        resistance=component.resistance)

        return noise_source

    def parse_voltage_output(self, output_str):
        self.add_voltage_output(output_str)

    def add_voltage_output(self, output):
        # split by colon
        params = output.split()

        node = self.circuit.get_node(params[0])
        scales = params[1:]

        # TODO: can "all" be set here?
        output = LisoOutputElement(node, scales)

        if output in self.tf_outputs:
            raise SyntaxError("output already specified")
            
        self.tf_outputs.append(output)

    def parse_current_output(self, output_str):
        self.add_current_output(output_str)

    def add_current_output(self, output):
        # split by colon
        params = output.split()

        # get rid of component type in first param
        component_name = params[0].split(":")[1]

        component = self.circuit.get_component(component_name)
        scales = params[1:]

        # TODO: can "all" be set here?
        output = LisoOutputElement(component, scales)

        if output in self.tf_outputs:
            raise SyntaxError("output component '%s' already specified" % output)
        
        self.tf_outputs.append(output)