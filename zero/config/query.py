"""Library query parser"""

import logging
import operator
import fnmatch
from ply import lex, yacc

from ..config import OpAmpLibrary
from ..format import Quantity

LOGGER = logging.getLogger(__name__)
LIBRARY = OpAmpLibrary()


class LibraryQueryParser:
    """Op-amp library query parser.

    This implements a lexer to identify search terms for the Zero op-amp library, and returns
    lambda functions that perform corresponding checks.
    """
    # Parameter tokens.
    parameters = {
        "model": "MODEL",
        "a0": "OPEN_LOOP_GAIN",
        "gbw": "GAIN_BANDWIDTH",
        "vnoise": "VNOISE",
        "vcorner": "VNOISE_CORNER",
        "inoise": "INOISE",
        "icorner": "INOISE_CORNER",
        "vmax": "V_MAX",
        "imax": "I_MAX",
        "sr": "SLEW_RATE"
    }

    # top level tokens
    tokens = [
        'ID', # Numeric or string values.
        'PARAMETER',
        'EQUAL',
        'NOT_EQUAL',
        'GREATER_THAN',
        'GREATER_THAN_EQUAL',
        'LESS_THAN',
        'LESS_THAN_EQUAL',
        'AND',
        'OR',
        'LPAREN',
        'RPAREN'
    ]

    # operator method map
    _operators = {
        "==": "eq",
        "!=": "ne",
        ">": "gt",
        ">=": "ge",
        "<": "lt",
        "<=": "le",
        "&": "and_",
        "|": "or_"
    }

    # textual parameters (support wildcard comparisons)
    _text_params = {
        "model"
    }

    # ignore spaces and tabs
    t_ignore = ' \t'

    # simple tokens (don't need method)
    t_EQUAL = r'=='
    t_NOT_EQUAL = r'!='
    t_GREATER_THAN = r'>'
    t_GREATER_THAN_EQUAL = r'>='
    t_LESS_THAN = r'<'
    t_LESS_THAN_EQUAL = r'<='
    t_AND = r'\&'
    t_OR = r'\|'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'

    def __init__(self):
        # Parsed search filters.
        self._filters = None
        # Order in which parameters have been queried.
        self.parameter_query_order = []
        # Create lexer and parser handlers. Set lex and yacc to not generate grammar files, for
        # packaging simplicity, at the cost of a slight speed penalty.
        self.lexer = lex.lex(module=self, optimize=False, debug=False)
        self.parser = yacc.yacc(module=self, write_tables=False, debug=False)

    def parse(self, text):
        # clear existing filters
        self._filters = None
        self.parser.parse(text, lexer=self.lexer)
        return self._filters

    @classmethod
    def _get_comparison_method(cls, comparison, parameter):
        """Get appropriate comparison method for the specified comparison and parameter."""
        # check if this is a text comparison
        if parameter in cls._text_params:
            if comparison == getattr(operator, cls._operators["=="]):
                # equal
                comparison = cls._textual_equal
            elif comparison == getattr(operator, cls._operators["!="]):
                # not equal
                comparison = cls._textual_not_equal

        return comparison

    @classmethod
    def _textual_equal(cls, left, right):
        # slightly abuse unix file path matching
        return fnmatch.fnmatch(left.lower(), right.lower())

    @classmethod
    def _textual_not_equal(cls, left, right):
        return not cls._textual_equal(left, right)

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    # Error handling.
    def t_error(self, t):
        # Anything that gets past the other filters.
        raise ValueError(f"illegal character '{t.value[0]}' on line {t.lexer.lineno}")

    def t_eof(self, t):
        return None

    def t_ID(self, t):
        r'[a-zA-Z\?\*\d.-]+'
        if t.value.lower() in self.parameters:
            t.type = 'PARAMETER'
        return t

    def p_error(self, p):
        lineno = self.lexer.lineno

        if p:
            if hasattr(p, 'value'):
                # parser object
                # check for unexpected new line or end of file
                if p.type == "EOF":
                    message = "unexpected end of file"
                    # compensate for mistaken newline
                    lineno -= 1
                elif p.value.startswith("\n"):
                    message = "unexpected end of line"
                    # compensate for mistaken newlines
                    lineno -= p.value.count("\n")
                else:
                    message = f"'{p.value}'"
            else:
                # error message thrown by production
                message = str(p)

                # productions always end with newlines, so errors in productions are on previous
                # lines
                if lineno is not None:
                    lineno -= 1
        else:
            message = "unexpected end of file"

        raise LibraryParserError(message, line=lineno)

    def p_statement(self, t):
        'statement : expression'
        self._filters = t[1]

    def p_binary_operator(self, t):
        '''binary_operator : OR
                           | AND'''
        t[0] = getattr(operator, self._operators[t[1]])

    def p_comparison_operator(self, t):
        '''comparison_operator : EQUAL
                               | NOT_EQUAL
                               | GREATER_THAN
                               | GREATER_THAN_EQUAL
                               | LESS_THAN
                               | LESS_THAN_EQUAL'''
        t[0] = getattr(operator, self._operators[t[1]])

    def p_value_with_unit(self, t):
        'value_with_unit : ID ID'
        # Matches a value with a unit.
        t[0] = t[1] + t[2]

    def p_comparison_expression(self, t):
        '''expression : PARAMETER comparison_operator ID
                      | PARAMETER comparison_operator value_with_unit'''
        # parse value
        try:
            value = Quantity(t[3])
        except ValueError:
            # assume string
            value = t[3]

        parameter = t[1]
        comparison = t[2]

        if parameter not in self.parameter_query_order:
            # This parameter has not been seen yet.
            self.parameter_query_order.append(parameter)

        # change comparison method if necessary (e.g. text comparison)
        comparison = self._get_comparison_method(comparison, parameter)

        # create expression
        t[0] = lambda opamps: set([opamp for opamp in opamps
                                   if comparison(getattr(opamp, parameter), value)])

    def p_expression_group(self, t):
        'expression : LPAREN expression RPAREN'
        t[0] = t[2]

    def p_binary_expression(self, t):
        'expression : expression binary_operator expression'
        operation = t[2]
        lhs = t[1]
        rhs = t[3]
        # combine
        t[0] = lambda opamps: operation(lhs(opamps), rhs(opamps))


class LibraryParserError(ValueError):
    """Library parser error"""
    def __init__(self, message, line=None, pos=None, **kwargs):
        if line is not None:
            line = int(line)

            if pos is not None:
                pos = int(pos)

                # add line number and position
                message = f"{message} (line {line}, position {pos})"
            else:
                # add line number
                message = f"{message} (line {line})"

        # prepend message
        message = f"Syntax error: {message}"

        super().__init__(message, **kwargs)


class LibraryQueryEngine:
    """Query engine for op-amp library"""
    def __init__(self):
        self._parser = LibraryQueryParser()

    def query(self, text, sort_order=None):
        """Query the library.

        Parameters
        ----------
        text : :class:`str`
            The query text.
        sort_order : :class:`dict`, optional
            The sort order map. If specified, the items in this dictionary are used to determine the
            sort order for the returned results. The keys represent the parameter to filter, and
            the values represent the order (standard or reverse). The sorting is applied in the
            order that the parameter appears in the query text (left to right).

        Returns
        -------
        :class:`list`
            The matched op-amps.
        """
        expression = self._parser.parse(text)
        # Run query with op-amp set so we can use support for binary operators.
        parts = list(expression(self.opamp_set))
        if sort_order is not None:
            # Sort the results in the order they were specified (left to right).
            for parameter in reversed(self._parser.parameter_query_order):
                reverse = sort_order[parameter]
                parts = sorted(parts, key=lambda part: getattr(part, parameter), reverse=reverse)
        return parts

    @property
    def opamp_set(self):
        return set(LIBRARY.opamps)

    @property
    def parameters(self):
        return self._parser.parameters.keys()
