from ..config import ZeroConfig

# solvers
from .scipy import ScipySolver

CONF = ZeroConfig()

# available solver classes
solver_classes = [ScipySolver]

# dict of solver names and types
available_solvers = {_class.NAME: _class for _class in solver_classes}

# get solver from preferences
solver_name = CONF["algebra"]["solver"].lower()

if solver_name not in available_solvers:
    raise ValueError("Invalid solver \"%s\" specified in configuration. Choose from %s."
                     % (solver_name, ", ".join(available_solvers)))

# get default solver
DefaultSolver = available_solvers[solver_name]
