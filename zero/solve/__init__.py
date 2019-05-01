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
    available = ", ".join(available_solvers)
    raise ValueError(f"Invalid solver \"{solver_name}\" specified in configuration. Choose from "
                     f"{available}.")

# get default solver
DefaultSolver = available_solvers[solver_name]
