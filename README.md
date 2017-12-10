# electronics.py

Electronics calculator and simulator utility. This package provides tools
to calculate combinations of standard resistors to meet given criteria
(such as a target equivalent resistance or a regulator voltage), linear
circuit simulation and SI unit formatting, among other things.

## Installation
```bash
pip3 install git+https://github.com/SeanDS/electronics.py.git
```

## Program and library

### Component tools
Component tools are most easily utilised via the command line. Run
`electronics help` to get started.

### Simulator
Simulator code inspired by [Elektrotickle](https://github.com/tobin/Elektrotickle/)
by Tobin Fricke, which is itself based on [LISO](http://www2.mpq.mpg.de/~ros/geo600_docu/soft/liso/manual.pdf)
by Gerhard Heinzel.

## Future ideas
  - Export to Scipy system object
  - Visualise circuit with graphviz
  - Component optimisation: find best op-amp/resistor values to minimise
    noise in certain band

## Credits
Sean Leavey  
<electronics@attackllama.com>  
https://github.com/SeanDS
