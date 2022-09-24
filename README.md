# Hong and Page model 

## Installation 
The project is built using Python 3.9, and has no external dependencies. 
Upon running ``python3 main.py`` it prints the results from the model and creates a file named ``results.csv``. 

## Command line options 
```
usage: main.py [-h] [-o file] [-N N] [-M M]

Run a Hong & Page style simulation.

optional arguments:
-h, --help  show this help message and exit
-o file     file to write results to (defaults to output.csv)
-N N        size of landscape (default 2000)
-M M        number of iterations per strategy (default 500)

