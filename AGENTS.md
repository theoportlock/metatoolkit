When making new scripts for python, there should always be separate main, parseargs, and the other function names.
Input should be either first argument(s) or specified if needed.
Output should be -o or --output for individual file or --outputdir for direcotry
When an output is made, the base directory of the path should be made if not existant
All inputs and outputs should be tsv and index column is zero on read
There should always be some printed output to summarize the log of the output
no default output files
When scripts are made, test data should be included in the tests/data directory
