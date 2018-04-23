#! /usr/bin/python3

"""
Input:
Text file with one place (or person) name per line.
Output:
Text file with one place (or person) name per line in which all letters 
are separated by spaces and the spaces are transformed to <blanks>.
"""

import sys

def preprocess(inf, outf, lowercase=True):
    with open(outf, mode="w") as out:
        for line in open(inf):
            if lowercase:
                line = line.lower()
            line = [ c  if c != " " else "<blank>" for c in line][:-1]
            line = " ".join(line)
            out.write(line+"\n")
            
        
    

def main():
    preprocess(sys.argv[1], sys.argv[2])
    

if __name__ == "__main__":
    main()
