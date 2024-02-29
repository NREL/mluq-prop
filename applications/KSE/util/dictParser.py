import sys


def parseInputFile(fname=None):

    if fname is not None:
        input_fileName = fname
    else:
        print("No input file name found, assuming it is 'input'")
        input_fileName = "input"

    # ~~~~ Parse input
    inpt = {}
    f = open(input_fileName)
    data = f.readlines()
    for line in data:
        if ":" in line:
            key, value = line.split(":")
            inpt[key.strip()] = value.strip()
    f.close()

    return inpt
