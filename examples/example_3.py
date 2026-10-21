"""
example_3.py shows CAD-to-CSG decomposition using multiprocessing
"""

import mcio
import argparse


# place all code wthin a guard to stop repeted running of code during multiprocess
if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs="+", required=True, help='step file path')
    parser.add_argument('--cores', '-c', type=int, default=20, help='number of cores to run multiprocessing')
    args = parser.parse_args()

    # set up decomposer and run via multiprocess
    decomposer = mcio.decompose.Decomposer(step_file_paths=args.input)
    decomposer.run(cores=args.cores, debug=args.debug)
