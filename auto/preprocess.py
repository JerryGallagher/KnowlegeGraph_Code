#preProcess.py

import os
import argparse
from findsource import *
from tokenizer import *


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Search for source code files in a directory and subdirectories.")
    parser.add_argument("directory", type=str, help="The directory to search for source code files.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of subdirectories to search (default: no limit).")

    # Parse command-line arguments
    args = parser.parse_args()

    # Search for source code files
    source_code_files = find_source_code_files(args.directory, args.max_depth)
    print(source_code_files)
    # Output the result
    print(f"Found {len(source_code_files)} source code files:")
    for source_file in source_code_files:
        print(source_file)
        file = open(source_file, "r",errors='ignore')
        print(file.read())           
        #print(tokenizer(source_file))


if __name__ == "__main__":
    main()