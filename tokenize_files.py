import tiktoken
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
import numpy as np
from config import DecoderConfig
import tqdm
import argparse


config = DecoderConfig()

def tokenize_line(line):
    tokens = tiktoken.get_encoding(config.encoding_name).encode(line)
    return tokens


def process_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        lines = file.read().replace('\xa0', u' ').splitlines(True)
    
    # Use process_map to tokenize lines in parallel
    p = Pool(4)
    tokenized_lines = list(tqdm.tqdm(p.imap(tokenize_line, lines,chunksize = 1024),total=len(lines)))
    # Convert the list of tokenized lines to a numpy array
    # Select minum 
    tokenized_array = np.array(tokenized_lines,dtype=object)
    return tokenized_array

def save_tokenized_array(array, output_path):
    # Save the numpy array to a file
    np.save(output_path, array)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--output')
    parser.add_argument('-v', dest='verbose', action='store_true')
    args = parser.parse_args()
    
    input_file_path = "war_and_piece_untouched.txt"  # Path to your input text file
    output_file_path = "war_and_piece_untouched.npy"  # Path to save the tokenized numpy array

    tokenized_array = process_file(input_file_path)
    save_tokenized_array(tokenized_array, output_file_path)