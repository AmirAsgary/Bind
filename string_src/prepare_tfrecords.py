import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Amino acid vocabulary (20 standard amino acids + masked token)
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX['X'] = 20  # Masked/Unknown token
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}

def parse_fasta(fasta_path: str) -> Dict[str, str]:
    """Parse FASTA file and return dictionary of protein_id -> sequence."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # Add last sequence
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences

def encode_sequence(sequence: str) -> List[int]:
    """Encode amino acid sequence to integer indices."""
    encoded = []
    for aa in sequence:
        if aa in AA_TO_IDX:
            encoded.append(AA_TO_IDX[aa])
        else:
            encoded.append(AA_TO_IDX['X'])  # Unknown amino acid
    return encoded

def create_tf_example(seq1: List[int], seq2: List[int]) -> tf.train.Example:
    """Create a TensorFlow Example from two encoded sequences."""
    feature = {
        'input_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=seq1)),
        'output_seq': tf.train.Feature(int64_list=tf.train.Int64List(value=seq2)),
        'input_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(seq1)])),
        'output_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[len(seq2)])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def prepare_tfrecords(
    links_path: str,
    fasta_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    max_examples_per_file: int = 100000,
    min_score: int = 0
):
    """
    Prepare TFRecord files from STRING database files.
    
    Args:
        links_path: Path to protein.physical.links file
        fasta_path: Path to protein.sequences.fa file
        output_dir: Output directory for TFRecords
        train_ratio: Ratio of training data
        valid_ratio: Ratio of validation data
        max_examples_per_file: Maximum examples per TFRecord file
        min_score: Minimum combined score to include
    """
    
    print("Parsing FASTA file...")
    sequences = parse_fasta(fasta_path)
    print(f"Loaded {len(sequences)} protein sequences")
    
    print("Reading protein links...")
    pairs = []
    with open(links_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                protein1, protein2, score = parts[0], parts[1], int(parts[2])
                if score >= min_score and protein1 in sequences and protein2 in sequences:
                    pairs.append((protein1, protein2))
    
    print(f"Found {len(pairs)} valid protein pairs")
    
    # Create bidirectional pairs (protein1->protein2 and protein2->protein1)
    all_examples = []
    for p1, p2 in pairs:
        seq1 = encode_sequence(sequences[p1])
        seq2 = encode_sequence(sequences[p2])
        
        # Add both directions
        all_examples.append((seq1, seq2))
        all_examples.append((seq2, seq1))
    
    print(f"Total examples (bidirectional): {len(all_examples)}")
    
    # Shuffle and split
    random.shuffle(all_examples)
    n_total = len(all_examples)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    
    train_examples = all_examples[:n_train]
    valid_examples = all_examples[n_train:n_train + n_valid]
    test_examples = all_examples[n_train + n_valid:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Write TFRecords
    for split_name, split_data in [
        ('train', train_examples),
        ('valid', valid_examples),
        ('test', test_examples)
    ]:
        print(f"\nWriting {split_name} split ({len(split_data)} examples)...")
        
        n_files = (len(split_data) + max_examples_per_file - 1) // max_examples_per_file
        
        for file_idx in range(n_files):
            start_idx = file_idx * max_examples_per_file
            end_idx = min((file_idx + 1) * max_examples_per_file, len(split_data))
            
            if n_files > 1:
                filename = f"{split_name}_{file_idx:04d}.tfrecord"
            else:
                filename = f"{split_name}.tfrecord"
            
            filepath = output_path / filename
            
            with tf.io.TFRecordWriter(str(filepath)) as writer:
                for seq1, seq2 in split_data[start_idx:end_idx]:
                    example = create_tf_example(seq1, seq2)
                    writer.write(example.SerializeToString())
            
            print(f"  Written {filepath} ({end_idx - start_idx} examples)")
    
    print("\nDone!")
    print(f"Train: {len(train_examples)} examples")
    print(f"Valid: {len(valid_examples)} examples")
    print(f"Test: {len(test_examples)} examples")

if __name__ == "__main__":
    prepare_tfrecords(
        links_path="data/large/protein.physical.links.v12.0.txt",
        fasta_path="data/large/protein.sequences.v12.0.fa",
        output_dir="data/string/tfrecords",
        train_ratio=0.8,
        valid_ratio=0.1,
        max_examples_per_file=100000,
        min_score=0  # Adjust minimum score threshold if needed
    )