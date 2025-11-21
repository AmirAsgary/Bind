import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
class ProteinSequenceAugmentation:
    """
    Augmentation strategies for protein sequences to prevent length-dependent learning.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.1,
        masking_rate: float = 0.1,
        clipping_prob: float = 0.5,
        min_clip_ratio: float = 0.5,
        vocab_size: int = 21,
        mask_token_id: int = 20,
    ):
        """
        Initialize augmentation parameters.
        
        Args:
            mutation_rate: Probability of mutating each amino acid
            masking_rate: Probability of masking each amino acid
            clipping_prob: Probability of applying clipping
            min_clip_ratio: Minimum ratio of sequence to keep when clipping
            vocab_size: Vocabulary size (20 amino acids + 1 mask token)
            mask_token_id: Index of mask token (21st token = index 20)
        """
        self.mutation_rate = mutation_rate
        self.masking_rate = masking_rate
        self.clipping_prob = clipping_prob
        self.min_clip_ratio = min_clip_ratio
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
    
    def random_clip_sequence(self, sequence: tf.Tensor, seq_len: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Randomly clip a portion of the sequence.
        
        Args:
            sequence: Sequence tensor of shape (seq_len,)
            seq_len: Original sequence length
        
        Returns:
            Tuple of (clipped_sequence, new_length)
        """
        # Decide whether to clip
        do_clip = tf.random.uniform([]) < self.clipping_prob
        
        if not do_clip:
            return sequence, seq_len
        
        # Calculate new length (between min_clip_ratio * length and full length)
        min_len = tf.cast(tf.cast(seq_len, tf.float32) * self.min_clip_ratio, tf.int32)
        min_len = tf.maximum(min_len, 1)  # At least 1 amino acid
        new_len = tf.random.uniform([], minval=min_len, maxval=seq_len + 1, dtype=tf.int32)
        
        # Random start position
        max_start = seq_len - new_len
        start = tf.random.uniform([], minval=0, maxval=max_start + 1, dtype=tf.int32)
        
        # Clip sequence
        clipped = sequence[start:start + new_len]
        
        return clipped, new_len
    
    def random_mutate_sequence(self, sequence: tf.Tensor, seq_len: tf.Tensor) -> tf.Tensor:
        """
        Randomly mutate amino acids in the sequence.
        
        Args:
            sequence: Sequence tensor of shape (seq_len,)
            seq_len: Sequence length
        
        Returns:
            Mutated sequence
        """
        # Create mutation mask (True where we should mutate)
        mutation_mask = tf.random.uniform([seq_len]) < self.mutation_rate
        
        # Generate random amino acid replacements (0 to vocab_size-2, excluding mask token)
        random_amino_acids = tf.random.uniform(
            [seq_len],
            minval=0,
            maxval=self.vocab_size - 1,  # Exclude mask token
            dtype=tf.int64
        )
        
        # Apply mutations
        mutated = tf.where(mutation_mask, random_amino_acids, sequence)
        
        return mutated
    
    def random_mask_sequence(self, sequence: tf.Tensor, seq_len: tf.Tensor) -> tf.Tensor:
        """
        Randomly mask amino acids in the sequence.
        
        Args:
            sequence: Sequence tensor of shape (seq_len,)
            seq_len: Sequence length
        
        Returns:
            Masked sequence
        """
        # Create masking mask (True where we should mask)
        masking_mask = tf.random.uniform([seq_len]) < self.masking_rate
        
        # Create mask tokens
        mask_tokens = tf.fill([seq_len], tf.cast(self.mask_token_id, tf.int64))
        
        # Apply masking
        masked = tf.where(masking_mask, mask_tokens, sequence)
        
        return masked
    
    def augment_example(self, example: dict) -> dict:
        """
        Apply all augmentations to a single example.
        
        Args:
            example: Dictionary with 'input_seq', 'output_seq', 'input_len', 'output_len'
        
        Returns:
            Augmented example dictionary
        """
        input_seq = example['input_seq']
        output_seq = example['output_seq']
        input_len = example['input_len']
        output_len = example['output_len']
        
        # 1. Random clipping (both input and output)
        input_seq, input_len = self.random_clip_sequence(input_seq, input_len)
        output_seq, output_len = self.random_clip_sequence(output_seq, output_len)
        
        # 2. Random mutation (only on input)
        input_seq = self.random_mutate_sequence(input_seq, input_len)
        
        # 3. Random masking (only on input)
        input_seq = self.random_mask_sequence(input_seq, input_len)
        
        return {
            'input_seq': input_seq,
            'output_seq': output_seq,
            'input_len': input_len,
            'output_len': output_len,
        }
    
    def get_augmentation_fn(self):
        """
        Get the augmentation function for use with tf.data.Dataset.map()
        
        Returns:
            Callable augmentation function
        """
        def augment_fn(example):
            return tf.py_function(
                func=lambda x: self.augment_example({
                    'input_seq': x['input_seq'],
                    'output_seq': x['output_seq'],
                    'input_len': x['input_len'],
                    'output_len': x['output_len'],
                }),
                inp=[example],
                Tout={
                    'input_seq': tf.int64,
                    'output_seq': tf.int64,
                    'input_len': tf.int64,
                    'output_len': tf.int64,
                }
            )
        
        return augment_fn


# Standalone augmentation function (can be used directly with TFDataManager)
def create_augmentation_fn(
    mutation_rate: float = 0.1,
    masking_rate: float = 0.1,
    clipping_prob: float = 0.5,
    min_clip_ratio: float = 0.5,
):
    """
    Create an augmentation function with specified parameters.
    
    Args:
        mutation_rate: Probability of mutating each amino acid
        masking_rate: Probability of masking each amino acid
        clipping_prob: Probability of applying clipping
        min_clip_ratio: Minimum ratio of sequence to keep when clipping
    
    Returns:
        Augmentation function compatible with tf.data.Dataset.map()
    """
    augmenter = ProteinSequenceAugmentation(
        mutation_rate=mutation_rate,
        masking_rate=masking_rate,
        clipping_prob=clipping_prob,
        min_clip_ratio=min_clip_ratio,
    )
    
    def augment(example):
        # Apply augmentations
        input_seq = example['input_seq']
        output_seq = example['output_seq']
        input_len = example['input_len']
        output_len = example['output_len']
        
        # 1. Random clipping
        input_seq, input_len = augmenter.random_clip_sequence(input_seq, input_len)
        output_seq, output_len = augmenter.random_clip_sequence(output_seq, output_len)
        
        # 2. Random mutation (only input)
        input_seq = augmenter.random_mutate_sequence(input_seq, input_len)
        
        # 3. Random masking (only input)
        input_seq = augmenter.random_mask_sequence(input_seq, input_len)
        
        return {
            'input_seq': input_seq,
            'output_seq': output_seq,
            'input_len': input_len,
            'output_len': output_len,
        }
    
    return augment


# Example usage
if __name__ == "__main__":
    from tf_data_manager import TFDataManager
    
    # Create augmentation function
    augment_fn = create_augmentation_fn(
        mutation_rate=0.1,
        masking_rate=0.1,
        clipping_prob=0.5,
        min_clip_ratio=0.5
    )
    
    # Initialize data manager
    data_manager = TFDataManager(
        tfrecord_dir="data/string/tfrecords",
        batch_size=32,
        shuffle_buffer=10000,
    )
    
    # Create training dataset with augmentation
    train_dataset = data_manager.create_dataset(
        split='train',
        shuffle=True,
        repeat=True,
        augment=True,
        augment_fn=augment_fn
    )
    
    # Test with augmentation
    print("Testing augmentation:")
    for batch_idx, (inputs, outputs) in enumerate(train_dataset.take(2)):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Output shape: {outputs.shape}")
        
        # Check for masked tokens (index 20)
        has_mask = tf.reduce_any(tf.argmax(inputs, axis=-1) == 20)
        print(f"  Contains masked tokens: {has_mask.numpy()}")