import tensorflow as tf
import glob
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

class TFDataManager:
    """
    Manager for reading and processing TFRecord files for protein sequence pairs.
    Supports distributed reading, shuffling, batching, and dynamic padding.
    """
    
    def __init__(
        self,
        tfrecord_dir: str,
        batch_size: int = 32,
        shuffle_buffer: int = 10000,
        num_parallel_reads: int = 4,
        prefetch_size: int = tf.data.AUTOTUNE,
        vocab_size: int = 21,  # 20 amino acids + 1 masked token
    ):
        """
        Initialize TFDataManager.
        
        Args:
            tfrecord_dir: Directory containing TFRecord files
            batch_size: Batch size
            shuffle_buffer: Size of shuffle buffer
            num_parallel_reads: Number of files to read in parallel
            prefetch_size: Prefetch buffer size
            vocab_size: Size of vocabulary (21 for amino acids + mask)
        """
        self.tfrecord_dir = Path(tfrecord_dir)
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.num_parallel_reads = num_parallel_reads
        self.prefetch_size = prefetch_size
        self.vocab_size = vocab_size
    
    def _parse_tfrecord_fn(self, example_proto):
        """Parse a single TFRecord example."""
        feature_description = {
            'input_seq': tf.io.VarLenFeature(tf.int64),
            'output_seq': tf.io.VarLenFeature(tf.int64),
            'input_len': tf.io.FixedLenFeature([1], tf.int64),
            'output_len': tf.io.FixedLenFeature([1], tf.int64),
        }
        
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Convert sparse to dense
        input_seq = tf.sparse.to_dense(parsed['input_seq'])
        output_seq = tf.sparse.to_dense(parsed['output_seq'])
        input_len = parsed['input_len'][0]
        output_len = parsed['output_len'][0]
        
        return {
            'input_seq': input_seq,
            'output_seq': output_seq,
            'input_len': input_len,
            'output_len': output_len,
        }
    
    def _get_tfrecord_files(self, split: str) -> List[str]:
        """Get all TFRecord files for a given split."""
        pattern = str(self.tfrecord_dir / f"{split}*.tfrecord")
        files = sorted(glob.glob(pattern))
        if not files:
            raise ValueError(f"No TFRecord files found for split '{split}' in {self.tfrecord_dir}")
        return files
    
    def _pad_and_one_hot_encode(self, batch):
        """
        Pad sequences in batch to max length and one-hot encode.
        
        Args:
            batch: Dictionary with 'input_seq', 'output_seq', 'input_len', 'output_len'
        
        Returns:
            Tuple of (input_tensor, output_tensor) with shapes:
                - input_tensor: (B, max_input_len, vocab_size)
                - output_tensor: (B, max_output_len, vocab_size)
        """
        input_seqs = batch['input_seq']
        output_seqs = batch['output_seq']
        
        # Get max lengths in this batch
        max_input_len = tf.reduce_max(batch['input_len'])
        max_output_len = tf.reduce_max(batch['output_len'])
        
        # Pad sequences (pad with 0, which we'll one-hot encode safely)
        input_seqs_padded = tf.keras.preprocessing.sequence.pad_sequences(
            input_seqs.numpy(),
            maxlen=max_input_len.numpy(),
            padding='post',
            value=0
        )
        output_seqs_padded = tf.keras.preprocessing.sequence.pad_sequences(
            output_seqs.numpy(),
            maxlen=max_output_len.numpy(),
            padding='post',
            value=0
        )
        
        # One-hot encode
        input_one_hot = tf.one_hot(input_seqs_padded, depth=self.vocab_size)
        output_one_hot = tf.one_hot(output_seqs_padded, depth=self.vocab_size)
        
        return input_one_hot, output_one_hot
    
    def create_dataset(
        self,
        split: str = 'train',
        shuffle: bool = True,
        repeat: bool = True,
        augment: bool = False,
        augment_fn: Optional[callable] = None,
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset for training/evaluation.
        
        Args:
            split: Data split ('train', 'valid', or 'test')
            shuffle: Whether to shuffle data
            repeat: Whether to repeat dataset infinitely
            augment: Whether to apply augmentation
            augment_fn: Custom augmentation function
        
        Returns:
            tf.data.Dataset yielding (input_tensor, output_tensor) tuples
        """
        # Get TFRecord files
        tfrecord_files = self._get_tfrecord_files(split)
        print(f"Found {len(tfrecord_files)} TFRecord files for split '{split}'")
        
        # Create dataset from file list
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
        
        # Shuffle files
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(tfrecord_files))
        
        # Interleave reading from multiple files in parallel
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=self.num_parallel_reads,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
        
        # Parse examples
        dataset = dataset.map(
            self._parse_tfrecord_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation if specified
        if augment and augment_fn is not None:
            dataset = dataset.map(
                augment_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Shuffle examples
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer)
        
        # Repeat if needed
        if repeat:
            dataset = dataset.repeat()
        
        # Batch
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        
        # Pad and one-hot encode (using py_function for dynamic padding)
        dataset = dataset.map(
            lambda x: tf.py_function(
                func=self._pad_and_one_hot_encode,
                inp=[x],
                Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Prefetch
        dataset = dataset.prefetch(buffer_size=self.prefetch_size)
        
        return dataset
    
    def get_dataset_size(self, split: str) -> int:
        """
        Get approximate number of examples in a split.
        Note: This counts all records, which can be slow for large datasets.
        """
        tfrecord_files = self._get_tfrecord_files(split)
        count = 0
        for filepath in tfrecord_files:
            count += sum(1 for _ in tf.data.TFRecordDataset(filepath))
        return count

# Example usage
if __name__ == "__main__":
    # Initialize data manager
    data_manager = TFDataManager(
        tfrecord_dir="data/string/tfrecords",
        batch_size=32,
        shuffle_buffer=10000,
        num_parallel_reads=4
    )
    
    # Create training dataset
    train_dataset = data_manager.create_dataset(
        split='train',
        shuffle=True,
        repeat=True
    )
    
    # Test iteration
    for batch_idx, (inputs, outputs) in enumerate(train_dataset.take(3)):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Output shape: {outputs.shape}")