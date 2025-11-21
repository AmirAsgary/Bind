"""
Complete example showing how to use the data pipeline for VAE training.
"""

import tensorflow as tf
import string_src
from tensorflow import keras
from string_src.tf_data_manager import TFDataManager
from string_src.augmentation import create_augmentation_fn

# ============================================================================
# 1. PREPARE DATA (Run once to create TFRecords)
# ============================================================================
def prepare_data():
    """
    Run this once to prepare TFRecord files from raw data.
    """
    from string_src.prepare_tfrecords import prepare_tfrecords
    
    prepare_tfrecords(
        links_path="data/large/protein.physical.links.v12.0.txt",
        fasta_path="data/large/protein.sequences.v12.0.fa",
        output_dir="data/string/tfrecords",
        train_ratio=0.8,
        valid_ratio=0.1,
        max_examples_per_file=100000,
        min_score=0
    )


# ============================================================================
# 2. CREATE DATA PIPELINE FOR TRAINING
# ============================================================================
def create_data_pipeline(batch_size=32, use_augmentation=True):
    """
    Create training, validation, and test datasets.
    
    Args:
        batch_size: Batch size for training
        use_augmentation: Whether to use augmentation for training data
    
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    # Initialize data manager
    data_manager = TFDataManager(
        tfrecord_dir="data/string/tfrecords",
        batch_size=batch_size,
        shuffle_buffer=10000,
        num_parallel_reads=4,
        prefetch_size=tf.data.AUTOTUNE,
        vocab_size=21
    )
    
    # Create augmentation function
    if use_augmentation:
        augment_fn = create_augmentation_fn(
            mutation_rate=0.1,      # 10% mutation
            masking_rate=0.1,       # 10% masking
            clipping_prob=0.5,      # 50% chance of clipping
            min_clip_ratio=0.5      # Keep at least 50% of sequence
        )
    else:
        augment_fn = None
    
    # Create datasets
    train_dataset = data_manager.create_dataset(
        split='train',
        shuffle=True,
        repeat=True,
        augment=use_augmentation,
        augment_fn=augment_fn
    )
    
    valid_dataset = data_manager.create_dataset(
        split='valid',
        shuffle=False,
        repeat=False,
        augment=False,
        augment_fn=None
    )
    
    test_dataset = data_manager.create_dataset(
        split='test',
        shuffle=False,
        repeat=False,
        augment=False,
        augment_fn=None
    )
    
    return train_dataset, valid_dataset, test_dataset


# ============================================================================
# 3. SIMPLE VAE MODEL EXAMPLE (Replace with your own architecture)
# ============================================================================
class ProteinVAE(keras.Model):
    """
    Simple VAE example for protein sequences.
    You should replace this with your own architecture.
    """
    
    def __init__(self, latent_dim=256, vocab_size=21):
        super(ProteinVAE, self).__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        
        # Encoder
        self.encoder_lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(256, return_sequences=False)
        )
        self.z_mean = keras.layers.Dense(latent_dim)
        self.z_log_var = keras.layers.Dense(latent_dim)
        
        # Decoder
        self.decoder_repeat = keras.layers.RepeatVector(1)  # Will be dynamic
        self.decoder_lstm = keras.layers.LSTM(512, return_sequences=True)
        self.decoder_output = keras.layers.TimeDistributed(
            keras.layers.Dense(vocab_size, activation='softmax')
        )
    
    def encode(self, x):
        h = self.encoder_lstm(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, self.latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z, output_length):
        # Repeat z for each time step
        repeated_z = tf.tile(tf.expand_dims(z, 1), [1, output_length, 1])
        h = self.decoder_lstm(repeated_z)
        return self.decoder_output(h)
    
    def call(self, inputs, training=None):
        x_input, x_output = inputs
        output_length = tf.shape(x_output)[1]
        
        # Encode
        z_mean, z_log_var = self.encode(x_input)
        
        # Sample
        z = self.reparameterize(z_mean, z_log_var)
        
        # Decode
        reconstruction = self.decode(z, output_length)
        
        # Calculate KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        
        return reconstruction


# ============================================================================
# 4. TRAINING SCRIPT
# ============================================================================
def train_model():
    """
    Main training script.
    """
    # Hyperparameters
    BATCH_SIZE = 32
    LATENT_DIM = 256
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Create data pipeline
    print("Creating data pipeline...")
    train_dataset, valid_dataset, test_dataset = create_data_pipeline(
        batch_size=BATCH_SIZE,
        use_augmentation=True
    )
    
    # Create model
    print("Creating model...")
    model = ProteinVAE(latent_dim=LATENT_DIM, vocab_size=21)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Calculate steps per epoch
    # Note: This is approximate and can be slow for large datasets
    # Consider hardcoding this value after first run
    print("Calculating dataset sizes...")
    data_manager = TFDataManager(tfrecord_dir="data/string/tfrecords", batch_size=BATCH_SIZE)
    
    # For faster startup, you can hardcode these values after first run
    # train_size = 1000000  # example
    # valid_size = 100000   # example
    
    steps_per_epoch = 1000  # Adjust based on your dataset size
    validation_steps = 100  # Adjust based on your dataset size
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/vae_best.keras',
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.TensorBoard(log_dir='logs')
    ]
    
    # Train
    print("Starting training...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_dataset, steps=100)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    
    return model, history


# ============================================================================
# 5. TESTING AND VISUALIZATION
# ============================================================================
def test_data_pipeline():
    """
    Test the data pipeline to ensure everything works correctly.
    """
    print("Testing data pipeline...")
    
    # Create data manager
    data_manager = TFDataManager(
        tfrecord_dir="data/string/tfrecords",
        batch_size=4,
        shuffle_buffer=100,
    )
    
    # Create augmented dataset
    augment_fn = create_augmentation_fn()
    dataset = data_manager.create_dataset(
        split='train',
        shuffle=True,
        repeat=False,
        augment=True,
        augment_fn=augment_fn
    )
    
    # Test a few batches
    for batch_idx, (inputs, outputs) in enumerate(dataset.take(3)):
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Output shape: {outputs.shape}")
        
        # Check for masked tokens
        input_indices = tf.argmax(inputs, axis=-1)
        output_indices = tf.argmax(outputs, axis=-1)
        
        print(f"  Input sequence length: {inputs.shape[1]}")
        print(f"  Output sequence length: {outputs.shape[1]}")
        print(f"  Contains masked tokens: {tf.reduce_any(input_indices == 20).numpy()}")
        
        # Show first sequence
        first_input = input_indices[0].numpy()
        first_output = output_indices[0].numpy()
        print(f"  First input (indices): {first_input[:20]}...")
        print(f"  First output (indices): {first_output[:20]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "prepare":
            print("Preparing TFRecord files...")
            prepare_data()
        elif sys.argv[1] == "test":
            print("Testing data pipeline...")
            test_data_pipeline()
        elif sys.argv[1] == "train":
            print("Training model...")
            train_model()
        else:
            print("Unknown command. Use: prepare, test, or train")
    else:
        print("Usage:")
        print("  python train_example.py prepare  # Prepare TFRecord files")
        print("  python train_example.py test     # Test data pipeline")
        print("  python train_example.py train    # Train model")
