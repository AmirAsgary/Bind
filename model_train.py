from src import utils, model
from src.model import create_model_vae_plddt, create_model_ae_plddt, recon_loss, SeparateDense, Split
from src.utils import TFRecordManager
from src.constants import angle_columns_291, cos_sine_columns_291, not_angle_columns_291, remove_indices_291, keep_indices_291, angle_columns_281, cos_sine_columns_281, not_angle_columns_281, remove_indices_281,keep_indices_281
import tensorflow as tf
import keras
from keras import layers
import json

tfrecord_path = '/scratch-scc/users/u14286/piplines/Bind/data/large/alphafold_db/tfrecords/repId_sim_to_1.tfrecord'
epochs = 100
angle_columns = tf.constant(list(angle_columns_291.values()), dtype=tf.int32) #used after remove set (D,)
cos_sine_columns = tf.constant(list(cos_sine_columns_291.values()), dtype=tf.int32) #used after remove set (M,2)
not_angle_columns = tf.constant(list(not_angle_columns_291.values()), dtype=tf.int32) #used after remove set (N,)
keep_indices = keep_indices_291
keep_indices_tf = tf.constant(keep_indices, dtype=tf.int32)
# Create models
'''
encoder, decoder, plddt_model, seq_model = create_model_ae_plddt(input_shape=(291,), mask_ratio=0.10, num_layers=1, 
                                                layer_dims=[128], latent_dim=128, 
                                                batchnorm=False, activation='relu')

model = keras.Model(
    inputs=encoder.input,
    outputs=[decoder(encoder.output[-1]), plddt_model(encoder.output[-1]), seq_model(encoder.output[-1])],
    name='vae_plddt_seq_model'
)

model = keras.Model(
    inputs=encoder.input,
    outputs=[decoder(encoder.output), plddt_model(encoder.output), seq_model(encoder.output)],
    name='ae_plddt_seq_model'
)

'''
# Dataset
tfmanager = TFRecordManager(tfrecord_path=tfrecord_path, feature_dim=301, plddt=True, batch_size=2056)
train_ds = tfmanager.read_dataset()

# Count total steps in dataset
total_steps_per_epoch = 1799 #sum(1 for _ in train_ds)



# Training loop
import os

# Prepare directory to save best model weights
checkpoint_dir = './checkpoints/ae4_291dim'
os.makedirs(checkpoint_dir, exist_ok=True)
#


# if model exists, load it

model = keras.models.load_model('checkpoints/ae4_291dim/best_model_epoch.keras')
encoder = keras.models.load_model('checkpoints/ae4_291dim/encoder.keras')
decoder = keras.models.load_model('checkpoints/ae4_291dim/decoder.keras')
plddt_model = keras.models.load_model('checkpoints/ae4_291dim/plddt_model.keras')
seq_model = keras.models.load_model('checkpoints/ae4_291dim/seq_model.keras')
model = keras.Model(
    inputs=encoder.input,
    outputs=[decoder(encoder.output), plddt_model(encoder.output), seq_model(encoder.output)],
    name='ae_plddt_seq_model')

model.compile(optimizer=tf.keras.optimizers.Lion(learning_rate=1e-6, use_ema=True, ema_overwrite_frequency=2))
print(model.summary())  # Print model summary to verify architecture
input()

best_loss = float(2.)
for e in range(epochs):
    train_ds = tfmanager.read_dataset()
    print(f"Epoch {e+1}/{epochs} — Total steps: {total_steps_per_epoch}")
    for step, data in enumerate(train_ds, start=1):
        x, label, _, plddt_true = data
        x = tf.cast(x, tf.float32)
        label = tf.cast(label, tf.int32)
        
        filtered_data = tf.gather(x, keep_indices_tf, axis=1)
        x = tf.cast(filtered_data, tf.float32)
        label_ohe = tf.one_hot(tf.squeeze(label, axis=-1), depth=21, dtype=tf.float32)
        plddt_true = tf.cast(plddt_true, tf.float32)/ 100.0  # Normalize PLDDT to [0, 1]

        with tf.GradientTape() as tape:
            #z_mean, z_log_var, z = encoder(x)
            z = encoder(x)
            y = decoder(z)
            plddt = plddt_model(z)
            label_pred = seq_model(z)
            #y, plddt, label_pred = model(x)
            angle_loss, cos_sin_loss, not_angle_loss = recon_loss(x, y, angle_columns, cos_sine_columns, not_angle_columns,
                                    weight_angle=2.0, weight_cos_sin=3.0, weight_not_angle=1.0,
                                    return_all=True)
            rec_loss = angle_loss * 0.5 + cos_sin_loss * 0.03 + not_angle_loss * 0.3
            plddt_loss = tf.reduce_mean(keras.losses.mean_squared_error(plddt_true, plddt))
            #kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            #kl_loss = tf.reduce_mean(kl_loss)  # scalar
            cce_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(label_ohe, label_pred)
            )

            total_loss = rec_loss + plddt_loss * 1.1 + 1.5 * cce_loss  #+ kl_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"[Epoch {e+1} Step {step}/{total_steps_per_epoch}] "
              f"Reconstruction Loss: {rec_loss.numpy():.4f}, "
              f"PLDDT Loss: {plddt_loss.numpy():.4f}, "
              #f"KL Loss: {kl_loss.numpy():.4f}, ",
                f"CCE Loss: {cce_loss.numpy():.4f}, ",
                f"Angle Loss: {angle_loss.numpy():.4f},"
                f"Cosine-Sine Loss: {cos_sin_loss.numpy():.4f}, "
                f"Not Angle Loss: {not_angle_loss.numpy():.4f}, "
              f"Total Loss: {total_loss.numpy():.4f}")

        # Save model if total_loss is lower than best_loss
        current_loss = total_loss.numpy()
        if current_loss < best_loss:
            best_loss = current_loss
            save_path = os.path.join(checkpoint_dir, f'best_model_epoch.keras')
            model.save(save_path)
            encoder.save(os.path.join(checkpoint_dir, 'encoder.keras'))
            decoder.save(os.path.join(checkpoint_dir, 'decoder.keras'))
            plddt_model.save(os.path.join(checkpoint_dir, 'plddt_model.keras'))
            seq_model.save(os.path.join(checkpoint_dir, 'seq_model.keras'))
            print(f"Saved new best model with total loss {best_loss:.4f} at {save_path}")
