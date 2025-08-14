import tensorflow as tf
import keras
from keras import layers
from keras.saving import register_keras_serializable


@register_keras_serializable(package='custom_layers', name='RandomMasking')
class RandomMasking(layers.Layer):
    def __init__(self, mask_ratio=0.20, name='random_masking', **kwargs):
        super(RandomMasking, self).__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.name = 'random_masking'
        assert 0 <= self.mask_ratio < 1, "mask_ratio must be between 0 and 1"

    def call(self, inputs, training=True):
        if training:
            mask = tf.random.uniform(tf.shape(inputs), minval=0, maxval=1, dtype=tf.float32)
            mask = tf.cast(mask < self.mask_ratio, dtype=tf.float32)
            inputs = inputs * (1 - mask)
        return inputs

    def get_config(self):
        config = super(RandomMasking, self).get_config()
        config.update({"mask_ratio": self.mask_ratio})
        config.update({"name": self.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package='custom_layers', name='Sampling')
class Sampling(layers.Layer):
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        return super(Sampling, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable(package='custom_layers', name='SeparateDense')
class SeparateDense(tf.keras.layers.Layer):
    def __init__(self, columns, units=32, activation='relu', layer_prefix="sep_dense", divide=None, **kwargs):
        super().__init__(**kwargs)
        self.columns = columns
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.layer_prefix = layer_prefix
        self.divide = divide

    def build(self, input_shape):
        n = len(self.columns)
        self.w = self.add_weight(
            shape=(n, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name=f'{self.layer_prefix}_weight')
        
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name=f'{self.layer_prefix}_bias')

    def call(self, x):
        x_splitted = tf.gather(x, self.columns, axis=-1)
        if self.divide:
            x_splitted / tf.cast(self.divide, tf.float32)
        out = tf.matmul(x_splitted, self.w) + self.b
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "columns": self.columns,
            "units": self.units,
            "activation": self.activation,
            "layer_prefix": self.layer_prefix
        })
        return config


@register_keras_serializable(package='custom_layers', name='Split')
class Split(tf.keras.layers.Layer):
    def __init__(self, start, end, **kwargs):
        """
        start: int, starting index (inclusive)
        end: int, ending index (exclusive)
        """
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def call(self, x):
        # First part: selected range
        part1 = x[..., self.start:self.end]
        # Second part: everything else

        part2 = tf.concat([x[..., :self.start], x[..., self.end:]], axis=-1)
        return part1, part2

    def get_config(self):
        config = super().get_config()
        config.update({
            "start": self.start,
            "end": self.end,

        })
        return config
        



def create_model_vae_plddt(input_shape, mask_ratio=0.0, num_layers=3, 
                            layer_dims=[512, 256, 128], latent_dim=128
                            ):
    # Encoder model
    fetures_inp = layers.Input(shape=input_shape, name='features_input')
    x = RandomMasking(mask_ratio=mask_ratio)(fetures_inp)
    for i in range(num_layers):
        x = layers.Dense(layer_dims[i], activation='relu', name=f'encoder_dense_{i}')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    #Decoder model
    y = layers.Dense(layer_dims[-1], activation='relu', name='decoder_dense_0')(z)
    for i in range(1, num_layers):
        y = layers.Dense(layer_dims[-i-1], activation='relu', name=f'decoder_dense_{i}')(y)
    plddt_out = layers.Dense(32, activation='relu', name='plddt_1')(z)
    plddt_out = layers.Dense(1, name='plddt_output')(plddt_out)
    seq_out = layers.Dense(256, activation='relu', name='seq_1')(z)
    seq_out = layers.Dense(21, activation='softmax', name='seq_output')(seq_out)
    features_out = layers.Dense(input_shape[-1], name='features_output')(y)
    encoder = keras.Model(fetures_inp, [z_mean, z_log_var, z], name='encoder')
    decoder = keras.Model(z, features_out, name='decoder')
    plddt_model = keras.Model(z, plddt_out, name='plddt_model')
    seq_model = keras.Model(z, seq_out, name='seq_model')
    return encoder, decoder, plddt_model, seq_model



def create_model_ae_plddt(input_shape, mask_ratio=0.20, num_layers=3, 
                            layer_dims=[128, 64, 64], latent_dim=64, 
                            batchnorm=True, activation='silu'):
    assert num_layers == len(layer_dims)
    # Encoder model
    fetures_inp = layers.Input(shape=input_shape, name='encoder_inp')
    x = layers.GaussianDropout(rate=0.2, seed=42)(fetures_inp)
    x = RandomMasking(mask_ratio=mask_ratio)(fetures_inp)
    if batchnorm:
        x = layers.BatchNormalization(name=f'encoder_bn_input')(x)
    for i in range(num_layers):
        x = layers.Dense(layer_dims[i], activation=activation, name=f'encoder_dense_{i}')(x)
    z = layers.Dense(latent_dim, activation=None, name='encoder_latent')(x)
    #Decoder model
    y = layers.Dense(layer_dims[-1], activation=activation, name='decoder_dense_0')(z)
    y = layers.Dropout(0.1)(y)
    if num_layers != 1:
        for i in range(1, num_layers):
            y = layers.Dense(layer_dims[-i-1], activation=activation, name=f'decoder_dense_{i}')(y)
    features_out = layers.Dense(input_shape[-1], name='decoder_out')(y)

    plddt_out = layers.Dense(32, activation=activation, name='plddt_dense')(z)
    plddt_out = layers.Dropout(0.2)(plddt_out)
    plddt_out = layers.Dense(1, activation='sigmoid', name='plddt_output')(plddt_out)
    seq_out = layers.Dense(128, activation=activation, name='seq_dense')(z)
    seq_out = layers.Dropout(0.2)(seq_out)
    seq_out = layers.Dense(21, activation='softmax', name='seq_output')(seq_out)
    
    encoder = keras.Model(fetures_inp, z, name='encoder')
    decoder = keras.Model(z, features_out, name='decoder')
    plddt_model = keras.Model(z, plddt_out, name='plddt_model')
    seq_model = keras.Model(z, seq_out, name='seq_model')
    return encoder, decoder, plddt_model, seq_model



def recon_loss(y_true, y_pred, angle_columns, cos_sine_columns, not_angle_columns,
             weight_angle=2.0, weight_cos_sin=3.0, weight_not_angle=1.0,
             return_all=False):
    angle_columns = tf.constant(angle_columns, dtype=tf.int32)  
    cos_sine_columns = tf.constant(cos_sine_columns, dtype=tf.int32)
    not_angle_columns = tf.constant(not_angle_columns, dtype=tf.int32)
    # mse for angle columns
    angle_pred = tf.gather(y_pred, angle_columns, axis=1) #(N, D_angle)
    angle_true = tf.gather(y_true, angle_columns, axis=1) #(N, D_angle)
    #condition = (angle_pred > 1.) | (angle_pred < -1.)
    #angle_outrange_penalty = tf.where(condition, tf.minimum(tf.abs(2. * angle_pred), 5.0), 1.)
    angle_loss = tf.abs(angle_true - tf.nn.tanh(angle_pred)) #* angle_outrange_penalty
    angle_loss = tf.square(angle_loss)
    #angle_loss = tf.where(angle_loss <= 0.0025, angle_loss / 2., angle_loss)
    angle_loss = tf.reduce_mean(angle_loss, axis=-1)  # (N,)
    # mse for cos_sin columns
    cos_sin_pred = tf.gather(y_pred, cos_sine_columns, axis=1)  # (N, M, 2)
    cos_sin_pred = tf.nn.tanh(cos_sin_pred)
    norms = tf.reduce_sum(tf.square(cos_sin_pred), axis=-1)  # (N, M)
    cos_sin_loss = tf.reduce_mean(tf.square(1.0 - norms), axis=1)  # (N,)
    # mse for non angle columns
    not_angle_pred = tf.gather(y_pred, not_angle_columns, axis=1)
    not_angle_true = tf.gather(y_true, not_angle_columns, axis=1)
    not_angle_loss = tf.reduce_mean(tf.square(not_angle_true - not_angle_pred), axis=-1)  # (N,)

    if return_all:
        return tf.reduce_mean(angle_loss), tf.reduce_mean(cos_sin_loss), tf.reduce_mean(not_angle_loss)
    else:
        total_loss = (weight_angle * angle_loss + 
                    weight_cos_sin * cos_sin_loss + 
                    weight_not_angle * not_angle_loss)
        total_loss = tf.reduce_mean(total_loss)  # scalar
        return total_loss


