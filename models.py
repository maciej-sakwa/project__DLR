import tensorflow as tf


def dense_model(l_units:list, input_shape: tuple, output_sequence: int, name: str, **kwargs):
    
    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(input)
    for un in l_units:
        x = tf.keras.layers.Dense(units=un, **kwargs)(x)
    output = tf.keras.layers.Dense(units=output_sequence, activation='linear')(x)

    return tf.keras.Model(inputs=input, outputs=output, name=name)