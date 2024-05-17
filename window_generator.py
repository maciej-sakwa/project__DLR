import numpy as np
import tensorflow as tf

class TimeWindowGenerator():

    def __init__(self, df_train, df_val, df_test, input_width, output_width=1, output_offset=1,  column_names = ['temp']):

        # define the raw datasets
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        # Window parameters
        self.input_width = input_width
        self.output_width = output_width
        self.offset = output_offset
        self.total_size = self.input_width + self.offset + self.output_width
        
        # Define the indices
        self.input_indices = np.arange(0, self.input_width)
        self.output_indices = np.arange(self.total_size - self.output_width, self.total_size)

        # Work out the label column indices.
        self.column_names = column_names
        if column_names is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(column_names)}
        self.column_indices = {name: i for i, name in enumerate(self.df_train.columns)}

        # Define the slices
        self.input_slice = slice(0, self.input_width)
        self.output_slice = slice(self.total_size - self.output_width, self.total_size)

        
    # model representation when called
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.output_indices}'])
    
  
    def slice(self, features):

        # Input shape should be (None, window lenght, all-features)
        # Output shape should be (None, prediction_length, label-feature)
        inputs = features[:, self.input_slice, :]
        outputs = features[:, self.output_slice, :]
        if self.column_names is not None:
            outputs = tf.stack([outputs[:, :, self.column_indices[name]] for name in self.column_names], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        outputs.set_shape([None, self.output_width, None])

        return inputs, outputs

    
    def compile_dataset(self, data, opt_shuffle=True):
        # Convert to numpy float32 - for the keras object
        data_array = np.array(data, dtype='float32')
        
        keras_dataset = tf.keras.utils.timeseries_dataset_from_array(
            data = data_array,
            targets=None,
            sequence_length=self.total_size,
            sequence_stride=1,
            shuffle=opt_shuffle,
            batch_size=32
        )

        keras_dataset = keras_dataset.map(self.slice)

        return keras_dataset
    
    @property
    def train(self):
        return self.compile_dataset(self.df_train)

    @property
    def val(self):
        return self.compile_dataset(self.df_val)

    @property
    def test(self):
        return self.compile_dataset(self.df_test, opt_shuffle=False)
    
    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result