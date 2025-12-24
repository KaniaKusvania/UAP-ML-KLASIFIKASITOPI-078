import tensorflow as tf
import json
import h5py

# Function to fix model config for batch_shape
def fix_model_config(config):
    if isinstance(config, dict):
        for key, value in config.items():
            if key == 'config' and isinstance(value, dict) and 'batch_shape' in value:
                batch_shape = value.pop('batch_shape')
                value['shape'] = batch_shape[1:]
                value['batch_size'] = batch_shape[0]
            else:
                fix_model_config(value)
    elif isinstance(config, list):
        for item in config:
            fix_model_config(item)

model_path = 'model/cnn_base_model.h5'
with h5py.File(model_path, 'r') as f:
    model_config = f.attrs['model_config']
    config = json.loads(model_config)
    fix_model_config(config)
    model = tf.keras.models.model_from_config(config)
    # Load weights
    model.load_weights(model_path)
    print('Model loaded successfully')
    print('Model summary:')
    model.summary()