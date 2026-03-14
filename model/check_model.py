import tensorflow as tf

model = tf.keras.models.load_model('trained_models/skin_cancer_cnn_model.h5')
print(f'Input shape: {model.input_shape}')
print(f'Output shape: {model.output_shape}')
print(f'Layers: {len(model.layers)}')
print('\nLayer details:')
for i, layer in enumerate(model.layers):
    print(f'{i}: {layer.name} - {layer.__class__.__name__}')
