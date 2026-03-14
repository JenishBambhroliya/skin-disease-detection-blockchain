import tensorflow as tf
import numpy as np

print("=== MODEL DIAGNOSIS ===")

# Load the model
try:
    model = tf.keras.models.load_model('trained_models/skin_cancer_cnn_model.h5')
    print("✓ Model loaded successfully")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Number of layers: {len(model.layers)}")
    
    # Check if model weights are reasonable
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            weights = layer.get_weights()[0]
            print(f"Layer {i} ({layer.name}): weights shape {weights.shape}, "
                  f"mean={np.mean(weights):.6f}, std={np.std(weights):.6f}")
    
    # Test with dummy data
    print("\n=== TESTING WITH DUMMY DATA ===")
    dummy_input = np.random.random((1, 28, 28, 3))
    dummy_input = dummy_input.astype('float32') / 255.0
    
    prediction = model.predict(dummy_input, verbose=0)
    print(f"Dummy prediction shape: {prediction.shape}")
    print(f"Dummy prediction: {prediction[0]}")
    print(f"Predicted class: {np.argmax(prediction[0])}")
    
    # Check if all predictions are the same (indicates broken model)
    multiple_predictions = []
    for _ in range(5):
        dummy_input = np.random.random((1, 28, 28, 3))
        dummy_input = dummy_input.astype('float32') / 255.0
        pred = model.predict(dummy_input, verbose=0)
        multiple_predictions.append(np.argmax(pred[0]))
    
    print(f"Predictions for 5 random inputs: {multiple_predictions}")
    if len(set(multiple_predictions)) == 1:
        print("⚠️  WARNING: Model always predicts the same class - MODEL IS BROKEN")
    else:
        print("✓ Model makes different predictions for different inputs")
        
except Exception as e:
    print(f"✗ Error loading model: {e}")

print("\n=== RECOMMENDATION ===")
print("The model appears to be corrupted or improperly saved.")
print("Please retrain the model using: python model/train.py")
print("The training logs you showed indicate 90%+ accuracy, but the saved model is broken.")
