import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DATASET_DIR = 'dataset'
MODEL_PATH = 'trained_models/skin_cancer_cnn_model.h5'
OUTPUT_DIR = 'model/eval_outputs'
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_model_accuracy_correct():
    print("Loading CSV dataset (hmnist_28_28_RGB.csv)...")
    csv_path = os.path.join(DATASET_DIR, 'hmnist_28_28_RGB.csv')
    if not os.path.exists(csv_path):
        print(f"Error: CSV dataset not found at {csv_path}")
        return

    # Load CSV data
    df = pd.read_csv(csv_path)
    print(f"CSV dataset shape: {df.shape}")
    
    # The last column is typically the label
    # First columns are pixel values (28*28*3 = 2352 pixels for RGB)
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values   # Last column is label
    
    # Reshape to image format (28, 28, 3)
    X = X.reshape(-1, 28, 28, 3)
    print(f"Reshaped X to: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Normalize pixel values to 0-1 range (same as training)
    X = X.astype('float32') / 255.0
    
    # Convert labels to categorical (7 classes: 0-6)
    y_categorical = tf.keras.utils.to_categorical(y, num_classes=7)
    
    # Split data (80% train, 20% test) - same as training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Class distribution in test set: {np.sum(y_test, axis=0)}")
    
    print("Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Compile model for evaluation
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Running evaluation...")
    results = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    # Get predictions for detailed metrics
    print("Generating detailed metrics...")
    Y_pred = model.predict(X_test, verbose=0)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy manually
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Manual Accuracy Calculation: {accuracy:.4f}")
    
    # Class labels mapping
    class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    print(f"Class labels: {class_labels}")
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    with open(os.path.join(OUTPUT_DIR, 'correct_accuracy_metrics.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Print detailed report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(OUTPUT_DIR, 'correct_accuracy_confusion_matrix.png'))
    plt.close()
    
    # Save summary results
    summary = {
        'total_test_samples': len(y_true),
        'test_loss': float(results[0]),
        'test_accuracy': float(results[1]),
        'manual_accuracy': float(accuracy),
        'class_distribution': {class_labels[i]: int(np.sum(y_true == i)) for i in range(len(class_labels))}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'correct_accuracy_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nEvaluation finished. Results saved in {OUTPUT_DIR}")
    print(f"Summary: {summary['test_accuracy']:.1%} accuracy on {summary['total_test_samples']} test samples")
    
    return summary

if __name__ == "__main__":
    test_model_accuracy_correct()
