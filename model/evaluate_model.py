import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants - matching train.py logic
DATASET_DIR = 'dataset'
MODEL_PATH = 'trained_models/skin_cancer_cnn_model.h5'
OUTPUT_DIR = 'model/eval_outputs'
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate():
    print("Loading datasets...")
    metadata_path = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata not found at {metadata_path}")
        return

    df = pd.read_csv(metadata_path)
    
    def get_path(image_id):
        p1 = os.path.join(DATASET_DIR, 'HAM10000_images_part_1', f'{image_id}.jpg')
        p2 = os.path.join(DATASET_DIR, 'HAM10000_images_part_2', f'{image_id}.jpg')
        if os.path.exists(p1): return p1
        return p2
        
    df['path'] = df['image_id'].apply(get_path)
    # Shuffle exactly like train.py set random_state=42
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total mapped: {len(df)}")
    
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2
    )
    
    validation_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        y_col='dx',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Running evaluation...")
    results = model.evaluate(validation_generator)
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f}")
    
    # Detailed metrics
    print("Generating detailed metrics...")
    validation_generator.reset()
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    
    report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    with open(os.path.join(OUTPUT_DIR, 'latest_metrics.json'), 'w') as f:
        json.dump(report, f, indent=4)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(OUTPUT_DIR, 'latest_cm.png'))
    plt.close()
    
    print(f"Evaluation finished. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    evaluate()
