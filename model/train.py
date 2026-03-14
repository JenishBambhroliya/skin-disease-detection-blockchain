import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

# Constants
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'trained_models/skin_cancer_cnn_model.h5'
OUTPUT_DIR = 'eval_outputs'
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 7

# Ensure output directories exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_model():
    print("Loading datasets with Pandas...")
    metadata_path = os.path.join(DATASET_DIR, 'HAM10000_metadata.csv')
    df = pd.read_csv(metadata_path)
    
    # Create absolute paths
    def get_path(image_id):
        p1 = os.path.join(DATASET_DIR, 'HAM10000_images_part_1', f'{image_id}.jpg')
        p2 = os.path.join(DATASET_DIR, 'HAM10000_images_part_2', f'{image_id}.jpg')
        if os.path.exists(p1): return p1
        return p2
        
    df['path'] = df['image_id'].apply(get_path)
    # Shuffle the dataset to ensure validation_split draws a diverse subset of labels!
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total valid images mapped: {len(df)}")
    
    print("Initializing Data Generators with Augmentation...")
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        y_col='dx',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        y_col='dx',
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Save the class mappings explicitly matching how they were alphabetized!
    class_map_file = 'model/class_labels.json'
    class_indices = train_generator.class_indices
    full_names_map = {
        'akiec': "Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)",
        'bcc': 'basal cell carcinoma (bcc)',
        'bkl': 'benign keratosis-like lesions (bkl)',
        'df': 'dermatofibroma (df)',
        'nv': 'melanocytic nevi (nv)',
        'mel': 'melanoma (mel)',
        'vasc': 'vascular lesions (vasc)'
    }
    
    class_dict = {str(v): full_names_map.get(k, k) for k, v in class_indices.items()}
    with open(class_map_file, 'w') as f:
        json.dump(class_dict, f, indent=4)
    print("Exported class_labels.json mapping:", class_dict)
    
    # Create and compile model
    print("Building Model...")
    model = create_model(NUM_CLASSES)
    
    # Callbacks (Early Stopping to prevent overfitting)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train Model
    print("Starting Training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stop]
    )
    
    # Save the model
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    # --- Evaluation Outputs ---
    generate_evaluations(model, validation_generator, history)

def generate_evaluations(model, validation_generator, history):
    print("Generating Evaluation Outputs...")
    
    # 1. Save Training History Plot
    plt.figure(figsize=(12, 4))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()
    
    # 2. Confusion Matrix & Classification Report
    print("Evaluating Validation Set for Metrics & Confusion Matrix...")
    validation_generator.reset()
    Y_pred = model.predict(validation_generator, steps=int(np.ceil(validation_generator.samples / validation_generator.batch_size)))
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = validation_generator.classes

    class_labels = list(validation_generator.class_indices.keys())
    
    # Save Classification Report as JSON
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(report_dict, f, indent=4)
        
    # Generate and Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    print(f"Training Complete. Artifacts saved in {OUTPUT_DIR}/ and model saved to {MODEL_SAVE_PATH}.")

if __name__ == '__main__':
    train_model()
