"""
Training and Evaluation Module
Enhanced with comprehensive metrics and visualization
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                       ModelCheckpoint, TensorBoard)
from utils import plot_confusion_matrix, plot_roc_curve, plot_training_history
import os
from datetime import datetime

def train_and_evaluate(model, train_gen, val_gen, test_gen, epochs, model_path):
    """
    Train the model with callbacks, save it, evaluate on test set, and plot metrics.
    
    Args:
        model: Compiled Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        test_gen: Test data generator
        epochs: Number of training epochs
        model_path: Path to save the best model
    """
    try:
        # Create directories for logs and checkpoints
        log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]
        
        print("\n=== Starting Training ===")
        print(f"Training samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Batch size: {train_gen.batch_size}")
        print(f"Steps per epoch: {train_gen.samples // train_gen.batch_size}")
        
        # Train model
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\nModel saved to {model_path}")
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate on test set
        print("\n=== Evaluating on Test Set ===")
        test_steps = test_gen.samples // test_gen.batch_size + 1
        
        y_pred_prob = model.predict(test_gen, steps=test_steps, verbose=1)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        y_true = test_gen.classes[:len(y_pred)]
        y_pred_prob = y_pred_prob[:len(y_true)].flatten()
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_pred_prob)
        except ValueError:
            auc = 0.0
            print("Warning: Could not calculate AUC-ROC (only one class in test set)")
        
        print("\n=== Test Set Performance ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        
        # Classification report
        print("\n=== Classification Report ===")
        class_names = ['Normal', 'Pothole']
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=class_names)
        
        # ROC curve
        if auc > 0:
            plot_roc_curve(y_true, y_pred_prob)
        
        # Save metrics to file
        metrics_file = model_path.replace('.h5', '_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Test Set Performance\n")
            f.write(f"{'='*50}\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n")
            f.write(f"AUC-ROC:   {auc:.4f}\n\n")
            f.write(classification_report(y_true, y_pred, target_names=class_names))
        
        print(f"\nMetrics saved to {metrics_file}")
        
        return history
        
    except Exception as e:
        print(f"Error in training/evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise