"""
Utility Functions Module
Grad-CAM, visualization, and helper functions
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def generate_grad_cam(model, img, layer_name=None):
    """
    Generate Grad-CAM heatmap for interpretability.
    Fixed version for binary classification.
    
    Args:
        model: Trained Keras model
        img: Preprocessed image (normalized, shape: (H, W, 3))
        layer_name: Name of target layer (auto-detected if None)
    
    Returns:
        heatmap: Grad-CAM heatmap (values 0-1)
    """
    try:
        # Find last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            raise ValueError("No convolutional layer found in model")
        
        # Create a model that outputs both predictions and conv layer output
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Prepare input
        img_tensor = tf.expand_dims(img, 0)
        
        # Compute gradient
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            # For binary classification, we want the predicted class score
            loss = predictions[0][0]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Compute weighted combination
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Average over all filters
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        return heatmap
        
    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        # Return empty heatmap on error
        return np.zeros((7, 7))


def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix with better styling.
    
    Args:
        cm: Confusion matrix array
        classes: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_pred_prob, save_path='roc_curve.png'):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.show()


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Keras training history object
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.show()


def save_predictions_sample(model, test_gen, num_samples=16, save_path='predictions_sample.png'):
    """
    Visualize model predictions on test samples.
    
    Args:
        model: Trained model
        test_gen: Test data generator
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    # Get a batch of test images
    test_images, test_labels = next(test_gen)
    test_images = test_images[:num_samples]
    test_labels = test_labels[:num_samples]
    
    # Make predictions
    predictions = model.predict(test_images)
    
    # Plot
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    class_names = ['Normal', 'Pothole']
    
    for i in range(num_samples):
        axes[i].imshow(test_images[i])
        
        true_label = class_names[int(test_labels[i])]
        pred_label = class_names[int(predictions[i] > 0.5)]
        confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})",
                         color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Predictions sample saved to {save_path}")
    plt.show()