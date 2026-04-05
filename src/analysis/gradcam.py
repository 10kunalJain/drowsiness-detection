"""
Grad-CAM visualization for model interpretability.

Supports both:
- Custom CNN: direct layer access (simple, no nested models)
- MobileNetV2: nested Functional model with two-output backbone trick

Keras 3 compatible.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def _find_target_conv_layer(model: tf.keras.Model, layer_name: str = None):
    """Find the target convolutional layer for Grad-CAM."""
    if layer_name:
        # Try direct lookup first (works for custom CNN)
        try:
            return model.get_layer(layer_name)
        except ValueError:
            pass

        # Try looking inside a nested backbone (for MobileNetV2)
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    return layer.get_layer(layer_name)
                except ValueError:
                    continue

    # Fallback: find last Conv2D in the model
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    # Check inside nested models
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer

    return None


def _has_nested_backbone(model: tf.keras.Model) -> bool:
    """Check if the model has a nested pretrained backbone."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and (
            layer.name.startswith("mobilenet") or layer.name.startswith("resnet")
        ):
            return True
    return False


def compute_gradcam(model: tf.keras.Model, image: np.ndarray,
                    layer_name: str = None) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single image."""
    if layer_name is None:
        layer_name = config.GRADCAM_LAYER

    target_layer = _find_target_conv_layer(model, layer_name)
    if target_layer is None:
        return np.ones((config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE)) * 0.5

    if _has_nested_backbone(model):
        return _gradcam_nested(model, image, target_layer)
    else:
        return _gradcam_flat(model, image, target_layer)


def _gradcam_flat(model: tf.keras.Model, image: np.ndarray,
                  target_layer) -> np.ndarray:
    """Grad-CAM for flat models (custom CNN) — straightforward."""
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output],
    )

    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), 0)

    with tf.GradientTape() as tape:
        conv_features, prediction = grad_model(img_tensor, training=False)
        tape.watch(conv_features)
        loss = prediction[0, 0]

    grads = tape.gradient(loss, conv_features)
    if grads is None:
        return np.ones((config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE)) * 0.5

    return _cam_from_grads(conv_features, grads)


def _gradcam_nested(model: tf.keras.Model, image: np.ndarray,
                    target_layer) -> np.ndarray:
    """Grad-CAM for nested models (MobileNetV2/ResNet) — uses two-output backbone."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and (
            layer.name.startswith("mobilenet") or layer.name.startswith("resnet")
        ):
            backbone = layer
            break

    backbone_dual = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output],
    )

    head_layers = []
    found = False
    for layer in model.layers:
        if layer is backbone:
            found = True
            continue
        if found:
            head_layers.append(layer)

    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), 0)

    with tf.GradientTape() as tape:
        conv_features, backbone_out = backbone_dual(img_tensor, training=False)
        tape.watch(conv_features)
        x = backbone_out
        for hl in head_layers:
            x = hl(x, training=False)
        loss = x[0, 0]

    grads = tape.gradient(loss, conv_features)
    if grads is None:
        return np.ones((config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE)) * 0.5

    return _cam_from_grads(conv_features, grads)


def _cam_from_grads(conv_features, grads) -> np.ndarray:
    """Convert gradients + features into a Grad-CAM heatmap."""
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_features[0] * weights, axis=-1).numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_resized = tf.image.resize(
        cam[..., np.newaxis],
        (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE),
    ).numpy().squeeze()
    return cam_resized


def generate_gradcam_grid(model: tf.keras.Model, images: np.ndarray,
                          labels: np.ndarray, predictions: np.ndarray,
                          num_samples: int = None, save_path: Path = None):
    """Generate Grad-CAM grid for both classes."""
    if num_samples is None:
        num_samples = config.NUM_GRADCAM_SAMPLES
    if save_path is None:
        save_path = config.GRADCAM_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        mask = labels == class_idx
        class_images = images[mask]
        class_preds = predictions[mask]

        n = min(num_samples // 2, len(class_images))
        indices = np.random.RandomState(42).choice(len(class_images), n, replace=False)

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for i, idx in enumerate(indices):
            img = class_images[idx]
            pred = class_preds[idx]
            heatmap = compute_gradcam(model, img)

            axes[i, 0].imshow(img[:, :, 0], cmap="gray")
            axes[i, 0].set_title(f"True: {class_name}\nPred: {pred:.2f}", fontsize=9)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(heatmap, cmap="jet")
            axes[i, 1].set_title("Grad-CAM", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(img[:, :, 0], cmap="gray")
            axes[i, 2].imshow(heatmap, cmap="jet", alpha=0.4)
            axes[i, 2].set_title("Overlay", fontsize=9)
            axes[i, 2].axis("off")

        plt.suptitle(f"Grad-CAM — {class_name} Class", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path / f"gradcam_{class_name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Grad-CAM visualizations saved to {save_path}")


def generate_comparative_gradcam(model: tf.keras.Model, images: np.ndarray,
                                  labels: np.ndarray, predictions: np.ndarray,
                                  save_path: Path = None):
    """Side-by-side: correct vs misclassified."""
    if save_path is None:
        save_path = config.GRADCAM_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    y_pred_class = (predictions > 0.5).astype(int)
    correct_mask = y_pred_class == labels
    incorrect_mask = ~correct_mask

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for row, (mask, title) in enumerate([
        (correct_mask, "Correct Predictions"),
        (incorrect_mask, "Misclassifications"),
    ]):
        subset_images = images[mask]
        subset_labels = labels[mask]
        subset_preds = predictions[mask]

        n = min(5, len(subset_images))
        if n == 0:
            for ax in axes[row]:
                ax.text(0.5, 0.5, "No samples", ha="center", va="center")
                ax.axis("off")
            continue

        indices = np.random.RandomState(42).choice(len(subset_images), n, replace=False)

        for col, idx in enumerate(indices):
            img = subset_images[idx]
            heatmap = compute_gradcam(model, img)
            axes[row, col].imshow(img[:, :, 0], cmap="gray")
            axes[row, col].imshow(heatmap, cmap="jet", alpha=0.4)
            true_label = config.CLASS_NAMES[int(subset_labels[idx])]
            axes[row, col].set_title(
                f"True: {true_label}\nP(drowsy): {subset_preds[idx]:.2f}", fontsize=8,
            )
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(title, fontsize=12, fontweight="bold")

    plt.suptitle("Grad-CAM: Correct vs Misclassified", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path / "gradcam_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparative Grad-CAM saved to {save_path}")
