"""
Drowsiness Detection Models.

Two options:
1. CustomCNN: Purpose-built for small grayscale eye images (64x64x1)
   - 4 conv blocks with BatchNorm + progressive channel widening
   - Much better suited than ImageNet-pretrained models for this domain
   - Faster inference, smaller model, trains from scratch effectively

2. MobileNetV2: Transfer learning baseline for comparison
   - Requires 3-channel input and larger resolution
   - Good when data is more diverse, but over-parameterized for 48px eyes
"""
import tensorflow as tf
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


def build_model(model_type: str = None, input_shape: tuple = None) -> tf.keras.Model:
    """Build the drowsiness detection model."""
    if model_type is None:
        model_type = config.MODEL_TYPE

    if model_type == "custom_cnn":
        return _build_custom_cnn(input_shape)
    elif model_type == "mobilenetv2":
        return _build_mobilenetv2(input_shape)
    elif model_type == "resnet50v2":
        return _build_resnet(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _build_custom_cnn(input_shape: tuple = None) -> tf.keras.Model:
    """
    Custom CNN designed specifically for small grayscale eye images.

    Architecture: 4 conv blocks (32→64→128→256 channels) with:
    - BatchNorm after every conv for stable training
    - MaxPool to progressively reduce spatial dims
    - Dropout between blocks for regularization
    - Global Average Pooling (not Flatten) to reduce overfitting
    """
    if input_shape is None:
        input_shape = (config.MODEL_INPUT_SIZE, config.MODEL_INPUT_SIZE, config.NUM_CHANNELS)

    inputs = tf.keras.Input(shape=input_shape, name="eye_input")

    # Block 1: 64x64 → 32x32
    x = tf.keras.layers.Conv2D(32, 3, padding="same", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", name="conv1b")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1b")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 2: 32x32 → 16x16
    x = tf.keras.layers.Conv2D(64, 3, padding="same", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", name="conv2b")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2b")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Block 3: 16x16 → 8x8
    x = tf.keras.layers.Conv2D(128, 3, padding="same", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", name="conv3b")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3b")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Block 4: 8x8 → 4x4
    x = tf.keras.layers.Conv2D(256, 3, padding="same", name="conv4")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="fc_1")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_head")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drowsy_prob")(x)

    return tf.keras.Model(inputs, outputs, name="DrowsinessDetector_CustomCNN")


def _build_mobilenetv2(input_shape: tuple = None) -> tf.keras.Model:
    """MobileNetV2 transfer learning model — requires 3-channel, min 96x96."""
    if input_shape is None:
        size = max(config.MODEL_INPUT_SIZE, 96)
        input_shape = (size, size, 3)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="eye_input")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="fc_1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drowsy_prob")(x)

    return tf.keras.Model(inputs, outputs, name="DrowsinessDetector_MobileNetV2")


def _build_resnet(input_shape: tuple = None) -> tf.keras.Model:
    """ResNet50V2 transfer learning model."""
    if input_shape is None:
        size = max(config.MODEL_INPUT_SIZE, 96)
        input_shape = (size, size, 3)

    backbone = tf.keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="eye_input")
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="fc_1")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="drowsy_prob")(x)

    return tf.keras.Model(inputs, outputs, name="DrowsinessDetector_ResNet50V2")


def compile_model(model: tf.keras.Model, learning_rate: float = None) -> tf.keras.Model:
    """Compile with label smoothing and appropriate learning rate."""
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(
            label_smoothing=config.LABEL_SMOOTHING,
        ),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def unfreeze_backbone(model: tf.keras.Model, unfreeze_from: int = 100) -> tf.keras.Model:
    """Unfreeze top layers of pretrained backbone for fine-tuning."""
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and (
            layer.name.startswith("mobilenet") or layer.name.startswith("resnet")
        ):
            backbone = layer
            break

    if backbone is None:
        print("  No pretrained backbone found — skipping unfreeze (custom CNN trains fully)")
        return model

    backbone.trainable = True
    for layer in backbone.layers[:unfreeze_from]:
        layer.trainable = False

    trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"  Fine-tuning: {trainable} backbone layers unfrozen (from layer {unfreeze_from})")
    return model


def save_model(model: tf.keras.Model, name: str = "drowsiness_detector"):
    """Save model in both .keras and TFLite formats."""
    save_path = config.MODEL_DIR / f"{name}.keras"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"  Saved Keras model: {save_path}")

    # Export TFLite for edge/mobile deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = config.MODEL_DIR / f"{name}.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"  Saved TFLite model: {tflite_path} ({len(tflite_model)/1024:.0f} KB)")

    return save_path, tflite_path
