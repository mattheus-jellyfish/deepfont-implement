import os
import argparse
import random
import string
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import glob
import itertools
from imutils import paths
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, UpSampling2D, Input
from tensorflow.keras import callbacks, optimizers, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def pil_image(img_path):
    """Convert image to PIL format and resize."""
    pil_im = Image.open(img_path).convert('L')
    pil_im = pil_im.resize((105, 105))
    return pil_im


def noise_image(pil_im):
    """Add random noise to image."""
    img_array = np.asarray(pil_im)
    mean = 0.0
    std = 5
    noisy_img = img_array + np.random.normal(mean, std, img_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    noise_img = Image.fromarray(np.uint8(noisy_img_clipped))
    noise_img = noise_img.resize((105, 105))
    return noise_img


def blur_image(pil_im):
    """Apply Gaussian blur to image."""
    blur_img = pil_im.filter(ImageFilter.GaussianBlur(radius=3))
    blur_img = blur_img.resize((105, 105))
    return blur_img


def affine_rotation(img):
    """Apply affine transformation to image."""
    if isinstance(img, Image.Image):
        img = np.array(img)
        
    rows, columns = img.shape
    point1 = np.float32([[10, 10], [30, 10], [10, 30]])
    point2 = np.float32([[20, 15], [40, 10], [20, 40]])
    A = cv2.getAffineTransform(point1, point2)
    output = cv2.warpAffine(img, A, (columns, rows))
    affine_img = Image.fromarray(np.uint8(output))
    affine_img = affine_img.resize((105, 105))
    return affine_img


def gradient_fill(image):
    """Apply Laplacian gradient to image."""
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.resize(laplacian, (105, 105))
    return laplacian


def variable_aspect_ratio(pil_img):
    """
    Apply variable aspect ratio by randomly squeezing or stretching the width.
    Following the paper's approach: width is randomly adjusted by a ratio drawn
    uniformly from the range [5/6, 7/6].
    """
    width, height = pil_img.size
    
    # Random aspect ratio factor drawn from [5/6, 7/6] = [0.833, 1.167]
    aspect_ratio_factor = random.uniform(5/6, 7/6)
    
    # Calculate new width maintaining height
    new_width = int(width * aspect_ratio_factor)
    
    # Resize with the new aspect ratio
    stretched_img = pil_img.resize((new_width, height), Image.LANCZOS)
    
    # Ensure final size is 105x105 by center cropping or padding
    if new_width > 105:
        # Center crop if wider than 105
        left = (new_width - 105) // 2
        stretched_img = stretched_img.crop((left, 0, left + 105, 105))
    else:
        # Pad with white if narrower than 105
        result_img = Image.new('L', (105, 105), 255)
        paste_x = (105 - new_width) // 2
        result_img.paste(stretched_img, (paste_x, 0))
        stretched_img = result_img
    
    return stretched_img


def apply_augmentations(pil_img, augment_types=None):
    """
    Apply various augmentations to an input image.
    
    Args:
        pil_img: PIL Image to augment
        augment_types: List of augmentation types to apply (default: ["blur", "noise", "affine", "gradient", "aspect_ratio"])
    
    Returns:
        Dictionary of augmented images with augmentation type as key
    """
    if augment_types is None:
        augment_types = ["blur", "noise", "affine", "gradient", "aspect_ratio"]
    
    augmented_images = {}
    augmented_images['original'] = pil_img
    
    # Individual augmentations
    augmented_images['noise'] = noise_image(pil_img)
    augmented_images['blur'] = blur_image(pil_img)
    
    open_cv_img = np.array(pil_img)
    affine_img = affine_rotation(open_cv_img)
    augmented_images['affine'] = affine_img
    
    gradient_img = gradient_fill(open_cv_img)
    if isinstance(gradient_img, np.ndarray):
        gradient_img = Image.fromarray(np.uint8(np.clip(gradient_img, 0, 255)))
    augmented_images['gradient'] = gradient_img
    
    # New aspect ratio augmentation
    augmented_images['aspect_ratio'] = variable_aspect_ratio(pil_img)
    
    # Combinations
    for l in range(2, len(augment_types) + 1):
        combinations = list(itertools.combinations(augment_types, l))
        
        for combo in combinations:
            combo_name = "+".join(combo)
            temp_img = pil_img
            
            for aug_type in combo:
                if aug_type == 'noise':
                    temp_img = noise_image(temp_img)
                elif aug_type == 'blur':
                    temp_img = blur_image(temp_img)
                elif aug_type == 'affine':
                    open_cv_affine = np.array(temp_img)
                    temp_img = affine_rotation(open_cv_affine)
                elif aug_type == 'gradient':
                    open_cv_gradient = np.array(temp_img)
                    temp_img = gradient_fill(open_cv_gradient)
                elif aug_type == 'aspect_ratio':
                    temp_img = variable_aspect_ratio(temp_img)
                
                # Convert to PIL image if needed
                if isinstance(temp_img, np.ndarray):
                    temp_img = Image.fromarray(np.uint8(np.clip(temp_img, 0, 255)))
            
            augmented_images[combo_name] = temp_img
    
    return augmented_images


def get_random_text(length):
    """Generate random text of specified length."""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def create_image(size, message, font, variable_spacing=False):
    """Create an image with text using the specified font.
    
    Args:
        size: Tuple of (width, height)
        message: Text to render
        font: Font to use
        variable_spacing: Whether to apply variable character spacing
    """
    width, height = size
    image = Image.new('L', size, 'white')
    draw = ImageDraw.Draw(image)
    
    if not variable_spacing:
        # Standard text rendering - center the text
        bbox = font.getbbox(message)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), message, font=font, fill='black')
    else:
        # Apply variable character spacing
        # Start position for the first character
        x_start = 10
        y = height // 2 - font.getbbox("X")[3] // 2
        
        # Render each character with random spacing
        for char in message:
            # Draw the character
            draw.text((x_start, y), char, font=font, fill='black')
            
            # Get width of current character
            char_width = font.getbbox(char)[2]
            
            # Apply random spacing using Gaussian distribution 
            # as per paper (mean=10, std=40, bounded between 0 and 50)
            spacing = min(max(0, int(np.random.normal(10, 40))), 100)
            
            # Move to next character position
            x_start += char_width + spacing
            
            # If we've run out of space, stop rendering
            if x_start > width - 10:
                break
    
    return image


def create_dataset(brand_font_path="fonts/Denton-Light.otf", other_fonts_dir=None, output_dir="train_data", count_per_font=500):
    """
    Create a dataset of images with brand font (positive class) and other fonts (negative class).
    
    Args:
        brand_font_path: Path to the brand font file
        other_fonts_dir: Directory containing other font files (optional)
        output_dir: Directory to save generated images
        count_per_font: Number of images to generate per font
    """
    # Check if the brand font file exists
    if not os.path.exists(brand_font_path):
        # Try looking for it in the fonts directory
        alt_path = os.path.join("fonts", os.path.basename(brand_font_path))
        if os.path.exists(alt_path):
            print(f"Font not found at {brand_font_path}, using {alt_path} instead")
            brand_font_path = alt_path
        else:
            raise FileNotFoundError(f"Brand font file not found at {brand_font_path} or {alt_path}. Please provide a valid font file path.")
    
    # Create output directories
    Path(f"{output_dir}/positive").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/negative").mkdir(parents=True, exist_ok=True)
    
    # Generate positive samples with brand font
    brand_font_name = Path(brand_font_path).stem
    print(f"Generating {count_per_font} positive samples with font: {brand_font_name}")
    
    try:
        for c in range(count_per_font):
            height = 80
            width = 600
            fontsize = 48
            font = ImageFont.truetype(brand_font_path, fontsize)
            msg = get_random_text(random.randint(6, 18))
            
            # Apply variable character spacing to approximately half of the samples
            # use_variable_spacing = random.choice([True, False])
            use_variable_spacing = False
            pil_img = create_image((width, height), msg, font, variable_spacing=use_variable_spacing)
            
            Path(f"{output_dir}/positive/{brand_font_name}").mkdir(parents=True, exist_ok=True)
            pil_img.save(f"{output_dir}/positive/{brand_font_name}/{msg}.jpg")
    except Exception as e:
        print(f"Error creating positive samples: {e}")
        raise
    
    # Generate negative samples with other fonts if directory is provided
    if other_fonts_dir:
        # Check if other_fonts_dir exists
        if not os.path.exists(other_fonts_dir):
            print(f"Warning: Other fonts directory {other_fonts_dir} not found")
            return
            
        # List font files excluding the brand font
        other_font_files = []
        for ext in ['*.ttf', '*.otf', '*.ttc']:
            other_font_files.extend(glob.glob(os.path.join(other_fonts_dir, ext)))
        
        # Filter out the brand font
        other_font_files = [f for f in other_font_files 
                          if Path(f).stem != brand_font_name]
        
        if not other_font_files:
            print(f"Warning: No other fonts found in {other_fonts_dir}")
            return
        
        samples_per_font = count_per_font // len(other_font_files)
        if samples_per_font < 1:
            samples_per_font = 1
        
        for font_file in other_font_files:
            font_name = Path(font_file).stem
            print(f"Generating ~{samples_per_font} negative samples with font: {font_name}")
            
            # Create font-specific subfolder under negative
            Path(f"{output_dir}/negative/{font_name}").mkdir(parents=True, exist_ok=True)
            
            try:
                for c in range(samples_per_font):
                    height = 80
                    width = 600
                    fontsize = 48
                    try:
                        font = ImageFont.truetype(font_file, fontsize)
                        msg = get_random_text(random.randint(6, 18))
                        
                        # Apply variable character spacing to approximately half of the samples
                        # use_variable_spacing = random.choice([True, False])
                        use_variable_spacing = False
                        pil_img = create_image((width, height), msg, font, variable_spacing=use_variable_spacing)
                        
                        # Save in font-specific subfolder
                        pil_img.save(f"{output_dir}/negative/{font_name}/{msg}.jpg")
                    except Exception as e:
                        print(f"Warning: Could not use font {font_name}: {e}")
                        continue
            except Exception as e:
                print(f"Error processing font {font_name}: {e}")
                continue


def create_autoencoder():
    """
    Create a stacked convolutional autoencoder for domain adaptation.
    Following Figure 6 in the DeepFont paper.
    
    Returns:
        Autoencoder model
    """
    # Create input layer
    inputs = Input(shape=(105, 105, 1))
    
    # Encoder (Cu layers)
    # First convolutional layer
    x = Conv2D(64, kernel_size=(11, 11), strides=2, activation='relu', padding='valid')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second convolutional layer - smaller kernel as per paper
    x = Conv2D(128, kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x)
    
    # Decoder
    x = Conv2DTranspose(64, kernel_size=(1, 1), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D(size=(2, 2))(x)
    outputs = Conv2DTranspose(1, kernel_size=(11, 11), strides=2, activation='sigmoid', padding='valid')(x)
    
    # Create model
    autoencoder = Model(inputs=inputs, outputs=outputs)
    
    return autoencoder


def create_model(use_domain_adaptation=True, pretrained_encoder=None):
    """
    Create a binary classification model for font detection with optional domain adaptation.
    Following the 8-layer network architecture in Figure 5 of the DeepFont paper,
    adapted for binary classification.
    
    Args:
        use_domain_adaptation: Whether to structure the model to support domain adaptation
        pretrained_encoder: Optional pretrained encoder weights for domain adaptation
    
    Returns:
        Model for binary classification
    """
    # Create input layer
    inputs = Input(shape=(105, 105, 1))
    
    # Cu layers - Feature extraction layers (first 2 layers in DeepFont paper)
    # Conv1 layer
    x = Conv2D(64, kernel_size=(11, 11), strides=2, activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Conv2 layer
    x = Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Cs layers - Classification specific layers (next 6 layers in DeepFont paper)
    # Conv3-5 layers (3 convolutional layers with same parameters)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    
    # Fully connected layers
    x = Flatten()(x)
    
    # fc6
    # x = Dense(4096, activation='relu')(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # fc7
    # x = Dense(4096, activation='relu')(x)
    # x = Dense(1024, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # fc8 (output layer) - adapted for binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # If using domain adaptation, transfer weights from pretrained encoder
    if use_domain_adaptation and pretrained_encoder is not None:
        # Get the encoder part of the autoencoder
        encoder_layers = [layer for layer in pretrained_encoder.layers if isinstance(layer, Conv2D)]
        
        # Transfer weights to the corresponding layers
        model_conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
        
        # Transfer weights to the first two Conv layers
        model_conv_layers[0].set_weights(encoder_layers[0].get_weights())
        model_conv_layers[1].set_weights(encoder_layers[1].get_weights())
        
        # Freeze the transferred layers during initial training
        model_conv_layers[0].trainable = False
        model_conv_layers[1].trainable = False
    
    return model


def precision_m(y_true, y_pred):
    """
    Custom precision metric for use during training.
    For binary classification where positive class is 1.
    """
    # Ensure both tensors are float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    y_pred_binary = K.round(y_pred)  # Round predictions to 0 or 1
    true_positives = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)))
    predicted_positives = K.sum(y_pred_binary)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_m(y_true, y_pred):
    """
    Custom recall metric for use during training.
    For binary classification where positive class is 1.
    """
    # Ensure both tensors are float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    y_pred_binary = K.round(y_pred)  # Round predictions to 0 or 1
    true_positives = K.sum(K.round(K.clip(y_true * y_pred_binary, 0, 1)))
    actual_positives = K.sum(y_true)
    recall = true_positives / (actual_positives + K.epsilon())
    return recall


def f1_m(y_true, y_pred):
    """
    Custom F1 score metric for use during training.
    Based on the precision and recall metrics.
    """
    # Ensure both tensors are float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def train_model(data_path, epochs=20, batch_size=32, use_domain_adaptation=True, final_model_path="denton_font_model_final.keras"):
    """
    Train the font detection model.
    
    Args:
        data_path: Path to the dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training (paper uses 128)
        use_domain_adaptation: Whether to use domain adaptation with SCAE
        final_model_path: Path to save the final model
        
    Returns:
        Trained model
    """
    data = []
    labels = []
    
    # Load positive examples (brand font)
    positive_paths = sorted(list(paths.list_images(os.path.join(data_path, "positive"))))
    print(f"Found {len(positive_paths)} positive examples")
    
    # Load negative examples (other fonts)
    negative_paths = sorted(list(paths.list_images(os.path.join(data_path, "negative"))))
    print(f"Found {len(negative_paths)} negative examples")
    
    # Calculate class weights based on dataset distribution
    n_samples = len(positive_paths) + len(negative_paths)
    n_positive = len(positive_paths)
    n_negative = len(negative_paths)
    
    # Class weight calculation options
    # Option 1: Balanced - inversely proportional to class frequencies
    weight_for_0 = n_samples / (2.0 * n_negative)
    weight_for_1 = n_samples / (2.0 * n_positive)
    
    # Option 2: Manually scaled weight approach (more emphasis on positive)
    # weight_for_0 = 1.0
    # weight_for_1 = n_negative / n_positive * 1.5  # Additional multiplier for positive class
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Using class weights: {class_weight}")
    
    # Combine and shuffle
    image_paths = positive_paths + negative_paths
    random.seed(42)
    random.shuffle(image_paths)
    
    # Apply augmentations to increase dataset size
    augment_types = ["blur", "noise", "affine", "gradient", "aspect_ratio"]
    
    for image_path in image_paths:
        # Determine label from directory
        label = 1 if "positive" in image_path else 0
        
        # Load and process original image
        pil_img = pil_image(image_path)
        
        # Apply augmentations
        augmented_imgs = apply_augmentations(pil_img, augment_types)
        
        # Add all augmented images to the dataset
        for aug_img in augmented_imgs.values():
            img_array = img_to_array(aug_img)
            data.append(img_array)
            labels.append(label)
    
    # Convert to numpy arrays and normalize
    data = np.asarray(data, dtype="float") / 255.0
    labels = np.array(labels)
    
    # Print distribution after augmentation
    n_pos_after_aug = np.sum(labels == 1)
    n_neg_after_aug = np.sum(labels == 0)
    print(f"After augmentation: {n_pos_after_aug} positive samples, {n_neg_after_aug} negative samples")
    
    # Recalculate class weights if desired after augmentation
    # Uncomment the following lines to adjust weights based on augmented distribution
    # weight_for_0 = len(labels) / (2.0 * n_neg_after_aug)
    # weight_for_1 = len(labels) / (2.0 * n_pos_after_aug)
    # class_weight = {0: weight_for_0, 1: weight_for_1}
    # print(f"Adjusted class weights after augmentation: {class_weight}")
    
    # Split data
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    
    # Count distribution in train/test sets
    train_pos = np.sum(trainY == 1)
    train_neg = np.sum(trainY == 0)
    test_pos = np.sum(testY == 1)
    test_neg = np.sum(testY == 0)
    print(f"Training set: {train_pos} positive, {train_neg} negative")
    print(f"Test set: {test_pos} positive, {test_neg} negative")
    
    # Domain adaptation with autoencoder if enabled
    pretrained_encoder = None
    if use_domain_adaptation:
        print("Performing domain adaptation with autoencoder...")
        autoencoder = create_autoencoder()
        
        # Configure SGD optimizer as specified in the paper
        sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=0.0005)
        autoencoder.compile(optimizer=sgd, loss='mse')
        
        # Train autoencoder on all data (unsupervised)
        autoencoder.fit(
            data, data,
            epochs=10,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            verbose=1
        )
        
        pretrained_encoder = autoencoder
    
    # Create and compile model
    model = create_model(use_domain_adaptation, pretrained_encoder)
    
    # Configure optimizer as specified in the paper
    if use_domain_adaptation:
        # Lower learning rate for fine-tuning if we used pretrained weights
        opt = optimizers.SGD(learning_rate=0.001, momentum=0.9, weight_decay=0.0005)
    else:
        # Full learning rate for training from scratch
        opt = optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=0.0005)
    
    model.compile(
        loss='binary_crossentropy', 
        optimizer=opt, 
        metrics=['accuracy', precision_m, recall_m, f1_m]
    )
    
    # Learning rate scheduler - divide by 10 when validation error plateaus
    def lr_scheduler(epoch, lr):
        if epoch > 0 and epoch % 5 == 0:  # Check every 5 epochs
            return lr * 0.1 if epoch > 10 else lr
        return lr
    
    # Set up callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=10, 
        verbose=1, 
        mode='min'
    )
    
    lr_callback = callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    
    # Remove ModelCheckpoint callback
    callbacks_list = [early_stopping, lr_callback]
    
    # Train model
    print("Starting model training...")
    history = model.fit(
        trainX, 
        trainY,
        validation_data=(testX, testY),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list,
        class_weight=class_weight  # Pass class weights to fit
    )
    
    # Evaluate model
    scores = model.evaluate(testX, testY, verbose=0)
    print(f"Test loss: {scores[0]}")
    print(f"Test accuracy: {scores[1]}")
    print(f"Test precision: {scores[2]}")
    print(f"Test recall: {scores[3]}")
    print(f"Test F1 score: {scores[4]}")
    
    # Get model name for plots
    model_name = os.path.splitext(os.path.basename(final_model_path))[0]
    history_plot_path = f"{model_name}_training_history.png"
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Accuracy subplot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss subplot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # F1 Score subplot
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_m'])
    plt.plot(history.history['val_f1_m'])
    plt.title('Model F1 Score')
    plt.ylabel('F1 Score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(history_plot_path)
    plt.close()
    print(f"Training history saved to {history_plot_path}")
    
    # If we used domain adaptation, unfreeze the encoder layers and fine-tune
    if use_domain_adaptation and pretrained_encoder is not None:
        print("Fine-tuning the entire model...")
        # Unfreeze all layers
        for layer in model.layers:
            layer.trainable = True
            
        # Recompile with a lower learning rate
        opt = optimizers.SGD(learning_rate=0.001, momentum=0.9, weight_decay=0.0005)
        model.compile(
            loss='binary_crossentropy', 
            optimizer=opt, 
            metrics=['accuracy', precision_m, recall_m, f1_m]
        )
        
        # Fine-tune
        ft_history = model.fit(
            trainX, 
            trainY,
            validation_data=(testX, testY),
            batch_size=batch_size,
            epochs=5,  # Just a few epochs for fine-tuning
            verbose=1,
            class_weight=class_weight  # Pass class weights to fit
        )
    
    # Always save the final model
    print(f"Saving model to {final_model_path}")
    model.save(final_model_path)
    print(f"Model saved successfully!")
    
    return model


def evaluate_model(model_path, test_data_path, threshold=0.5):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to the test dataset directory
        threshold: Confidence threshold for positive class prediction
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    try:
        model = load_model(model_path, custom_objects={
            'precision_m': precision_m,
            'recall_m': recall_m,
            'f1_m': f1_m
        })
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Check if test_data_path is a directory or a file
    if os.path.isfile(test_data_path):
        # Single file evaluation
        print(f"Evaluating single file: {test_data_path}")
        result, confidence = predict(model_path, test_data_path, threshold)
        print(f"Prediction: {result} (Confidence: {confidence:.2%})")
        return None
    
    # Check directory structure
    has_structured_dirs = os.path.exists(os.path.join(test_data_path, "positive")) and \
                        os.path.exists(os.path.join(test_data_path, "negative"))
    
    data = []
    true_labels = []
    filenames = []
    
    if has_structured_dirs:
        # Structured directory with positive/negative folders
        # Load positive examples (brand font)
        positive_paths = sorted(list(paths.list_images(os.path.join(test_data_path, "positive"))))
        print(f"Found {len(positive_paths)} positive test examples")
        
        # Load negative examples (other fonts)
        negative_paths = sorted(list(paths.list_images(os.path.join(test_data_path, "negative"))))
        print(f"Found {len(negative_paths)} negative test examples")
        
        # Process all test images
        image_paths = positive_paths + negative_paths
        
        for image_path in image_paths:
            # Determine true label
            true_label = 1 if "positive" in image_path else 0
            true_labels.append(true_label)
            
            # Load and process image
            pil_img = pil_image(image_path)
            img_array = img_to_array(pil_img)
            data.append(img_array)
            filenames.append(os.path.basename(image_path))
    else:
        # Flat directory structure - treat all as unknown
        print("Directory doesn't have positive/negative structure.")
        print("Treating all images as unknown (no ground truth labels).")
        
        # Get all image files
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(test_data_path, ext)))
        
        print(f"Found {len(image_paths)} image files for testing")
        
        if not image_paths:
            print(f"No image files found in {test_data_path}")
            print("Please ensure the directory contains image files or has the correct structure")
            print("Expected structure: test_data/positive/ and test_data/negative/")
            return None
        
        # Process all test images without labels
        for image_path in image_paths:
            # Load and process image
            try:
                pil_img = pil_image(image_path)
                img_array = img_to_array(pil_img)
                data.append(img_array)
                filenames.append(os.path.basename(image_path))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    # Check if we have any data to evaluate
    if not data:
        print("No valid images found for evaluation.")
        return None
    
    # Convert to numpy arrays and normalize
    data = np.asarray(data, dtype="float") / 255.0
    
    # Get predictions
    print(f"Making predictions on {len(data)} images...")
    raw_predictions = model.predict(data)
    
    # Check for all-negative predictions before applying threshold
    max_confidence = np.max(raw_predictions)
    if max_confidence < threshold:
        print(f"\nWARNING: All predictions are below the threshold ({threshold})")
        print(f"Highest confidence score is only {max_confidence:.4f}")
        print(f"Consider lowering the threshold (current: {threshold}, try: {max_confidence * 0.9:.2f})")
    
    # Convert raw predictions to binary predictions using threshold
    binary_predictions = (raw_predictions >= threshold).astype(int)
    
    # Check for all-negative predictions
    if np.sum(binary_predictions) == 0:
        print("\nWARNING: Model is predicting ALL samples as negative (Not Traget Font)")
        print("This could indicate a model issue or threshold that's too high")
    
    # If we have true labels, calculate metrics
    has_true_labels = len(true_labels) > 0
    if has_true_labels:
        true_labels = np.array(true_labels)
        
        # Calculate metrics using scikit-learn functions with zero_division parameter
        precision = precision_score(true_labels, binary_predictions, zero_division=0)
        recall = recall_score(true_labels, binary_predictions, zero_division=0)
        f1 = f1_score(true_labels, binary_predictions, zero_division=0)
        cm = confusion_matrix(true_labels, binary_predictions)
        
        # Calculate accuracy manually
        accuracy = np.sum(binary_predictions.flatten() == true_labels) / len(true_labels)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nConfusion Matrix:")
        print("          Predicted")
        print("         Neg    Pos")
        print(f"Actual Neg {cm[0][0]:4d}   {cm[0][1]:4d}")
        print(f"       Pos {cm[1][0]:4d}   {cm[1][1]:4d}")
        
        # Return metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm
        }
    else:
        metrics = {}
    
    # Display predictions for all images
    print("\nPredictions:")
    for i, filename in enumerate(filenames):
        pred_label = "Traget Font" if binary_predictions[i][0] == 1 else "Not Traget Font"
        confidence = raw_predictions[i][0] if binary_predictions[i][0] == 1 else 1 - raw_predictions[i][0]
        print(f"Image: {filename}")
        print(f"  Predicted: {pred_label} (Confidence: {confidence:.2%})")
        
        # If we have true labels, show ground truth
        if has_true_labels:
            true_label = "Traget Font" if true_labels[i] == 1 else "Not Traget Font" 
            correct = "✓" if binary_predictions[i][0] == true_labels[i] else "✗"
            print(f"  True: {true_label} {correct}")
        
        print("")
    
    return metrics


def predict(model_path, image_path, threshold=0.5):
    """
    Predict whether an image contains the Denton-Light font.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the image to predict
        threshold: Confidence threshold for positive detection
        
    Returns:
        Prediction result and confidence score
    """
    # Load model
    model = load_model(model_path, custom_objects={
        'precision_m': precision_m,
        'recall_m': recall_m,
        'f1_m': f1_m
    })
    
    # Process image
    pil_im = Image.open(image_path).convert('L')
    pil_im = pil_im.resize((105, 105))
    img_array = img_to_array(pil_im)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)[0][0]
    
    # Determine result based on threshold
    result = "Denton-Light Font" if prediction >= threshold else "Not Denton-Light Font"
    confidence = prediction if result == "Denton-Light Font" else 1 - prediction
    
    # Get model name and image name for result plot
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_plot_path = f"{model_name}_{image_name}_prediction.png"
    
    # Display results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(pil_im, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(['Not Denton-Light Font', 'Denton-Light Font'], [1-prediction, prediction])
    plt.ylim(0, 1)
    plt.title(f'Prediction: {result} ({confidence:.2%})')
    
    plt.tight_layout()
    plt.savefig(result_plot_path)
    plt.close()
    print(f"Prediction result saved to {result_plot_path}")
    
    return result, prediction


def test_augmentation(image_path, output_dir="augmented_images"):
    """
    Test data augmentation on a single image or all images in a directory.
    
    Args:
        image_path: Path to the input image or directory containing images
        output_dir: Directory to save augmented images (optional)
    
    Returns:
        Dictionary of augmented images
    """
    # Check if image_path is a directory
    if os.path.isdir(image_path):
        image_files = glob.glob(os.path.join(image_path, "*.jpg")) + \
                     glob.glob(os.path.join(image_path, "*.jpeg")) + \
                     glob.glob(os.path.join(image_path, "*.png"))
        if not image_files:
            print(f"No image files found in {image_path}")
            return {}
            
        # Process the first image in the directory
        print(f"Found {len(image_files)} images. Processing first image: {os.path.basename(image_files[0])}")
        image_path = image_files[0]
    
    # Load and process original image
    pil_img = pil_image(image_path)
    
    # Apply augmentations
    augmented_images = apply_augmentations(pil_img)
    
    # Save images if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        for aug_name, img in augmented_images.items():
            img.save(os.path.join(output_dir, f"{aug_name}.jpg"))
        
        # Create a grid visualization
        rows = (len(augmented_images) + 3) // 4  # Ceiling division by 4
        cols = min(4, len(augmented_images))
        
        plt.figure(figsize=(12, 3 * rows))
        for i, (aug_name, img) in enumerate(augmented_images.items()):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(aug_name)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "augmentation_grid.jpg"))
        plt.close()
    
    return augmented_images


def main():
    parser = argparse.ArgumentParser(description='Denton-Light Font Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create dataset command
    create_parser = subparsers.add_parser('create_dataset', help='Create a dataset for training')
    create_parser.add_argument('--brand_font', '-bf', default='fonts/Denton-Light.otf', 
                               help='Path to the Denton-Light font file')
    create_parser.add_argument('--other_fonts', '-of', default='fonts',
                              help='Directory containing other font files')
    create_parser.add_argument('--output', '-o', required=True, help='Output directory for dataset')
    create_parser.add_argument('--count', '-c', type=int, default=500, help='Number of images per font')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the font detection model')
    train_parser.add_argument('--data', '-d', required=True, help='Path to the dataset directory')
    train_parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of training epochs')
    train_parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size for training')
    train_parser.add_argument('--domain_adaptation', '-da', action='store_true', help='Use domain adaptation')
    train_parser.add_argument('--output', '-o', default="denton_font_model_final.keras", help='Path to save the final model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new images')
    predict_parser.add_argument('--model', '-m', required=True, help='Path to the trained model')
    predict_parser.add_argument('--image', '-i', required=True, help='Path to the image to predict')
    predict_parser.add_argument('--threshold', '-t', type=float, default=0.5, 
                                help='Confidence threshold for positive detection')
    
    # Evaluate model command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model on test data')
    evaluate_parser.add_argument('--model', '-m', required=True, help='Path to the trained model')
    evaluate_parser.add_argument('--data', '-d', required=True, help='Path to the test dataset directory')
    evaluate_parser.add_argument('--threshold', '-t', type=float, default=0.5,
                                help='Confidence threshold for positive detection')
    
    # Test augmentation command
    test_parser = subparsers.add_parser('test_augmentation', help='Test data augmentation on a single image or directory')
    test_parser.add_argument('--image', '-i', required=True, help='Path to the image file or directory containing images')
    test_parser.add_argument('--output', '-o', help='Output directory for augmented images')
    
    args = parser.parse_args()
    
    if args.command == 'create_dataset':
        create_dataset(args.brand_font, args.other_fonts, args.output, args.count)
    elif args.command == 'train':
        train_model(args.data, args.epochs, args.batch_size, args.domain_adaptation, args.output)
    elif args.command == 'predict':
        result, confidence = predict(args.model, args.image, args.threshold)
        print(f"Prediction: {result} (Confidence: {confidence:.2%})")
    elif args.command == 'evaluate':
        evaluate_model(args.model, args.data, args.threshold)
    elif args.command == 'test_augmentation':
        test_augmentation(args.image, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 