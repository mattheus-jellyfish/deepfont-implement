# %% Setup

# from matplotlib.pyplot import imshow
# import matplotlib.cm as cm
# import matplotlib.pylab as plt
import os
import random
import PIL
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import cv2
import argparse
import itertools
import string
import numpy as np
import datetime
from imutils import paths
from tensorflow.keras.utils import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import callbacks, optimizers
from keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose
from keras import backend as K
import glob
from pathlib import Path
import tensorflow as tf

# %% Create dataset
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
            # (mean=10, std=40, bounded between 0 and 50)
            spacing = min(max(0, int(np.random.normal(10, 40))), 50)
            
            # Move to next character position
            x_start += char_width + spacing
            
            # If we've run out of space, stop rendering
            if x_start > width - 10:
                break
    
    return image

def create_dataset(input_dir="fonts", output_dir="train_data", count_per_font=500):
    """
    Create a dataset of images with all available fonts.
    
    Args:
        output_dir: Directory to save generated images
        count_per_font: Number of images to generate per font
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all font files
    font_files = glob.glob(f"{input_dir}/*")
    if not font_files:
        print("No font files found in 'fonts' directory. Please add some fonts first.")
        return
    
    print(f"Found {len(font_files)} font files.")
    
    # Process each font
    for font_file in font_files:
        font_name = Path(font_file).stem
        print(f"Generating {count_per_font} samples with font: {font_name}")
        
        # Create font-specific directory
        Path(f"{output_dir}/{font_name}").mkdir(parents=True, exist_ok=True)
        
        try:
            for c in range(count_per_font):
                height = 80
                width = 600
                fontsize = 48
                font = ImageFont.truetype(font_file, fontsize)
                msg = get_random_text(random.randint(6, 18))
                
                # Enable variable character spacing (randomly) for half of the samples
                use_variable_spacing = random.choice([True, False])
                pil_img = create_image((width, height), msg, font, variable_spacing=use_variable_spacing)
                
                # Save image
                pil_img.save(f"{output_dir}/{font_name}/{msg}.jpg")
        except Exception as e:
            print(f"Error processing font {font_name}: {e}")
            continue

# %% Augmentations
def pil_image(img_path):
    """Convert image to PIL format and resize."""
    pil_im =PIL.Image.open(img_path).convert('L')
    pil_im=pil_im.resize((105,105))
    #imshow(np.asarray(pil_im))
    return pil_im

def noise_image(pil_im):
    """Add random noise to image."""
    img_array = np.asarray(pil_im)
    mean = 0.0
    # Change std from 5 to 3 as per reference implementation
    std = 3
    noisy_img = img_array + np.random.normal(mean, std, img_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    noise_img = PIL.Image.fromarray(np.uint8(noisy_img_clipped))
    noise_img=noise_img.resize((105,105))
    return noise_img

def blur_image(pil_im):
    """Apply Gaussian blur to image with random radius between 2.5-3.5."""
    # Use random radius between 2.5 and 3.5 as in reference implementation
    radius = random.uniform(2.5, 3.5)
    blur_img = pil_im.filter(ImageFilter.GaussianBlur(radius=radius))
    blur_img=blur_img.resize((105,105))
    return blur_img

def affine_rotation(img):
    """Apply random affine transformation to image."""
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
        
    rows, columns = img.shape
    
    # Generate random points for affine transformation
    # Create base points first
    src_pts = np.float32([[0, 0], [columns-1, 0], [0, rows-1]])
    
    # Create randomly perturbed destination points
    # Add random offsets (between -10% and +10% of image dimensions)
    x_offset_max = columns * 0.1
    y_offset_max = rows * 0.1
    
    dst_pts = np.float32([
        [random.uniform(-x_offset_max, x_offset_max), 
         random.uniform(-y_offset_max, y_offset_max)],
        [columns-1 + random.uniform(-x_offset_max, x_offset_max), 
         random.uniform(-y_offset_max, y_offset_max)],
        [random.uniform(-x_offset_max, x_offset_max), 
         rows-1 + random.uniform(-y_offset_max, y_offset_max)]
    ])
    
    # Get the transformation matrix
    A = cv2.getAffineTransform(src_pts, dst_pts)
    output = cv2.warpAffine(img, A, (columns, rows))
    affine_img = PIL.Image.fromarray(np.uint8(output))
    affine_img=affine_img.resize((105,105))
    return affine_img

def gradient_fill(img):
    """
    Apply background gradient shading.
    Replaces the incorrect Laplacian edge detection with proper gradient background.
    """
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    
    # Create a gradient background (lighter on one side, darker on the other)
    rows, cols = img.shape
    
    # Determine gradient direction (horizontal, vertical, or diagonal)
    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
    
    # Create gradient array
    gradient = np.zeros((rows, cols), dtype=np.uint8)
    
    if direction == 'horizontal':
        for col in range(cols):
            # Linear gradient from 220 to 250 (subtle light variation)
            value = int(220 + (col / cols) * 30)
            gradient[:, col] = value
    elif direction == 'vertical':
        for row in range(rows):
            # Linear gradient from 220 to 250 (subtle light variation)
            value = int(220 + (row / rows) * 30)
            gradient[row, :] = value
    else:  # diagonal
        for row in range(rows):
            for col in range(cols):
                # Diagonal gradient
                value = int(220 + ((row + col) / (rows + cols)) * 30)
                gradient[row, col] = value
    
    # Create a binary threshold to separate text (black) from background (white)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
    # Combine: keep text from original image, use gradient for background
    result = np.copy(img)
    result[binary == 255] = gradient[binary == 255]
    
    # Resize to expected dimensions
    result = cv2.resize(result, (105, 105))
    return result

def variable_aspect_ratio(pil_img):
    """
    Apply variable aspect ratio by randomly squeezing or stretching the width.
    Width is randomly adjusted by a ratio drawn uniformly from the range [5/6, 7/6].
    """
    width, height = pil_img.size
    
    # Random aspect ratio factor drawn from [5/6, 7/6] = [0.833, 1.167]
    aspect_ratio_factor = random.uniform(5/6, 7/6)
    
    # Calculate new width maintaining height
    new_width = int(width * aspect_ratio_factor)
    
    # Resize with the new aspect ratio
    stretched_img = pil_img.resize((new_width, height), PIL.Image.LANCZOS)
    
    # Ensure final size is 105x105 by center cropping or padding
    if new_width > 105:
        # Center crop if wider than 105
        left = (new_width - 105) // 2
        stretched_img = stretched_img.crop((left, 0, left + 105, 105))
    else:
        # Pad with white if narrower than 105
        result_img = PIL.Image.new('L', (105, 105), 255)
        paste_x = (105 - new_width) // 2
        result_img.paste(stretched_img, (paste_x, 0))
        stretched_img = result_img
    
    return stretched_img

def apply_augmentations(pil_img, augment_types=None):
    """
    Apply various augmentations to an input image.
    Uses random selection of augmentations instead of all combinations.
    
    Args:
        pil_img: PIL Image to augment
        augment_types: List of augmentation types to apply
    
    Returns:
        Dictionary of augmented images with augmentation type as key
    """
    if augment_types is None:
        augment_types = ["blur", "noise", "affine", "gradient", "aspect_ratio"]
    
    augmented_images = {}
    
    # Always include original image
    augmented_images['original'] = pil_img
    
    # Random number of augmentations to apply (0 to 3)
    # This prevents excessive stacking of all augmentations
    num_augs = random.randint(0, min(3, len(augment_types)))
    
    if num_augs > 0:
        # Randomly select which augmentations to apply
        selected_augs = random.sample(augment_types, num_augs)
        
        # Apply individual augmentations
        for aug_type in augment_types:
            if aug_type == 'noise':
                augmented_images['noise'] = noise_image(pil_img)
            elif aug_type == 'blur':
                augmented_images['blur'] = blur_image(pil_img)
            elif aug_type == 'affine':
                open_cv_img = np.array(pil_img)
                augmented_images['affine'] = affine_rotation(open_cv_img)
            elif aug_type == 'gradient':
                open_cv_img = np.array(pil_img)
                gradient_img = gradient_fill(open_cv_img)
                if isinstance(gradient_img, np.ndarray):
                    gradient_img = PIL.Image.fromarray(np.uint8(np.clip(gradient_img, 0, 255)))
                augmented_images['gradient'] = gradient_img
            elif aug_type == 'aspect_ratio':
                augmented_images['aspect_ratio'] = variable_aspect_ratio(pil_img)
        
        # Apply a random combination of augmentations (if selected)
        if len(selected_augs) > 1:
            combo_name = "+".join(selected_augs)
            temp_img = pil_img
            
            for aug_type in selected_augs:
                if aug_type == 'noise':
                    temp_img = noise_image(temp_img)
                elif aug_type == 'blur':
                    temp_img = blur_image(temp_img)
                elif aug_type == 'affine':
                    open_cv_combo = np.array(temp_img)
                    temp_img = affine_rotation(open_cv_combo)
                elif aug_type == 'gradient':
                    open_cv_combo = np.array(temp_img)
                    temp_img = gradient_fill(open_cv_combo)
                elif aug_type == 'aspect_ratio':
                    temp_img = variable_aspect_ratio(temp_img)
                
                # Convert to PIL image if needed
                if isinstance(temp_img, np.ndarray):
                    temp_img = PIL.Image.fromarray(np.uint8(np.clip(temp_img, 0, 255)))
            
            augmented_images[combo_name] = temp_img
    
    return augmented_images

# %% Create model

def create_font_mapping(fonts_dir="fonts"):
    """
    Create mapping of font names to indices.
    
    Args:
        fonts_dir: Directory containing font files
        
    Returns:
        Dictionary mapping font names to indices
    """
    font_files = sorted(glob.glob(f"{fonts_dir}/*"))
    if not font_files:
        print(f"Warning: No font files found in '{fonts_dir}' directory. Make sure fonts are placed there.")
        return {}
    font_names = [os.path.splitext(os.path.basename(f))[0] for f in font_files]
    return {font: idx for idx, font in enumerate(font_names)}

# Global font mapping - will be initialized properly when needed
FONT_MAPPING = {}

def conv_label(label, fonts_dir="fonts"):
    """
    Convert a font name to its corresponding index.
    
    Args:
        label: Font name
        fonts_dir: Directory containing font files
        
    Returns:
        Index corresponding to the font name
    """
    global FONT_MAPPING
    # Initialize mapping if empty
    if not FONT_MAPPING:
        FONT_MAPPING = create_font_mapping(fonts_dir)
    return FONT_MAPPING.get(label)

def create_model():
    """Create and return the CNN model for font detection."""
    # Create model with Input layer as first layer (fixes input_shape warning)
    inputs = Input(shape=(105, 105, 1))
    
    # First convolutional block
    x = Conv2D(64, kernel_size=(48, 48), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second convolutional block
    x = Conv2D(128, kernel_size=(24, 24), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Transpose and upsampling blocks
    x = Conv2DTranspose(128, (24, 24), strides=(2, 2), activation='relu', padding='same', kernel_initializer='uniform')(x)
    x = UpSampling2D(size=(2, 2))(x)
    
    x = Conv2DTranspose(64, (12, 12), strides=(2, 2), activation='relu', padding='same', kernel_initializer='uniform')(x)
    x = UpSampling2D(size=(2, 2))(x)
    
    # Final convolutional blocks
    x = Conv2D(256, kernel_size=(12, 12), activation='relu')(x)
    x = Conv2D(256, kernel_size=(12, 12), activation='relu')(x)
    x = Conv2D(256, kernel_size=(12, 12), activation='relu')(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2383, activation='relu')(x)
    outputs = Dense(len(FONT_MAPPING), activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    return model

def check_gpu():
    """
    Check if TensorFlow can access a GPU and print detailed information about it.
    This function should be called before training to verify hardware acceleration.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    print("\n==== TensorFlow GPU Configuration ====")
    print(f"TensorFlow version: {tf.__version__}")
    
    # List physical devices
    print("\nPhysical devices detected by TensorFlow:")
    devices = tf.config.list_physical_devices()
    for device in devices:
        print(f" - {device.name} ({device.device_type})")
    
    # Check specifically for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("\nNo GPU detected. Training will use CPU which may be slow.")
        print("If you have a CUDA-compatible GPU, ensure you have installed:")
        print(" - CUDA Toolkit")
        print(" - cuDNN")
        print(" - GPU-compatible TensorFlow version")
        return False
    
    print(f"\nFound {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f" - GPU {i}: {gpu.name}")
    
    # Get detailed GPU info
    try:
        from tensorflow.python.client import device_lib
        gpu_details = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        for i, gpu in enumerate(gpu_details):
            desc = gpu.physical_device_desc
            print(f"\nGPU {i} Details:")
            # Parse the device description for more readable output
            for part in desc.split(", "):
                print(f" - {part}")
    except:
        print("Could not retrieve detailed GPU information")
    
    # Check memory allocation
    print("\nGPU Memory Management:")
    try:
        # Try to limit memory growth to avoid taking all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(" - Memory growth enabled for all GPUs")
    except RuntimeError as e:
        print(f" - Error configuring memory growth: {e}")
    
    # Run a simple test to verify GPU operation
    print("\nRunning test computation on GPU...")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
    
    print(f"Test matrix multiplication result shape: {c.shape}")
    print("Test successful - GPU is operational!")
    
    # Additional monitoring tips
    print("\nTo monitor GPU usage during training:")
    print(" - Use 'nvidia-smi' in a separate terminal window (refresh with: nvidia-smi -l 1)")
    print(" - Use 'gpustat' if installed (pip install gpustat)")
    print(" - Check TensorBoard for device placement visualization")
    
    print("\nGPU is available and configured for training!")
    return True

def train(batch_size=128, epochs=25, data_path="train_data/", output_dir="runs/experiment", model_name="model.keras", fonts_dir="fonts"):
    """
    Train the font detection model.
    
    Args:
        batch_size: Batch size for training
        epochs: Number of training epochs
        data_path: Path to dataset
        output_dir: Directory to save model and logs (will be created if it doesn't exist)
        model_name: Name of the model file to save (using .keras extension)
        fonts_dir: Directory containing font files
    """
    # Check GPU availability before training
    gpu_available = check_gpu()
    if not gpu_available:
        print("Warning: Training on CPU may be slow. Consider using a GPU if available.")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    log_dir = os.path.join(output_dir, "logs")
    
    # Initialize font mappings
    global FONT_MAPPING
    FONT_MAPPING = create_font_mapping(fonts_dir)
    
    if not FONT_MAPPING:
        raise ValueError(f"No fonts found in '{fonts_dir}'. Cannot proceed with training.")
    
    data=[]
    labels=[]
    imagePaths = sorted(list(paths.list_images(data_path)))
    random.seed(42)
    random.shuffle(imagePaths)

    # Define augmentation types
    augment_types = ["blur", "noise", "affine", "gradient", "aspect_ratio"]

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        label = conv_label(label, fonts_dir)
        pil_img = pil_image(imagePath)
        
        # Apply augmentations using the new function
        augmented_images = apply_augmentations(pil_img, augment_types)
        
        # Add all augmented images to the dataset
        for aug_img in augmented_images.values():
            img_array = img_to_array(aug_img)
            data.append(img_array)
            labels.append(label)

    data = np.asarray(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    # convert the labels from integers to vectors
    num_classes = len(FONT_MAPPING)
    trainY = to_categorical(trainY, num_classes=num_classes)
    testY = to_categorical(testY, num_classes=num_classes)

    # aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)
    K.set_image_data_format('channels_last')

    model = create_model()
    
    # Use SGD optimizer without the deprecated decay parameter
    # Replace with learning rate scheduler if needed
    sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    
    # Add learning rate scheduler to callbacks if decay functionality is needed
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Create TensorBoard callback
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,  # Generate histograms of weights
        write_graph=True,  # Visualize graph
        write_images=True,  # Visualize model weights as images
        update_freq='epoch',  # Update at end of each epoch
        profile_batch=0  # No profiling for now
    )
    
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')

    checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    callbacks_list = [early_stopping, checkpoint, lr_scheduler, tensorboard_callback]
    
    # Save training config to output directory
    with open(os.path.join(output_dir, "training_config.txt"), "w") as f:
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Fonts directory: {fonts_dir}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Training date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    model.fit(trainX, 
        trainY,
        shuffle=True,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(testX, testY),
        callbacks=callbacks_list)
    score = model.evaluate(testX, testY, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save and return the model path
    print(f"Saving model to {model_path}")
    model.save(model_path)
    print(f"Model saved successfully in Keras format")
    
    return model_path

# Create a reverse mapping of indices to font names
def create_reverse_font_mapping(fonts_dir="fonts"):
    """
    Create mapping of indices to font names.
    
    Args:
        fonts_dir: Directory containing font files
        
    Returns:
        Dictionary mapping indices to font names
    """
    font_files = sorted(glob.glob(f"{fonts_dir}/*"))
    font_names = [os.path.splitext(os.path.basename(f))[0] for f in font_files]
    return {idx: font for idx, font in enumerate(font_names)}

# Global reverse font mapping - will be initialized properly when needed
REVERSE_FONT_MAPPING = {}

def rev_conv_label(label, fonts_dir="fonts"):
    """
    Return the font name for the given index.
    
    Args:
        label: Index
        fonts_dir: Directory containing font files
        
    Returns:
        Font name corresponding to the index
    """
    global REVERSE_FONT_MAPPING
    # Initialize mapping if empty
    if not REVERSE_FONT_MAPPING:
        REVERSE_FONT_MAPPING = create_reverse_font_mapping(fonts_dir)
    return REVERSE_FONT_MAPPING.get(label, "Unknown")

def get_data(img_path):
    """Process an image for prediction"""
    pil_im = PIL.Image.open(img_path).convert('L')
    pil_im = pil_im.resize((105,105))
    org_img = img_to_array(pil_im)
    data = []
    data.append(org_img)
    data = np.asarray(data, dtype="float") / 255.0
    
    # Display the image for visualization
    plt_img = np.array(pil_im)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.imshow(plt_img, cmap='gray')
    plt.title(f"Image: {os.path.basename(img_path)}")
    plt.axis('on')
    plt.show()
    
    return data

def predict(img_path, model_file, fonts_dir="fonts"):
    """
    Predict the font used in an image.
    
    Args:
        img_path: Path to the image file
        model_file: Path to the trained model file (.keras format)
        fonts_dir: Directory containing font files
    """
    # Initialize reverse font mapping
    global REVERSE_FONT_MAPPING
    REVERSE_FONT_MAPPING = create_reverse_font_mapping(fonts_dir)
    
    if not REVERSE_FONT_MAPPING:
        raise ValueError(f"No fonts found in '{fonts_dir}'. Cannot proceed with prediction.")
    
    data = get_data(img_path)

    print(f"Loading model from {model_file}")
    model = load_model(model_file)
    predict_y = model.predict(data)
    
    # Get the top 5 predictions
    top_indices = np.argsort(predict_y[0])[-5:][::-1]  # Sort and get last 5 (highest) values, then reverse to get descending order
    
    print(f"\nFont prediction for {img_path}:")
    print("-------------------------------")
    
    for i, idx in enumerate(top_indices):
        font_name = rev_conv_label(idx, fonts_dir)
        probability = predict_y[0][idx] * 100  # Convert to percentage
        print(f"{i+1}. {font_name}: {probability:.2f}%")
    
    # Also keep the original output for compatibility
    classes_y = np.argmax(predict_y, axis=1)
    top_font = rev_conv_label(classes_y[0], fonts_dir)
    print(f"\nTop prediction: {top_font}")

# %% Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Font detection tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create dataset command
    create_parser = subparsers.add_parser('create_dataset', help='Create a dataset for training')
    create_parser.add_argument('--input', '-i', default='fonts', 
                              help='Input directory for fonts')
    create_parser.add_argument('--output', '-o', default='train_data', 
                              help='Output directory for dataset')
    create_parser.add_argument('--count', '-c', type=int, default=500, 
                              help='Number of images per font')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the font detection model')
    train_parser.add_argument('--epochs', '-e', type=int, required=True, 
                             help='Number of training epochs')
    train_parser.add_argument('--batch_size', '-b', type=int, default=128, 
                             help='Batch size for training')
    train_parser.add_argument('--data', '-d', default='train_data', 
                             help='Path to the dataset directory')
    train_parser.add_argument('--output-dir', '-o', default='runs/experiment',
                             help='Directory to save model and logs')
    train_parser.add_argument('--model-name', '-m', default='model.keras',
                             help='Name of the model file to save (with .keras extension)')
    train_parser.add_argument('--fonts-dir', '-f', default='fonts',
                             help='Directory containing font files for mapping')
    
    # Predict font command
    predict_parser = subparsers.add_parser('predict', help='Predict font from an image')
    predict_parser.add_argument('--data', '-d', required=True,
                               help='Path to the image file')
    predict_parser.add_argument('--model', '-m', required=True,
                               help='Path to the trained model file (.keras format)')
    predict_parser.add_argument('--fonts-dir', '-f', default='fonts',
                               help='Directory containing font files for mapping')
    
    # Add GPU check command
    gpu_parser = subparsers.add_parser('check_gpu', help='Check GPU availability for TensorFlow')
    
    args = parser.parse_args()
    
    if args.command == 'create_dataset':
        create_dataset(args.input, args.output, args.count)
    elif args.command == 'train':
        saved_model_path = train(
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            data_path=args.data, 
            output_dir=args.output_dir,
            model_name=args.model_name,
            fonts_dir=args.fonts_dir
        )
        print(f"Training complete. Model saved to: {saved_model_path}")
        print(f"To view training history in TensorBoard, run: tensorboard --logdir {os.path.join(args.output_dir, 'logs')}")
    elif args.command == 'predict':
        predict(args.data, args.model, args.fonts_dir)
    elif args.command == 'check_gpu':
        check_gpu()
    else:
        parser.print_help()
        
    # Example commands:
    # To create a dataset with default settings:
    # python ai_labs_font_detector.py create_dataset
    #
    # To create a dataset with custom settings:
    # python ai_labs_font_detector.py create_dataset --input my_fonts --output my_dataset --count 1000
    #
    # To train with custom settings and font directory:
    # python ai_labs_font_detector.py train --epochs 100 --batch_size 64 --data my_dataset --output-dir runs/my_experiment --model-name my_model.keras --fonts-dir my_fonts
    #
    # To predict font from an image with custom font directory:
    # python ai_labs_font_detector.py predict --data sample_image.jpg --model runs/my_experiment/model.keras --fonts-dir my_fonts
    #
    # To view training history in TensorBoard after training:
    # tensorboard --logdir runs/my_experiment/logs
    #
    # To check GPU availability:
    # python ai_labs_font_detector.py check_gpu
