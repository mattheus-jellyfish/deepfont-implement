import os
import argparse
import random
import string
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import glob
import itertools
from imutils import paths
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K


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


def apply_augmentations(pil_img, augment_types=None):
    """
    Apply various augmentations to an input image.
    
    Args:
        pil_img: PIL Image to augment
        augment_types: List of augmentation types to apply (default: ["blur", "noise", "affine", "gradient"])
    
    Returns:
        Dictionary of augmented images with augmentation type as key
    """
    if augment_types is None:
        augment_types = ["blur", "noise", "affine", "gradient"]
    
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
                
                # Convert to PIL image if needed
                if isinstance(temp_img, np.ndarray):
                    temp_img = Image.fromarray(np.uint8(np.clip(temp_img, 0, 255)))
            
            augmented_images[combo_name] = temp_img
    
    return augmented_images


def get_random_text(length):
    """Generate random text of specified length."""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def create_image(size, message, font):
    """Create an image with text using the specified font."""
    width, height = size
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    
    # Get the bounding box of the text
    bbox = font.getbbox(message)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), message, font=font, fill='black')
    return image


def create_dataset(brand_font_path, other_fonts_dir=None, output_dir="train_data", count_per_font=500):
    """
    Create a dataset of images with brand font (positive class) and other fonts (negative class).
    
    Args:
        brand_font_path: Path to the brand font file
        other_fonts_dir: Directory containing other font files (optional)
        output_dir: Directory to save generated images
        count_per_font: Number of images to generate per font
    """
    # Create output directories
    Path(f"{output_dir}/positive").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/negative").mkdir(parents=True, exist_ok=True)
    
    # Generate positive samples with brand font
    brand_font_name = Path(brand_font_path).stem
    print(f"Generating {count_per_font} positive samples with font: {brand_font_name}")
    
    for c in range(count_per_font):
        height = 80
        width = 600
        fontsize = 48
        font = ImageFont.truetype(brand_font_path, fontsize)
        msg = get_random_text(random.randint(6, 18))
        pil_img = create_image((width, height), msg, font)
        Path(f"{output_dir}/positive/{brand_font_name}").mkdir(parents=True, exist_ok=True)
        pil_img.save(f"{output_dir}/positive/{brand_font_name}/{msg}.jpg")
    
    # Generate negative samples with other fonts if directory is provided
    if other_fonts_dir:
        other_font_files = [f for f in glob.glob(f"{other_fonts_dir}/*") 
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
            
            for c in range(samples_per_font):
                height = 80
                width = 600
                fontsize = 48
                font = ImageFont.truetype(font_file, fontsize)
                msg = get_random_text(random.randint(6, 18))
                pil_img = create_image((width, height), msg, font)
                pil_img.save(f"{output_dir}/negative/{font_name}_{msg}.jpg")


def create_model():
    """Create a binary classification model for font detection."""
    model = Sequential()
    
    # Feature extraction
    model.add(Conv2D(64, kernel_size=(24, 24), activation='relu', input_shape=(105, 105, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(12, 12), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Classification
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary output
    
    return model


def train_model(data_path, epochs=20, batch_size=32):
    """
    Train the font detection model.
    
    Args:
        data_path: Path to the dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        
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
    
    # Combine and shuffle
    image_paths = positive_paths + negative_paths
    random.seed(42)
    random.shuffle(image_paths)
    
    # Apply augmentations to increase dataset size
    augment_types = ["blur", "noise", "affine", "gradient"]
    
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
    
    # Split data
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    
    # Create and compile model
    model = create_model()
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Set up callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0, 
        patience=10, 
        verbose=1, 
        mode='min'
    )
    
    filepath = "denton_font_model.h5"
    checkpoint = callbacks.ModelCheckpoint(
        filepath, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min'
    )
    
    callbacks_list = [early_stopping, checkpoint]
    
    # Train model
    print("Starting model training...")
    history = model.fit(
        trainX, 
        trainY,
        validation_data=(testX, testY),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_list
    )
    
    # Evaluate model
    score = model.evaluate(testX, testY, verbose=0)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('denton_training_history.png')
    plt.close()
    
    return model


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
    model = load_model(model_path)
    
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
    plt.savefig('denton_prediction_result.png')
    
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
    create_parser.add_argument('--brand_font', '-bf', default='Denton-Light.otf', 
                               help='Path to the Denton-Light font file')
    create_parser.add_argument('--other_fonts', '-of', help='Directory containing other font files')
    create_parser.add_argument('--output', '-o', required=True, help='Output directory for dataset')
    create_parser.add_argument('--count', '-c', type=int, default=500, help='Number of images per font')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the font detection model')
    train_parser.add_argument('--data', '-d', required=True, help='Path to the dataset directory')
    train_parser.add_argument('--epochs', '-e', type=int, default=20, help='Number of training epochs')
    train_parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size for training')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new images')
    predict_parser.add_argument('--model', '-m', required=True, help='Path to the trained model')
    predict_parser.add_argument('--image', '-i', required=True, help='Path to the image to predict')
    predict_parser.add_argument('--threshold', '-t', type=float, default=0.5, 
                                help='Confidence threshold for positive detection')
    
    # Test augmentation command
    test_parser = subparsers.add_parser('test_augmentation', help='Test data augmentation on a single image or directory')
    test_parser.add_argument('--image', '-i', required=True, help='Path to the image file or directory containing images')
    test_parser.add_argument('--output', '-o', help='Output directory for augmented images')
    
    args = parser.parse_args()
    
    if args.command == 'create_dataset':
        create_dataset(args.brand_font, args.other_fonts, args.output, args.count)
    elif args.command == 'train':
        train_model(args.data, args.epochs, args.batch_size)
    elif args.command == 'predict':
        result, confidence = predict(args.model, args.image, args.threshold)
        print(f"Prediction: {result} (Confidence: {confidence:.2%})")
    elif args.command == 'test_augmentation':
        test_augmentation(args.image, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 