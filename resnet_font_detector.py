import os
import random
import numpy as np
import datetime
import argparse
import glob
from pathlib import Path
import PIL
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import string
import cv2
from tensorflow.keras.utils import img_to_array

# Constants
INPUT_SIZE = 512
DEFAULT_FONT_SIZE = 48
DEFAULT_IMAGE_SIZE = (600, 80)  # Width, Height

# Configuration
FONT_DIR = "fonts"  # Directory containing font files
OUTPUT_DIR = "runs/experiment"  # Directory to save model and logs
MODEL_NAME = "resnet_font_model.keras"  # Default name for the model file

# Create a global font mapping
FONT_MAPPING = {}
REVERSE_FONT_MAPPING = {}

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Font Mapping Functions
def create_font_mapping(fonts_dir=FONT_DIR):
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

def create_reverse_font_mapping(fonts_dir=FONT_DIR):
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

def conv_label(label, fonts_dir=FONT_DIR):
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

def rev_conv_label(label, fonts_dir=FONT_DIR):
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

# Background image generator
def background_image_generator(background_dir="background"):
    """
    Generator that yields background images from the specified directory.
    Cycles through all images continuously.
    
    Args:
        background_dir: Directory containing background images
    
    Yields:
        PIL.Image: A PIL Image object with the background
    """
    bg_files = glob.glob(f"{background_dir}/*.jpg") + glob.glob(f"{background_dir}/*.png")
    
    if not bg_files:
        print(f"Warning: No background images found in '{background_dir}'. Using plain backgrounds.")
        while True:
            yield Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), 'white')
    
    print(f"Found {len(bg_files)} background images in {background_dir}")
    
    while True:
        for bg_file in bg_files:
            try:
                bg_img = Image.open(bg_file).convert('RGB')
                # Resize with proper aspect ratio by cropping or padding if needed
                bg_width, bg_height = bg_img.size
                if bg_width / bg_height > 1:  # Wider than tall
                    # Resize height to INPUT_SIZE, then crop width
                    new_width = int(bg_width * INPUT_SIZE / bg_height)
                    bg_img = bg_img.resize((new_width, INPUT_SIZE), Image.LANCZOS)
                    left = (new_width - INPUT_SIZE) // 2
                    bg_img = bg_img.crop((left, 0, left + INPUT_SIZE, INPUT_SIZE))
                else:  # Taller than wide or square
                    # Resize width to INPUT_SIZE, then crop height
                    new_height = int(bg_height * INPUT_SIZE / bg_width)
                    bg_img = bg_img.resize((INPUT_SIZE, new_height), Image.LANCZOS)
                    top = (new_height - INPUT_SIZE) // 2
                    bg_img = bg_img.crop((0, top, INPUT_SIZE, top + INPUT_SIZE))
                
                yield bg_img
            except Exception as e:
                print(f"Error loading background {bg_file}: {e}")
                # Provide a fallback white image
                yield Image.new('RGB', (INPUT_SIZE, INPUT_SIZE), 'white')

# English Corpus Generator
class EnglishCorpusGenerator:
    """
    Generates realistic English text for font rendering.
    """
    def __init__(self):
        """Initialize the corpus generator with a list of English words."""
        self.wordlist_file = "wordlist.txt"
        
        # Load or download English words
        if os.path.exists(self.wordlist_file):
            with open(self.wordlist_file, "r", encoding="utf-8") as f:
                self.english_words = f.read().splitlines()
        else:
            try:
                import requests
                # Fetch a list of English words from MIT's website
                word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
                response = requests.get(word_site)
                self.english_words = response.text.splitlines()
                # Save for future use
                with open(self.wordlist_file, "w", encoding="utf-8") as f:
                    f.write(response.text)
            except:
                # Fallback with a smaller set of common words
                self.english_words = [
                    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
                    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
                    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
                    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
                    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other"
                ]
    
    def generate_text(self, min_length=10, max_length=20):
        """
        Generate a random English text phrase.
        
        Args:
            min_length: Minimum number of characters
            max_length: Maximum number of characters
            
        Returns:
            str: Generated text
        """
        words = []
        current_length = 0
        target_length = random.randint(min_length, max_length)
        
        while current_length < target_length:
            word = random.choice(self.english_words)
            words.append(word)
            current_length += len(word) + 1  # +1 for space
        
        text = " ".join(words)
        
        # Truncate if too long or pad if too short
        if len(text) > target_length:
            text = text[:target_length]
        elif len(text) < target_length:
            padding = " " * (target_length - len(text))
            text += padding
            
        return text

# Update the create_dataset function to use these new components
def create_dataset(input_dir=FONT_DIR, output_dir="train_data", count_per_font=500, seed=RANDOM_SEED):
    """
    Create a dataset of images with all available fonts using background images.
    
    Args:
        input_dir: Directory containing font files
        output_dir: Directory to save generated images
        count_per_font: Number of images to generate per font
        seed: Random seed for reproducible dataset generation
    """
    # Set random seed if provided for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize background generator and text generator
    backgrounds = background_image_generator()
    text_generator = EnglishCorpusGenerator()
    
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
                # Generate a random text length between 6 and 18 characters
                text_length = random.randint(6, 18)
                text = text_generator.generate_text(min_length=text_length, max_length=text_length)
                
                # Get a background image
                bg_img = next(backgrounds)
                
                # Create font with random size (30-60)
                fontsize = random.randint(30, 60)
                font = ImageFont.truetype(font_file, fontsize)
                
                # Setup variables for text placement
                height = INPUT_SIZE
                width = INPUT_SIZE
                
                # Create a copy of the background for drawing
                img = bg_img.copy()
                draw = ImageDraw.Draw(img)
                
                # Calculate text dimensions
                bbox = font.getbbox(text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Randomize text position within the image
                max_x = width - text_width - 20
                max_y = height - text_height - 20
                
                if max_x < 20:
                    max_x = 20
                if max_y < 20:
                    max_y = 20
                    
                x = random.randint(20, max_x)
                y = random.randint(20, max_y)
                
                # Determine text color based on background
                # Use contrasting colors to ensure text is visible
                # Extract the background color at the text position
                bg_color_sample = np.array(bg_img.crop((x, y, x+10, y+10)))
                
                # Calculate average brightness of the background sample
                brightness = np.mean(bg_color_sample)
                
                # Choose contrasting color (black for bright backgrounds, white for dark backgrounds)
                text_color = 'black' if brightness > 128 else 'white'
                
                # Add text shadow or outline for better readability
                shadow_offset = 2
                shadow_color = 'white' if text_color == 'black' else 'black'
                
                # Draw text shadow/outline
                draw.text((x+shadow_offset, y+shadow_offset), text, font=font, fill=shadow_color)
                
                # Draw main text
                draw.text((x, y), text, font=font, fill=text_color)
                
                # Save the image
                save_path = f"{output_dir}/{font_name}/{font_name}_{c}.jpg"
                img.save(save_path, quality=95)
                
                # Print progress periodically
                if c % 50 == 0 and c > 0:
                    print(f"  Generated {c}/{count_per_font} images for {font_name}")
                    
        except Exception as e:
            print(f"Error processing font {font_name}: {e}")
            continue
    
    print("Dataset creation completed!")

# Integrated Data Augmentation for Training
def create_augmentation_pipeline(augmentation_level='v3'):
    """
    Create a data augmentation pipeline for the training process.
    This integrates with the TensorFlow/Keras ImageDataGenerator and adds custom augmentations.
    
    Args:
        augmentation_level: Level of augmentation ('v1', 'v2', 'v3')
        
    Returns:
        A dictionary with augmentation parameters for ImageDataGenerator and preprocessing function
    """
    # Basic augmentation parameters for ImageDataGenerator
    augmentation_params = {
        'rescale': 1./255,  # Important: This happens BEFORE preprocessing_function
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
    }
    
    # Level specific augmentations
    if augmentation_level == 'v1':
        # Basic augmentations
        augmentation_params.update({
            'rotation_range': 5,
            'brightness_range': (0.8, 1.2),
            'zoom_range': 0.1,
            'fill_mode': 'nearest',
        })
        # No custom preprocessing
        preprocessing_fn = None
        
    elif augmentation_level == 'v2':
        # Moderate augmentations
        augmentation_params.update({
            'rotation_range': 10,
            'brightness_range': (0.7, 1.3),
            'zoom_range': 0.15,
            'shear_range': 0.1,
            'fill_mode': 'nearest',
        })
        # Add custom preprocessing
        preprocessing_fn = apply_v2_augmentations
        
    elif augmentation_level == 'v3':
        # Advanced augmentations (mimicking the PyTorch v3 implementation)
        augmentation_params.update({
            'rotation_range': 15,            # Random rotation up to 15 degrees
            'brightness_range': (0.5, 1.5),  # Wider brightness range
            'zoom_range': 0.2,               # Stronger zoom
            'shear_range': 0.2,              # More shearing
            'channel_shift_range': 0.1,      # Color channel shifts
            'fill_mode': 'nearest',
            'horizontal_flip': True,         # Allow horizontal flips
        })
        # Add custom preprocessing for advanced effects
        preprocessing_fn = apply_v3_augmentations
        
    else:
        raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    return {
        'augmentation_params': augmentation_params,
        'preprocessing_function': preprocessing_fn
    }

def apply_v2_augmentations(img):
    """Apply moderate custom augmentations (v2) to the image."""
    # Convert to numpy array if it's not already
    if isinstance(img, tf.Tensor):
        img = img.numpy()
    
    # Check if image is already normalized to 0-1 range (float)
    is_normalized = img.dtype.kind == 'f'
    
    # Convert to uint8 for processing if it's normalized
    if is_normalized:
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    # Apply gaussian blur (occasionally)
    if random.random() < 0.3:
        img_uint8 = cv2.GaussianBlur(img_uint8, (5, 5), random.uniform(0.1, 1.0))
        
    # Apply random noise (occasionally)
    if random.random() < 0.2:
        noise = np.random.normal(0, 10, img_uint8.shape).astype(np.uint8)
        img_uint8 = np.clip(img_uint8 + noise, 0, 255).astype(np.uint8)
    
    # Convert back to original format
    if is_normalized:
        return img_uint8.astype(np.float32) / 255.0
    else:
        return img_uint8

def apply_v3_augmentations(img):
    """Apply advanced custom augmentations (v3) to the image.
    This mimics the PyTorch v3 transformations including GaussianBlur, RandomDownSample, and RandomNoise.
    """
    # Convert to numpy array if it's not already
    if isinstance(img, tf.Tensor):
        img = img.numpy()
    
    # Check if image is already normalized to 0-1 range (float)
    is_normalized = img.dtype.kind == 'f'
    
    # Convert to uint8 for processing if it's normalized
    if is_normalized:
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    # Stack augmentations randomly
    augmentations_to_apply = []
    
    # Gaussian blur (25% chance)
    if random.random() < 0.25:
        # Random kernel size (1, 3, or 5)
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.1, 2.0)
        augmentations_to_apply.append(
            lambda x: cv2.GaussianBlur(x, (kernel_size, kernel_size), sigma)
        )
    
    # Random downsampling then upsampling (20% chance) - similar to RandomDownSample
    if random.random() < 0.2:
        scale_factor = random.uniform(0.5, 0.9)  # Downsample by 50-90%
        augmentations_to_apply.append(
            lambda x: cv2.resize(
                cv2.resize(x, 
                          (int(x.shape[1] * scale_factor), int(x.shape[0] * scale_factor)), 
                          interpolation=cv2.INTER_AREA),
                (x.shape[1], x.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        )
    
    # Random noise (20% chance) - similar to RandomNoise
    if random.random() < 0.2:
        noise_level = random.uniform(5, 15)
        augmentations_to_apply.append(
            lambda x: np.clip(x + np.random.normal(0, noise_level, x.shape), 0, 255).astype(np.uint8)
        )
    
    # Perspective transform (10% chance) - additional augmentation
    if random.random() < 0.1:
        h, w = img_uint8.shape[:2]
        # Define source points
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # Define random destination points with a max shift of 10% of width/height
        max_shift = 0.1
        dst_pts = np.float32([
            [random.uniform(0, w * max_shift), random.uniform(0, h * max_shift)],  # top-left
            [random.uniform(w * (1 - max_shift), w), random.uniform(0, h * max_shift)],  # top-right
            [random.uniform(w * (1 - max_shift), w), random.uniform(h * (1 - max_shift), h)],  # bottom-right
            [random.uniform(0, w * max_shift), random.uniform(h * (1 - max_shift), h)]  # bottom-left
        ])
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        augmentations_to_apply.append(
            lambda x: cv2.warpPerspective(x, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        )
    
    # Apply all selected augmentations in sequence
    augmented_img = img_uint8.copy()
    for aug_fn in augmentations_to_apply:
        augmented_img = aug_fn(augmented_img)
    
    # Convert back to original format
    if is_normalized:
        return augmented_img.astype(np.float32) / 255.0
    else:
        return augmented_img

# ResNet Models
def create_resnet18_model(num_classes, pretrained=False):
    """
    Create a ResNet18 model for font classification.
    
    Args:
        num_classes: Number of font classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Compiled ResNet18 model
    """
    # Note: TensorFlow doesn't have a direct ResNet18 equivalent
    # Using ResNet50V2 with a modified structure for similar capacity
    base_model = applications.ResNet50V2(
        include_top=False,
        weights='imagenet' if pretrained else None,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        pooling='avg'
    )
    
    # Freeze the base model if using pretrained weights
    if pretrained:
        for layer in base_model.layers:
            layer.trainable = False
    
    # PyTorch resnet18 has 512 features in the final layer
    model = models.Sequential([
        base_model,
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_resnet34_model(num_classes, pretrained=False):
    """
    Create a ResNet34-like model for font classification.
    TensorFlow doesn't have a direct ResNet34 implementation, so we adapt ResNet50V2.
    
    Args:
        num_classes: Number of font classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Compiled ResNet34-like model
    """
    base_model = applications.ResNet50V2(
        include_top=False,
        weights='imagenet' if pretrained else None,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        pooling='avg'
    )
    
    # Freeze the base model if using pretrained weights
    if pretrained:
        for layer in base_model.layers:
            layer.trainable = False
    
    # PyTorch resnet34 has 512 features in the final layer
    model = models.Sequential([
        base_model,
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_resnet50_model(num_classes, pretrained=False):
    """
    Create a ResNet50 model for font classification.
    
    Args:
        num_classes: Number of font classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Compiled ResNet50 model
    """
    base_model = applications.ResNet50V2(
        include_top=False,
        weights='imagenet' if pretrained else None,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        pooling='avg'
    )
    
    # Freeze the base model if using pretrained weights
    if pretrained:
        for layer in base_model.layers:
            layer.trainable = False
    
    # PyTorch resnet50 has 2048 features in the final layer
    model = models.Sequential([
        base_model,
        layers.Dense(2048, activation='relu'),  # Match PyTorch's 2048 features
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_resnet101_model(num_classes, pretrained=False):
    """
    Create a ResNet101 model for font classification.
    
    Args:
        num_classes: Number of font classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Compiled ResNet101 model
    """
    base_model = applications.ResNet101V2(
        include_top=False,
        weights='imagenet' if pretrained else None,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        pooling='avg'
    )
    
    # Freeze the base model if using pretrained weights
    if pretrained:
        for layer in base_model.layers:
            layer.trainable = False
    
    # PyTorch resnet101 has 2048 features in the final layer
    model = models.Sequential([
        base_model,
        layers.Dense(2048, activation='relu'),  # Match PyTorch's 2048 features
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_model(model_type, num_classes, pretrained=False):
    """
    Create a model based on the specified type.
    
    Args:
        model_type: Type of ResNet model ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        num_classes: Number of font classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Compiled model
    """
    if model_type == 'resnet18':
        model = create_resnet18_model(num_classes, pretrained)
    elif model_type == 'resnet34':
        model = create_resnet34_model(num_classes, pretrained)
    elif model_type == 'resnet50':
        model = create_resnet50_model(num_classes, pretrained)
    elif model_type == 'resnet101':
        model = create_resnet101_model(num_classes, pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Use SGD with momentum like in the PyTorch implementation
    # Starting with a slightly higher learning rate (0.01) with decay
    optimizer = optimizers.SGD(
        learning_rate=0.01,  # PyTorch default is lower, but we'll use lr scheduler
        momentum=0.9,
        nesterov=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']  # Add top-k accuracy
    )
    
    return model

# Training and Prediction Functions
def check_gpu():
    """
    Check if TensorFlow can access a GPU and print detailed information about it.
    This function should be called before training to verify hardware acceleration.
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    print("\n==== TensorFlow GPU Configuration ====")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU availability
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
    
    # Configure GPU memory growth
    try:
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
    
    return True

def train(model_type='resnet50', batch_size=32, epochs=25, data_path="train_data/", 
          output_dir=OUTPUT_DIR, model_name=MODEL_NAME, fonts_dir=FONT_DIR, pretrained=False,
          augmentation_level='v3', learning_rate=0.01, fine_tune_after=10):
    """
    Train the font detection model.
    
    Args:
        model_type: Type of ResNet model ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        batch_size: Batch size for training
        epochs: Number of training epochs
        data_path: Path to dataset
        output_dir: Directory to save model and logs
        model_name: Name of the model file to save
        fonts_dir: Directory containing font files
        pretrained: Whether to use pretrained weights
        augmentation_level: Level of augmentation to apply ('v1', 'v2', 'v3')
        learning_rate: Initial learning rate
        fine_tune_after: Epoch after which to fine-tune pretrained models
    """
    # Check GPU availability
    gpu_available = check_gpu()
    if not gpu_available:
        print("Warning: Training on CPU may be slow. Consider using a GPU if available.")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize font mappings
    global FONT_MAPPING
    FONT_MAPPING = create_font_mapping(fonts_dir)
    
    if not FONT_MAPPING:
        raise ValueError(f"No fonts found in '{fonts_dir}'. Cannot proceed with training.")
    
    num_classes = len(FONT_MAPPING)
    print(f"Found {num_classes} font classes for training.")
    
    # Set up augmentation pipeline
    augmentation_config = create_augmentation_pipeline(augmentation_level)
    
    # Set up data generators with augmentation
    train_datagen = ImageDataGenerator(
        **augmentation_config['augmentation_params'],
        preprocessing_function=augmentation_config['preprocessing_function'],
        validation_split=0.2  # Use 20% for validation
    )
    
    # Flow from directory for training data
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Flow from directory for validation data
    validation_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(INPUT_SIZE, INPUT_SIZE),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create the model
    model = create_model(model_type, num_classes, pretrained)
    
    # Set up callbacks
    callbacks_list = [
        # Model checkpoint to save the best model
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        # Reduce learning rate when training plateaus (similar to CosineWarmupScheduler)
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # Reduce by factor of 10 (like PyTorch implementation)
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    # Add fine-tuning callback if using pretrained weights
    if pretrained and fine_tune_after > 0:
        # Custom early stopping that only activates after fine-tuning begins
        class DelayedEarlyStopping(callbacks.EarlyStopping):
            def __init__(self, activation_epoch, **kwargs):
                super().__init__(**kwargs)
                self.activation_epoch = activation_epoch
                self._is_active = False
                
            def on_epoch_begin(self, epoch, logs=None):
                if epoch >= self.activation_epoch and not self._is_active:
                    print(f"\nEpoch {epoch}: Activating early stopping monitoring")
                    self._is_active = True
                    # Reset the wait counter when we start monitoring
                    self.wait = 0
                    self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            
            def on_epoch_end(self, epoch, logs=None):
                if not self._is_active:
                    return  # Skip early stopping checks until activated
                super().on_epoch_end(epoch, logs)
        
        # Add delayed early stopping (only activates after fine-tuning begins)
        callbacks_list.append(
            DelayedEarlyStopping(
                activation_epoch=fine_tune_after,
                monitor='val_loss',
                patience=5,
                verbose=1,
                mode='min'
            )
        )
        
        class FineTuningCallback(callbacks.Callback):
            def __init__(self, fine_tune_epoch):
                super().__init__()
                self.fine_tune_epoch = fine_tune_epoch
                
            def on_epoch_begin(self, epoch, logs=None):
                if epoch == self.fine_tune_epoch:
                    print(f"\nEpoch {epoch}: Fine-tuning the model by unfreezing base layers...")
                    for layer in self.model.layers[0].layers:
                        layer.trainable = True
                    # Use a lower learning rate for fine-tuning
                    self.model.optimizer.learning_rate = learning_rate / 10
                    print(f"Learning rate reduced to {learning_rate / 10} for fine-tuning")
        
        callbacks_list.append(FineTuningCallback(fine_tune_after))
    else:
        # If not using fine-tuning, add regular early stopping
        callbacks_list.append(
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,  # Reduced from 10 to match PyTorch reference
                verbose=1,
                mode='min'
            )
        )
    
    # Save training config to output directory
    with open(os.path.join(output_dir, "training_config.txt"), "w") as f:
        f.write(f"Model type: {model_type}\n")
        f.write(f"Pretrained: {pretrained}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Augmentation level: {augmentation_level}\n")
        f.write(f"Initial learning rate: {learning_rate}\n")
        f.write(f"Fine-tuning epoch: {fine_tune_after if pretrained else 'N/A'}\n")
        f.write(f"Fonts directory: {fonts_dir}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Training date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Train the model
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Evaluate the model
        print("\nEvaluating model on validation set...")
        val_metrics = model.evaluate(validation_generator)
        metric_names = model.metrics_names
        
        print("\nValidation Results:")
        for name, value in zip(metric_names, val_metrics):
            print(f"{name}: {value:.4f}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to save the model if possible
        try:
            model.save(model_path)
            print(f"Partial model saved to {model_path}")
        except Exception as save_error:
            print(f"Could not save the partial model: {save_error}")
    
    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

def predict(image_path, model_path, fonts_dir=FONT_DIR, top_k=5):
    """
    Predict the font used in an image.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model file
        fonts_dir: Directory containing font files
        top_k: Number of top predictions to return
    """
    # Initialize reverse font mapping
    global REVERSE_FONT_MAPPING
    REVERSE_FONT_MAPPING = create_reverse_font_mapping(fonts_dir)
    
    if not REVERSE_FONT_MAPPING:
        raise ValueError(f"No fonts found in '{fonts_dir}'. Cannot proceed with prediction.")
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get top-k predictions
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    
    print(f"\nFont prediction for {image_path}:")
    print("-------------------------------")
    
    for i, idx in enumerate(top_indices):
        font_name = rev_conv_label(idx, fonts_dir)
        probability = predictions[0][idx] * 100  # Convert to percentage
        print(f"{i+1}. {font_name}: {probability:.2f}%")
    
    # Return top prediction
    top_prediction = rev_conv_label(np.argmax(predictions, axis=1)[0], fonts_dir)
    return top_prediction

# Main Function
def main():
    """
    Main function to handle command-line arguments and execute operations.
    """
    parser = argparse.ArgumentParser(description='ResNet Font Detection Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create dataset command
    create_parser = subparsers.add_parser('create_dataset', help='Create a dataset for training')
    create_parser.add_argument('--input', '-i', default=FONT_DIR, 
                              help=f'Input directory for fonts (default: {FONT_DIR})')
    create_parser.add_argument('--output', '-o', default='train_data', 
                              help='Output directory for dataset (default: train_data)')
    create_parser.add_argument('--count', '-c', type=int, default=500, 
                              help='Number of images per font (default: 500)')
    create_parser.add_argument('--seed', '-s', type=int, default=RANDOM_SEED,
                              help=f'Random seed for reproducible dataset generation (default: {RANDOM_SEED})')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the font detection model')
    train_parser.add_argument('--model-type', '-t', default='resnet50', 
                             choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                             help='Model type to train (default: resnet50)')
    train_parser.add_argument('--pretrained', '-p', action='store_true', 
                             help='Use pretrained weights (default: False)')
    train_parser.add_argument('--epochs', '-e', type=int, default=25, 
                             help='Number of training epochs (default: 25)')
    train_parser.add_argument('--batch-size', '-b', type=int, default=32, 
                             help='Batch size for training (default: 32)')
    train_parser.add_argument('--learning-rate', '-l', type=float, default=0.01,
                             help='Initial learning rate (default: 0.01)')
    train_parser.add_argument('--fine-tune-after', type=int, default=10,
                             help='Epoch after which to fine-tune pretrained models (default: 10)')
    train_parser.add_argument('--data', '-d', default='train_data', 
                             help='Path to the dataset directory (default: train_data)')
    train_parser.add_argument('--output-dir', '-o', default=OUTPUT_DIR,
                             help=f'Directory to save model and logs (default: {OUTPUT_DIR})')
    train_parser.add_argument('--model-name', '-m', default=MODEL_NAME,
                             help=f'Name of the model file to save (default: {MODEL_NAME})')
    train_parser.add_argument('--fonts-dir', '-f', default=FONT_DIR,
                             help=f'Directory containing font files for mapping (default: {FONT_DIR})')
    train_parser.add_argument('--augmentation', '-a', default='v3',
                             choices=['v1', 'v2', 'v3'],
                             help='Augmentation level (default: v3)')
    
    # Predict font command
    predict_parser = subparsers.add_parser('predict', help='Predict font from an image')
    predict_parser.add_argument('--image', '-i', required=True,
                               help='Path to the image file')
    predict_parser.add_argument('--model', '-m', required=True,
                               help='Path to the trained model file')
    predict_parser.add_argument('--fonts-dir', '-f', default=FONT_DIR,
                               help=f'Directory containing font files for mapping (default: {FONT_DIR})')
    predict_parser.add_argument('--top-k', '-k', type=int, default=5,
                               help='Number of top predictions to return (default: 5)')
    
    # Check GPU command
    gpu_parser = subparsers.add_parser('check_gpu', help='Check GPU availability for TensorFlow')
    
    args = parser.parse_args()
    
    if args.command == 'create_dataset':
        create_dataset(args.input, args.output, args.count, args.seed)
    elif args.command == 'train':
        train(
            model_type=args.model_type,
            pretrained=args.pretrained,
            batch_size=args.batch_size,
            epochs=args.epochs,
            data_path=args.data,
            output_dir=args.output_dir,
            model_name=args.model_name,
            fonts_dir=args.fonts_dir,
            augmentation_level=args.augmentation,
            learning_rate=args.learning_rate,
            fine_tune_after=args.fine_tune_after
        )
    elif args.command == 'predict':
        predict(args.image, args.model, args.fonts_dir, args.top_k)
    elif args.command == 'check_gpu':
        check_gpu()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

    # Example usage commands (these are comments - not executable code)
    # --------------------------------------------------------------
    # Create dataset examples:
    # python resnet_font_detector.py create_dataset
    # python resnet_font_detector.py create_dataset --input fonts --output train_data_full --count 1000
    # python resnet_font_detector.py create_dataset -i custom_fonts -o custom_dataset -c 200 -s 123
    #
    # Train model examples:
    # python resnet_font_detector.py train
    # python resnet_font_detector.py train --model-type resnet18 --pretrained --epochs 50 --batch-size 64
    # python resnet_font_detector.py train -t resnet34 -p -e 30 -b 32 -d custom_dataset -o runs/custom_run -m custom_model.keras
    # python resnet_font_detector.py train -t resnet50 -p -e 100 -b 16 -l 0.001 --fine-tune-after 20 --augmentation v2
    #
    # Prediction examples:
    # python resnet_font_detector.py predict --image test_image.jpg --model runs/experiment/resnet_font_model.keras
    # python resnet_font_detector.py predict -i sample.png -m custom_model.keras -f custom_fonts -k 3
    #
    # Check GPU example:
    # python resnet_font_detector.py check_gpu