# from matplotlib.pyplot import imshow
# import matplotlib.cm as cm
# import matplotlib.pylab as plt
import PIL
import argparse
import numpy as np
import os
import glob
from tensorflow.keras.utils import img_to_array
from keras.models import load_model


# Create a reverse mapping of indices to font names
def create_reverse_font_mapping():
    font_files = sorted(glob.glob("fonts/*"))
    font_names = [os.path.splitext(os.path.basename(f))[0] for f in font_files]
    return {idx: font for idx, font in enumerate(font_names)}

# Global reverse font mapping
REVERSE_FONT_MAPPING = create_reverse_font_mapping()

def rev_conv_label(label):
    # Return the font name for the given index using the global reverse mapping
    return REVERSE_FONT_MAPPING.get(label, "Unknown")


def get_data(img_path):
    pil_im =PIL.Image.open(img_path).convert('L')
    pil_im=pil_im.resize((105,105))
    org_img = img_to_array(pil_im)
    data=[]
    data.append(org_img)
    data = np.asarray(data, dtype="float") / 255.0

    return data


def evaluate(img_path, model_file):
    data = get_data(img_path)

    model = load_model(model_file)
    predict_y = model.predict(data)
    
    # Get the top 5 predictions
    top_indices = np.argsort(predict_y[0])[-5:][::-1]  # Sort and get last 5 (highest) values, then reverse to get descending order
    
    print(f"\nFont prediction for {img_path}:")
    print("-------------------------------")
    
    for i, idx in enumerate(top_indices):
        font_name = rev_conv_label(idx)
        probability = predict_y[0][idx] * 100  # Convert to percentage
        print(f"{i+1}. {font_name}: {probability:.2f}%")
    
    # Also keep the original output for compatibility
    classes_y = np.argmax(predict_y, axis=1)
    top_font = rev_conv_label(classes_y[0])
    print(f"\nTop prediction: {top_font}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Put data creation parameters')
    parser.add_argument('--data','-d', required=True)
    parser.add_argument('--model','-m', required=True)
    
    args = parser.parse_args()
    evaluate(args.data, args.model)
