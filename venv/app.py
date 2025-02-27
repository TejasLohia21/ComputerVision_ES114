import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# Number of intensity levels
NO_OF_PIXELS = 256

# --- Helper Functions for Otsu Thresholding ---
def histogram(img, rows, cols):
    """Compute histogram for a grayscale image."""
    hist = np.zeros(NO_OF_PIXELS, dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            pix_value = img[i, j]
            hist[pix_value] += 1
    return hist

def weight(values, rows, cols):
    return np.sum(values) / (rows * cols)

def meanBackward(values):
    total = np.sum(values)
    mean_val = 0
    for a in range(len(values)):
        mean_val += a * values[a]
    return mean_val / total if total != 0 else 0

def meanForward(values, offset):
    total = np.sum(values)
    mean_val = 0
    for a in range(len(values)):
        intensity = a + offset
        mean_val += intensity * values[a]
    return mean_val / total if total != 0 else 0.0

def varianceBackward(values, x):
    total = np.sum(values)
    var_sum = 0
    for a in range(len(values)):
        var_sum += ((a - x) ** 2) * values[a]
    return var_sum / total if total != 0 else 0.0

def varianceForward(values, offset, x):
    total = np.sum(values)
    var_sum = 0
    for a in range(len(values)):
        intensity = a + offset
        var_sum += ((intensity - x) ** 2) * values[a]
    return var_sum / total if total != 0 else 0.0

def otsu_thresholding(image):
    """
    Perform Otsu thresholding on a grayscale image.
    Returns the threshold value and the binarized image.
    """
    rows, cols = image.shape
    hist = histogram(image, rows, cols)
    
    meanB, meanF = [], []
    weightB, weightF = [], []
    
    for a in range(NO_OF_PIXELS):
        lst1 = hist[:a]   # pixels below intensity 'a'
        lst2 = hist[a:]   # pixels from intensity 'a' onward
        meanB.append(meanBackward(lst1))
        meanF.append(meanForward(lst2, len(lst1)))
        weightB.append(weight(lst1, rows, cols))
        weightF.append(weight(lst2, rows, cols))
        
    varianceB, varianceF = [], []
    for a in range(NO_OF_PIXELS):
        lst1 = hist[:a]
        lst2 = hist[a:]
        varianceB.append(varianceBackward(lst1, meanB[a]))
        varianceF.append(varianceForward(lst2, len(lst1), meanF[a]))
        
    variance_matrix = np.zeros(NO_OF_PIXELS)
    for a in range(NO_OF_PIXELS):
        variance_matrix[a] = weightB[a] * varianceB[a] + weightF[a] * varianceF[a]
    
    # Exclude last element if needed (to match your original code)
    variance_matrix = variance_matrix[:-1]
    threshold = int(np.argmin(variance_matrix))
    
    # Apply the threshold to binarize the image
    thresh_image = np.copy(image)
    thresh_image[thresh_image >= threshold] = 255
    thresh_image[thresh_image < threshold] = 0
    
    return threshold, thresh_image

# --- Functions for Adding Gaussian Noise ---
def generate_gaussian_noise(shape, std):
    noise = np.random.normal(0, std, size=shape)
    return np.round(noise).astype(int)

def add_noise(image, stddev):
    rows, cols = image.shape
    noise_array = generate_gaussian_noise((rows, cols), stddev)
    noisy_img = image.astype(np.int32) + noise_array
    # Clip pixel values to be between 0 and 255
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

# --- Streamlit App ---
st.title("Otsu Thresholding Streamlit App")
st.write("Upload an image and apply Otsu thresholding. You can also optionally add Gaussian noise.")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Load image using PIL and convert to grayscale
    image = Image.open(uploaded_file)
    image_gray = np.array(image.convert('L'))
    
    st.subheader("Original Image (Grayscale)")
    st.image(image_gray, caption="Original Grayscale Image", use_column_width=True)
    
    # Optionally add Gaussian noise
    if st.checkbox("Add Gaussian Noise"):
        stddev = st.slider("Noise Standard Deviation", min_value=0, max_value=50, value=10)
        image_processed = add_noise(image_gray, stddev)
        st.subheader("Noisy Image")
        st.image(image_processed, caption="Image with Gaussian Noise", use_column_width=True)
    else:
        image_processed = image_gray
    
    # Apply Otsu thresholding
    threshold, thresh_image = otsu_thresholding(image_processed)
    st.write("Calculated Threshold Intensity Value:", threshold)
    
    st.subheader("Thresholded Image")
    st.image(thresh_image, caption="Otsu Thresholded Image", use_column_width=True)
    
    # Optionally display the histogram plot
    if st.checkbox("Show Histogram Plot"):
        rows, cols = image_processed.shape
        hist_values = histogram(image_processed, rows, cols)
        fig, ax = plt.subplots()
        ax.plot(hist_values)
        ax.set_title("Histogram of Image")
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Pixel Count")
        st.pyplot(fig)