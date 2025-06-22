import os
import csv
import numpy as np
from skimage import io, measure, morphology
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import zipfile

# Define the function to process a single image with given threshold and min_size
def process_image(image_path, blue_threshold, min_size):
    image = io.imread(image_path)
    
    # Convert the image to the RGB channels
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    
    # Threshold for blue channel
    binary_image = (b > blue_threshold) & (r < 100) & (g < 100)
    
    # Remove small objects
    filtered_image = morphology.remove_small_objects(binary_image, min_size=min_size)
    
    # Label connected components
    label_image = measure.label(filtered_image)
    num_labels = label_image.max()
    
    return num_labels, image, filtered_image

# Streamlit UI components
st.title("Image Processing for Blue Cells Detection")

# Upload image
uploaded_file = st.file_uploader("Choose a .tif image", type="tif")
if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Set up sliders for threshold and minimum size
    blue_threshold = st.slider("Blue Channel Threshold", 0, 255, 100)
    min_size = st.slider("Minimum Object Size", 0, 1000, 100)
    
    # Process the image
    num_labels, image, filtered_image = process_image(uploaded_file, blue_threshold, min_size)
    
    # Display results
    st.write(f"Number of Blue Cells Detected: {num_labels}")
    
    # Show original and filtered images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title(f'Filtered Image - Blue Cells: {num_labels}')
    st.pyplot(fig)

# Batch processing option
st.subheader("Batch Processing")

# Upload a directory of images (as a zip file)
uploaded_zip = st.file_uploader("Upload Zip file with Images", type=["zip"])

if uploaded_zip is not None:
    # Unzip the uploaded file
    with zipfile.ZipFile(BytesIO(uploaded_zip.read())) as zip_ref:
        zip_ref.extractall("uploaded_images")
    
    # List extracted files
    image_files = [f for f in os.listdir("uploaded_images") if f.endswith('.tif')]
    
    if image_files:
        csv_output_path = st.text_input("CSV Output Path", "./BMDM_results.csv")
        
        if st.button("Process Images"):
            results = []
            for image_file in image_files:
                image_path = os.path.join("uploaded_images", image_file)
                num_labels, image, filtered_image = process_image(image_path, blue_threshold, min_size)
                results.append([image_file, num_labels])
                
                # Save processed images
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(image)
                ax[0].set_title('Original Image')
                ax[1].imshow(filtered_image, cmap='gray')
                ax[1].set_title(f'Filtered Image - Blue Cells: {num_labels}')
                fig_path = image_path.replace('.tif', '_processed.png')
                plt.savefig(fig_path)
                plt.close()

            # Save results to CSV
            with open(csv_output_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Image', 'Blue Cells Count'])
                writer.writerows(results)
            
            st.success(f"Batch processing completed. Results saved to {csv_output_path}")
