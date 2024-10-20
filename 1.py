import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr

# Load the image file
img_path = r"lopo.jpg"

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Function to preprocess the image
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    return thresh

# Read the image using OpenCV
img = cv2.imread(img_path)

# Preprocess the image for better OCR results
preprocessed_img = preprocess_image(img)

# Perform OCR on the preprocessed image
ocr_results = reader.readtext(preprocessed_img)

# Print the OCR results for debugging
print("OCR Results:")
for item in ocr_results:
    print(item)

# Define target words to detect
target_words = ['fresh', 'abc', 'address']

# Draw rectangles around detected target words
for item in ocr_results:
    text = item[1]
    cord = item[0]

    # Check if the detected text (case insensitive) is one of the target words
    if any(word.lower() in text.lower() for word in target_words):
        # Extract minimum and maximum x and y coordinates
        xm, ym = [int(min(coord)) for coord in zip(*cord)]
        xma, yma = [int(max(coord)) for coord in zip(*cord)]

        # Draw rectangle around the detected text
        cv2.rectangle(img, (xm, ym), (xma, yma), (0, 255, 0), 2)

        # Write the found text above the rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (xm, ym - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

# Save the image with highlighted target words
output_image_path = r"C:\Users\Dell\OneDrive\Desktop\highlighted_result.jpg"
cv2.imwrite(output_image_path, img)
print(f"Result image saved at: {output_image_path}")

# Display the image with highlighted target words using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.title('Detected Target Words')
plt.show()
