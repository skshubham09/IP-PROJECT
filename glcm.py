import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import moment, skew, kurtosis
from skimage.color import rgb2gray
from scipy.stats import entropy
import matplotlib.pyplot as plt
from pre import process_image

def compute_glcm_features(gray_image):
    # Convert image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute GLCM
    distances = [1]
    angles = [0]
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Calculate GLCM features
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    
    # Additional GLCM features
    cluster_prominence_value, cluster_shade_value, sum_variance_value = compute_additional_glcm_features(glcm)
    
    return contrast, correlation, energy, homogeneity, dissimilarity, cluster_prominence_value, cluster_shade_value, sum_variance_value

def intensity_histogram_analysis(gray_image):
    # Convert image to grayscale
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    
    # Compute mean intensity
    mean_intensity = np.mean(gray_image)
    
    # Compute variance
    variance = np.var(gray_image)
    
    # Compute skewness
    skewness = skew(hist.ravel())
    
    # Compute kurtosis
    kurtosis_value = kurtosis(hist.ravel())
    
    # Compute entropy
    hist_normalized = hist.ravel() / hist.sum()
    entropy_value = entropy(hist_normalized, base=2)
    
    return mean_intensity, variance, skewness, kurtosis_value, entropy_value, hist

# Function to compute additional GLCM features
def compute_additional_glcm_features(glcm):
    normalized_glcm = normalize_glcm(glcm)
    cluster_prominence_value = cluster_prominence(normalized_glcm)
    cluster_shade_value = cluster_shade(normalized_glcm)
    sum_variance_value = sum_variance(normalized_glcm)
    
    return cluster_prominence_value, cluster_shade_value, sum_variance_value

def normalize_glcm(glcm):
    return glcm 

# Function to compute cluster prominence
def cluster_prominence(glcm):
    cp = 0
    ux, uy = np.mean(glcm, axis=0), np.mean(glcm, axis=1)
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            cp += ((i + j - ux[i] - uy[j]) ** 4) * glcm[i, j]
    return cp

# Function to compute cluster shade
def cluster_shade(glcm):
    cs = 0
    ux, uy = np.mean(glcm, axis=0), np.mean(glcm, axis=1)
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            cs += ((i + j - ux[i] - uy[j]) ** 3) * glcm[i, j]
    return cs

# Function to compute sum variance
def sum_variance(glcm):
    sv = 0
    ux, uy = np.mean(glcm, axis=0), np.mean(glcm, axis=1)
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            sv += ((i - uy[i]) ** 2 + (j - ux[j]) ** 2) * glcm[i, j]
    return sv

# Load and process the image

#image_path = 'kidney_image.jpg'
image_path = 'cyst.png'
#image_path = 'stone.png'
#image_path = 'bi.png'
#image_path = 'stone2.png'
processed_image = process_image(image_path)

# Perform intensity histogram analysis
mean_intensity, variance, skewness, kurtosis, entropy_value, hist = intensity_histogram_analysis(processed_image)

# Calculate GLCM features
contrast, correlation, energy, homogeneity, dissimilarity, cluster_prominence_value, cluster_shade_value, sum_variance_value = compute_glcm_features(processed_image)

# Limit values to 4 digits after the decimal point
mean_intensity = round(mean_intensity, 4)
variance = round(variance/1e7, 4)
skewness = round(skewness/3, 4)
kurtosis = round(kurtosis/5, 4)
entropy_value = round(entropy_value, 4)
contrast = round(contrast, 4)
correlation = round(correlation, 4)
energy = round(energy, 4)
homogeneity = round(homogeneity, 4)
dissimilarity = round(dissimilarity, 4)
cluster_prominence_value = round(cluster_prominence_value.item()/3000000, 4)
cluster_shade_value = round(cluster_shade_value.item()/100000, 4)
sum_variance_value = round(sum_variance_value.item()/500, 4)


# Display the computed statistical properties
print("Mean:", mean_intensity)
print("Variance:", variance)
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)
print("Entropy:", entropy_value)

# Display GLCM features
print("Contrast:", contrast)
print("Correlation:", correlation)
print("Energy:", energy)
print("Homogeneity:", homogeneity)
print("Dissimilarity:", dissimilarity)
print("Cluster Prominence:", cluster_prominence_value)
print("Cluster Shade:", cluster_shade_value)
print("Sum Variance:", sum_variance_value)

# Plotting intensity histogram
plt.figure(figsize=(8, 6))
plt.plot(hist)
plt.title('Intensity Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Plotting intensity histogram elements
elements = ['Mean Intensity', 'Variance', 'Skewness', 'Kurtosis', 'Entropy']
values = [mean_intensity, variance, skewness, kurtosis, entropy_value]
plt.figure(figsize=(8, 6))
plt.bar(elements, values)
plt.title('Intensity Histogram Elements')
plt.ylabel('Value')
plt.show()

# Plotting GLCM features
glcm_features = ['Contrast', 'Correlation', 'Energy', 'Homogeneity', 'Dissimilarity', 'Cluster Prominence', 'Cluster Shade', 'Sum Variance']
glcm_values = [contrast, correlation, energy, homogeneity, dissimilarity, cluster_prominence_value, cluster_shade_value, sum_variance_value]
plt.figure(figsize=(8, 6))
plt.bar(glcm_features, glcm_values)
plt.title('GLCM Features')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()
