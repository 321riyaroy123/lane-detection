import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import matplotlib.image as mpimg
import glob2
import math
from moviepy.editor import VideoFileClip

def blur_image(image):
    return(cv2.blur(image,(5,5)))
def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussianblur_image(image):
    return cv2.GaussianBlur(image, (7, 7), 0)

def canny_image(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

def yellow_white_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gblur = cv2.GaussianBlur(gray, (5, 5), 0)
    white_mask = cv2.threshold(gblur, 200, 255, cv2.THRESH_BINARY)[1]
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([210, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    white_mask = white_mask.astype(np.uint8)
    yellow_mask = yellow_mask.astype(np.uint8)
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return combined_mask

def region_of_interest(img, vertices):
    """
    Apply a mask to keep only the region of interest.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=1):    
    left_lines = []
    right_lines = []
    top_y = 1e6

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 != x2:  # Avoid vertical lines (infinite slope)
                    slope = (y2 - y1) / (x2 - x1)
                    if slope > 0:  # Left line
                        left_lines.append([x1, y1, x2, y2])
                        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                    else:  # Right line
                        right_lines.append([x1, y1, x2, y2])
                        cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], thickness)

                if top_y > y1:
                    top_y = y1
                if top_y > y2:
                    top_y = y2

    # Get the average position of each line and ensure valid slope
    if len(left_lines) > 0:
        left_line = np.mean(left_lines, axis=0)
        if left_line[2] - left_line[0] != 0:  # Avoid division by zero
            slope = (left_line[3] - left_line[1]) / (left_line[2] - left_line[0])
            top_x = left_line[0] + (top_y - left_line[1]) / slope
            bottom_x = left_line[0] + (img.shape[0] - left_line[1]) / slope
            if not math.isinf(bottom_x) and not math.isinf(top_x):  # Check for infinity
                cv2.line(img, (int(bottom_x), img.shape[0]), (int(top_x), int(top_y)), color, thickness * 10)

    if len(right_lines) > 0:
        right_line = np.mean(right_lines, axis=0)
        if right_line[2] - right_line[0] != 0:  # Avoid division by zero
            slope = (right_line[3] - right_line[1]) / (right_line[2] - right_line[0])
            top_x = right_line[0] + (top_y - right_line[1]) / slope
            bottom_x = right_line[0] + (img.shape[0] - right_line[1]) / slope
            if not math.isinf(bottom_x) and not math.isinf(top_x):  # Check for infinity
                cv2.line(img, (int(bottom_x), img.shape[0]), (int(top_x), int(top_y)), color, thickness * 10)
def filter_lines(lines):
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate the slope
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                # Filter based on slope and length
                if -1.5 < slope < 1.5:  # Adjusted slope range to be less strict
                    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if length > 30:  # Minimum line length
                        filtered_lines.append(line)
    return filtered_lines

def hough_transform(image):
    rho = 1
    theta = np.pi / 180
    threshold = 50
    min_line_len = 50
    max_line_gap = 150
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)


def fit_polynomial(positions):
    y_positions = positions[:, 0]
    x_positions = positions[:, 1]

    fit = np.polyfit(y_positions, x_positions, 2)  # Use degree 2 for curvature
    return fit

def draw_polynomial(img, fit):
    y_vals = np.linspace(0, img.shape[0] - 1, num=img.shape[0])
    x_vals = fit[0] * y_vals ** 2 + fit[1] * y_vals + fit[2]
    for x, y in zip(x_vals.astype(int), y_vals.astype(int)):
        if 0 <= x < img.shape[1]:
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # Draw fitted polynomial

def weighted_image(rgb_line_image,image,a=0.8,b=1,l=0):
    return cv2.addWeighted(rgb_line_image,a,image,b,l)


def draw_lane_line(image):
    # Preprocess the image to create a mask for yellow and white lines
    combined_mask = yellow_white_mask(image)
    blurred_mask = gaussianblur_image(combined_mask)

    # Specify low and high thresholds for Canny edge detection
    edges_image = canny_image(blurred_mask, 100, 200)  # Adjust the low and high thresholds

    # Define the vertices for the ROI
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (imshape[1] // 2, imshape[0] // 2), (imshape[1], imshape[0])]], dtype=np.int32)
    roi_image = region_of_interest(edges_image, vertices)

    # Detect lines using Hough Transform
    lines = hough_transform(roi_image)

    # Create a blank image to draw lines
    line_image = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)  # Ensure it's uint8

    # Draw lines on the line_image
    draw_lines(line_image, lines, color=(255, 0, 0))  # Draw lines in red

    # Ensure both images are of type uint8
    image_uint8 = image.astype(np.uint8)
    line_image_uint8 = line_image.astype(np.uint8)

    # Overlay lines on the original image
    final_image = cv2.addWeighted(image_uint8, 0.8, line_image_uint8, 1, 0)

    return final_image

def weighted_image(rgb_line_image, image, a=0.8, b=1, l=0):
    # Ensure that both images are in the same format
    if rgb_line_image.shape[:2] != image.shape[:2]:
        rgb_line_image = cv2.resize(rgb_line_image, (image.shape[1], image.shape[0]))

    # Convert the original image to 3 channels if it's grayscale
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Perform the blending
    return cv2.addWeighted(rgb_line_image, a, image, b, l)

def read_image(image_path):
    """Reads and returns image in RGB format."""
    img = mpimg.imread(image_path)
    if img.shape[2] == 4:  # Check if the image has 4 channels (RGBA)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # Convert to RGB
    return img

import os
# Load the test images
test_images = [
    read_image(os.path.join(r'D:\RIYA\LaneDetection\TestImage', i))
    for i in os.listdir(r'D:\RIYA\LaneDetection\TestImage')
]

# Process and display the images
for i in range(len(test_images)):
    print(f"Image {i} shape:", test_images[i].shape)
    plt.imshow(draw_lane_line(test_images[i]))

from moviepy.editor import VideoFileClip
def process_image(image):
    """Processes image through the lane detection pipeline and displays it."""
    result = draw_lane_line(image)

    # Convert the result to BGR format (OpenCV uses BGR, while matplotlib uses RGB)
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # Display the frame using OpenCV
    cv2.imshow('Lane Detection', result_bgr)

    # Add a delay to control the frame rate and allow quitting with 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        return None

    return result

# Load the video file
clip1 = VideoFileClip(r"D:\RIYA\LaneDetection\TestVideos\challenge.mp4")

# Loop through the video, process each frame, and display it
for frame in clip1.iter_frames(fps=clip1.fps, dtype='uint8'):
    result = process_image(frame)
    
    # Break the loop if the user presses 'q'
    if result is None:
        break

# Close all OpenCV windows after the video ends
cv2.destroyAllWindows()

