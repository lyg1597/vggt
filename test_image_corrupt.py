import cv2
import numpy as np
import os
import argparse
import glob

def check_image_alignment(image, inlier_threshold=0.7, min_matches=10):
    """
    Checks for internal geometric consistency in an image by matching its halves.

    This is useful for detecting rolling shutter artifacts or other corruptions
    that cause misalignment within a single frame.

    Args:
        image (np.array): The input image (BGR).
        inlier_threshold (float): The minimum ratio of inlier matches required
                                  for the image to be considered 'good'.
        min_matches (int): The minimum number of matches required to perform
                           the homography check.

    Returns:
        bool: True if the image is likely good, False if it's likely corrupted.
    """
    # Convert to grayscale for feature detection
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape

    # 1. Split the image into top and bottom halves
    top_half = gray_img[0:h//2, :]
    bottom_half = gray_img[h//2:, :]

    # 2. Detect and match features between the two halves
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(top_half, None)
    kp2, des2 = orb.detectAndCompute(bottom_half, None)

    if des1 is None or des2 is None:
        return True # Not enough features to check, assume it's okay

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 3. Check if we have enough matches to find a reliable transformation
    if len(matches) < min_matches:
        # Not enough texture to find matches, so we can't determine corruption.
        # We'll assume it's good to avoid false positives on low-texture images.
        return True

    # 4. Estimate the homography using RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Note: The y-coordinates in the bottom half are relative to that crop,
    # so we don't need to add h//2. We are just checking for a consistent
    # geometric relationship.
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return False # Homography failed completely

    # 5. Decide based on the inlier ratio
    inlier_ratio = np.sum(mask) / len(mask)

    # A good image should have a high ratio of inliers
    print(inlier_ratio)
    return inlier_ratio > inlier_threshold

if __name__ == "__main__":
    img_fn = 'big_room_undistort_rename/frame_00374.jpg'
    img = cv2.imread(img_fn)
    res = check_image_alignment(img)
    print(res)