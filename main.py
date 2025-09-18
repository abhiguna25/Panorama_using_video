#!/usr/bin/env python3
"""
better_panorama.py
Usage:
    python better_panorama.py img1.jpg img2.jpg [img3.jpg ...] -o panorama.jpg

Notes:
- Input images should be roughly left-to-right order (approximate).
- For best results use 30%+ overlap, similar exposure, and SIFT-capable OpenCV (opencv-contrib-python).
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
import os
import glob

# -----------------------
# Utilities / feature ops
# -----------------------
def make_detector(prefer_sift=True):
    """Return (detector, is_sift)"""
    if prefer_sift and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(), True
    # fallback ORB
    return cv2.ORB_create(5000), False

def match_descriptors(desc1, desc2, use_sift=True, ratio=0.75):
    """Match with FLANN (for SIFT) or BF+Hamming (for ORB). Returns list of good DMatch."""
    if desc1 is None or desc2 is None:
        return []
    if use_sift:
        # FLANN for SIFT (float descriptors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Ensure float32
        if desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)
        raw_matches = flann.knnMatch(desc1, desc2, k=2)
    else:
        # BF Hamming for ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m_n in raw_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

# -----------------------
# Homographies & transforms
# -----------------------
def compute_pairwise_homographies(kps, descs, use_sift, min_matches=8):
    """
    Compute homography H_i mapping image i -> image i-1 for i=1..n-1
    Returns list H_pairs where H_pairs[i] is H mapping img i to img i-1 (H_pairs[0] is None)
    """
    n = len(kps)
    H_pairs = [None] * n
    for i in range(1, n):
        matches = match_descriptors(descs[i], descs[i-1], use_sift)
        print(f"[pair {i-1} <-> {i}] good matches: {len(matches)}")
        if len(matches) < min_matches:
            print(f"  Warning: not enough matches between {i-1} and {i} (need {min_matches}).")
            H_pairs[i] = None
            continue
        src_pts = np.float32([kps[i][m.queryIdx].pt for m in matches]).reshape(-1,1,2)  # pts in img_i
        dst_pts = np.float32([kps[i-1][m.trainIdx].pt for m in matches]).reshape(-1,1,2) # pts in img_{i-1}
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            print(f"  Homography failed between {i-1} and {i}")
        H_pairs[i] = H
    return H_pairs

def compose_transforms_to_center(H_pairs, center_idx):
    """
    Given H_pairs (H_pairs[i] maps img i -> img i-1), produce H_to_center list
    where H_to_center[k] maps points in img k into the coordinate system of the center image.
    """
    n = len(H_pairs)
    # compute cumulative transforms to image 0: T[i] maps img i -> img 0
    T = [np.eye(3) for _ in range(n)]
    for i in range(1, n):
        if H_pairs[i] is None:
            T[i] = None
            continue
        if T[i-1] is None:
            # cannot chain
            T[i] = None
        else:
            T[i] = T[i-1] @ H_pairs[i]  # apply H_i then previous transforms
    # compute inverse of T[center]
    if T[center_idx] is None:
        raise RuntimeError("Cannot compute transforms: center transform is missing (insufficient matches).")
    T_center_inv = np.linalg.inv(T[center_idx])
    H_to_center = []
    for i in range(n):
        if T[i] is None:
            H_to_center.append(None)
        else:
            H_to_center.append(T_center_inv @ T[i])
    return H_to_center

# -----------------------
# Warping & blending
# -----------------------
def warp_images_and_prepare_canvas(images, H_to_center):
    """Compute output canvas size, translation, and warp each image into that canvas."""
    # collect all warped corners to determine canvas bounds
    all_corners = []
    sizes = []
    for img in images:
        h,w = img.shape[:2]
        sizes.append((w,h))
    for img_idx, (w,h) in enumerate(sizes):
        H = H_to_center[img_idx]
        if H is None:
            raise RuntimeError(f"Missing transform for image {img_idx}.")
        corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)
    all_corners = np.vstack(all_corners)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    width = xmax - xmin
    height = ymax - ymin
    translation = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float64)
    # Warp images
    warped_imgs = []
    warped_masks = []
    for idx, img in enumerate(images):
        H = translation @ H_to_center[idx]
        warped = cv2.warpPerspective(img, H, (width, height))
        # mask: non-black pixels
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8) * 255
        warped_imgs.append(warped)
        warped_masks.append(mask)
    return warped_imgs, warped_masks, (width, height), translation

def distance_weight_mask(mask):
    """Compute a per-pixel weight map (float32 0..1) for given binary mask using distance transform."""
    # mask expected as 0/255
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return np.zeros_like(bin_mask, dtype=np.float32)
    # distance to the nearest zero (i.e., inner regions get larger distance)
    inv = 1 - bin_mask
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    # Normalize by max distance within this mask
    maxd = dist.max()
    if maxd <= 0:
        return bin_mask.astype(np.float32)
    weight = dist / maxd
    # Slight boost inside regions (avoid exactly zero)
    weight = 0.01 + 0.99 * weight
    weight *= bin_mask
    return weight.astype(np.float32)

def blend_warped_images(warped_imgs, warped_masks):
    """Blend warped images using distance-transform weighting (accumulate weighted sum / weights)."""
    H, W = warped_imgs[0].shape[:2]
    acc = np.zeros((H, W, 3), dtype=np.float64)
    acc_weight = np.zeros((H, W), dtype=np.float64)

    for idx, (img, mask) in enumerate(zip(warped_imgs, warped_masks)):
        weight = distance_weight_mask(mask)  # float32 0..1
        # expand weight to 3 channels
        w3 = weight[..., None]
        acc += img.astype(np.float64) * w3
        acc_weight += weight.astype(np.float64)

    # avoid divide by zero
    nonzero = acc_weight > 1e-6
    result = np.zeros_like(acc, dtype=np.uint8)
    result[nonzero] = (acc[nonzero] / acc_weight[nonzero, None]).astype(np.uint8)
    return result

# -----------------------
# Main pipeline
# -----------------------
def create_panorama(image_paths, resize_to=None, debug=False):
    # load
    images = [cv2.imread(str(p)) for p in image_paths]
    if any(im is None for im in images):
        missing = [str(p) for p,im in zip(image_paths, images) if im is None]
        raise FileNotFoundError(f"Could not read images: {missing}")

    # optional resize to speed up (maintain aspect)
    if resize_to is not None:
        resized = []
        for im in images:
            h,w = im.shape[:2]
            scale = resize_to / max(h,w) if max(h,w) > resize_to else 1.0
            if scale != 1.0:
                resized.append(cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA))
            else:
                resized.append(im)
        images = resized

    # detect features
    detector, use_sift = make_detector(prefer_sift=True)
    print("Using detector:", "SIFT" if use_sift else "ORB (fallback)")

    keypoints = []
    descriptors = []
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = detector.detectAndCompute(gray, None)
        print(f"Image {idx}: {len(kp)} keypoints")
        keypoints.append(kp)
        descriptors.append(des)

    # compute pairwise homographies (adjacent images)
    H_pairs = compute_pairwise_homographies(keypoints, descriptors, use_sift)

    # choose center image (middle) to reduce drift
    center = len(images) // 2
    print("Center image index:", center)

    # compose transforms into center coordinate system
    try:
        H_to_center = compose_transforms_to_center(H_pairs, center)
    except RuntimeError as e:
        raise RuntimeError("Failed to compute transforms to center: " + str(e))

    # warp all images into common canvas
    warped_imgs, warped_masks, canvas_size, translation = warp_images_and_prepare_canvas(images, H_to_center)
    print("Canvas size:", canvas_size)

    # blend
    pano = blend_warped_images(warped_imgs, warped_masks)
    return pano

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build panorama from multiple images (SIFT + FLANN + RANSAC + blending)")
    p.add_argument("images", nargs="+", help="Input images in approximate left-to-right order")
    p.add_argument("-o", "--output", default="panorama_out.jpg", help="Output filename")
    p.add_argument("--resize", type=int, default=1600, help="Max dimension to resize images for speed (set 0 to disable)")
    return p.parse_args()


def video_to_frames(video_path, output_folder, prefix="frame"):
    # # Create output folder if it doesn't exist
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # # Open the video file
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print("Error: Could not open video.")
    #     return

    # frame_count = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:  # End of video
    #         break
    #     # Save frame as JPG
    #     filename = os.path.join(output_folder, f"{prefix}_{frame_count:05d}.jpg")
    #     cv2.imwrite(filename, frame)
    #     frame_count += 1

    # cap.release()
    # print(f"Done! Extracted {frame_count} frames to '{output_folder}'.")
    # Example usage
    # Create output folder if it doesn't exist
    frame_gap = 5
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Save every `frame_gap` frame
        if frame_count % frame_gap == 0:
            filename = os.path.join(output_folder, f"{prefix}_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done! Extracted {saved_count} frames (every {frame_gap}th) to '{output_folder}'.")
    
    
if __name__ == "__main__":
    video_path = "kailash.mp4"       # replace with your video file
    output_folder = "frames"       # folder to save images
    video_to_frames(video_path, output_folder)

    folder_path = Path("frames")
    
    # Collect all .jpg and .png files
    image_paths = sorted(
            [Path(p) for p in glob.glob(str(folder_path / "*.jpg"))] +
            [Path(p) for p in glob.glob(str(folder_path / "*.png"))]
        )
    if not image_paths:
        print(f"No images found in {folder_path}")
        sys.exit(1)

    #  set max dimension (None = no resize)
    maxdim = 1600  
    
    try:
        pano = create_panorama(image_paths, resize_to=maxdim)
        cv2.imwrite("panorama_out1.jpg", pano)
        print("Saved panorama to panorama_out1.jpg")
    except Exception as e:
        print("Panorama creation failed:", e)
        sys.exit(1)                             