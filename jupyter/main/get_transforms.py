import numpy as np
import pandas as pd
from pathlib import Path

from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp

from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation as pcc

def main(original_images):
    print("Looping through all images")
    registered_images, transformations, pairwise_transformations, results_df = register_all_images(
        original_images,
        reference="002",
        max_distance=20
    )
    # display(results_df)

    ref_img = registered_images["002"]
    
    shifts = {}
    aligned_images = {}
    
    for name, img in registered_images.items():
    
        # Calculate translation relative to reference image
        s = pcc(
            ref_img,
            img,
            normalization="phase",
            upsample_factor=100
        )[0]
    
        # Save translation (dy, dx)
        shifts[name] = s
    
        # Apply translation to the entire image
        aligned_images[name] = shift(img, s)
    
        # print(f"{name}: dy = {s[0]:.2f}, dx = {s[1]:.2f}")

    return registered_images, transformations, shifts


def register_image(ref_img, img, max_distance=20):
    # STEP 1: Find SIFT features in both images
    # Lower c_dog makes SIFT detect weaker contrast features.
    sift1 = SIFT(
        n_octaves=5,
        n_scales=4,
        c_dog=0.01,
        c_edge=15
    )
    sift1.detect_and_extract(ref_img)

    sift2 = SIFT(
        n_octaves=5,
        n_scales=4,
        c_dog=0.01,
        c_edge=15
    )
    sift2.detect_and_extract(img)

    # STEP 2: Match corresponding SIFT features
    matches = match_descriptors(
        sift1.descriptors,
        sift2.descriptors,
        cross_check=True,
        max_ratio=0.8
    )

    n_matches = len(matches)

    # STEP 3: Filter matches based on distance
    # Assumes the images are already approximately aligned.
    pts1 = sift1.keypoints[matches[:, 0]]
    pts2 = sift2.keypoints[matches[:, 1]]

    distances = np.linalg.norm(pts1 - pts2, axis=1)
    good_matches = matches[distances < max_distance]

    n_filtered = len(good_matches)

    # At least 3 non-collinear point pairs are needed
    # to estimate an affine transformation.
    if n_filtered < 3:
        raise ValueError("Not enough matches for affine transformation")

    # STEP 4: Convert coordinates from (row, col) to (x, y)
    src = sift1.keypoints[good_matches[:, 0]][:, ::-1]
    dst = sift2.keypoints[good_matches[:, 1]][:, ::-1]

    # STEP 5: Estimate affine transformation using RANSAC
    # RANSAC rejects matches that do not follow the same transformation.
    transform, inliers = ransac(
        (src, dst),
        AffineTransform,
        min_samples=3,
        residual_threshold=2,
        max_trials=1000,
        rng=0
    )

    if transform is None or np.sum(inliers) < 3:
        raise ValueError("RANSAC failed to find a valid transformation")

    n_inliers = int(np.sum(inliers))

    # STEP 6: Calculate RMS registration error
    # Compare transformed reference points with their matched positions.
    predicted = transform(src[inliers])

    errors = np.linalg.norm(predicted - dst[inliers], axis=1)
    rms_error = np.sqrt(np.mean(errors**2))

    # STEP 7: Apply affine transformation
    # Transform maps reference -> moving image coordinates.
    # warp uses this mapping to sample the moving image on
    # the reference image's pixel grid.
    img_corrected = warp(
        img,
        inverse_map=transform,
        output_shape=ref_img.shape,
        preserve_range=True
    )

    return {
        "image": img_corrected,
        "transform": transform,
        "n_matches": n_matches,
        "n_filtered": n_filtered,
        "n_inliers": n_inliers,
        "rms_error": rms_error
    }

#Normalisering sørger for en felles intensitetsskala når karakteristiske punkter skal identifiseres --> klargjør bildene til affintransformasjon
def normalize_image(image):
    # Convert image to float
    image = np.asarray(image, dtype=float)

    # Normalize intensity values to [0, 1]
    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        raise ValueError("Image has constant intensity")

    return (image - min_val) / (max_val - min_val)

def register_all_images(images, reference="002", max_distance=20):

    # Sort images in numerical order
    names = sorted(images.keys(), key=int)
    names = names[names.index(reference):]

    # Normalize all images
    normalized_images = {
        name: normalize_image(images[name].data)
        for name in names
    }

    ref_img = normalized_images[reference]

    # Dictionaries for registered images and transformations
    registered_images = {}
    transformations = {}
    pairwise_transformations = {}

    # Store registration statistics
    results = []

    # Reference image needs no transformation
    registered_images[reference] = ref_img

    # Identity matrix for reference image
    cumulative_matrix = np.eye(3)
    transformations[reference] = AffineTransform(
        matrix=cumulative_matrix.copy()
    )

    results.append({
        "image": reference,
        "matches": np.nan,
        "filtered": np.nan,
        "inliers": np.nan,
        "rms_error": 0.0,
        "status": "reference"
    })

    previous_name = reference

    for name in names[1:]:

        # print(f"Registering {name} to {previous_name}...")

        # Previous and current ORIGINAL normalized images
        previous_img = normalized_images[previous_name]
        img = normalized_images[name]

        # STEP 1: SIFT + RANSAC between neighboring images
        result = register_image(
            previous_img,
            img,
            max_distance=max_distance
        )

        # STEP 2: Store pairwise affine transformation
        pairwise_transform = result["transform"]
        pairwise_transformations[name] = pairwise_transform

        # STEP 3: Combine with all previous transformations
        cumulative_matrix = (
            pairwise_transform.params @ cumulative_matrix
        )

        cumulative_transform = AffineTransform(
            matrix=cumulative_matrix.copy()
        )

        transformations[name] = cumulative_transform

        # STEP 4: Transform original image directly to reference
        registered_images[name] = warp(
            img,
            inverse_map=cumulative_transform,
            output_shape=ref_img.shape,
            preserve_range=True
        )

        # STEP 5: Store statistics
        results.append({
            "image": name,
            "matches": result["n_matches"],
            "filtered": result["n_filtered"],
            "inliers": result["n_inliers"],
            "rms_error": result["rms_error"],
            "status": "registered"
        })

        # print(
        #     f"  Inliers: {result['n_inliers']}, "
        #     f"RMS: {result['rms_error']:.3f} pixels"
        # )

        previous_name = name

    results_df = pd.DataFrame(results)

    return (
        registered_images,
        transformations,
        pairwise_transformations,
        results_df
    )