# Advanced Image Processing Pipeline

This branch adds a camera-independent image-processing pipeline. RealSense depth, coordinate calculation and calibration are intentionally not included.

## Pipeline

RGB/BGR
-> Gaussian smoothing
-> HSV
-> adaptive foreground mask (saturation/value)
-> morphological opening
-> morphological closing
-> hole filling
-> connected external components
-> area filtering
-> contour geometry
-> color classification from eroded interior pixels
-> shape classification using polygon approximation + circularity + solidity + extent
-> visualization

## Run

Install:

    pip install -r requirements.txt

Webcam:

    python main_advanced.py --source 0 --show-mask

Image:

    python main_advanced.py --source path/to/image.jpg --show-mask

Video:

    python main_advanced.py --source path/to/video.mp4 --show-mask

## Important parameters

If small noise is detected, increase:

    --min-area 1800

If colored objects are missed because their saturation is low, reduce:

    --min-sat 35

The detector is deliberately separated from RealSense. Once the segmentation/recognition quality is stable, the existing D435i acquisition layer can call detect_objects(frame) without changing the image algorithms.
