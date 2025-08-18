#!/usr/bin/env python
"""
YOLO Object Detection Implementation for Images, Videos, and Webcams.

Usage:
    # Image Detection
    python yolo_improved.py --image images/baggage_claim.jpg --output output/baggage_claim.jpg

    # Video Detection
    python yolo_improved.py --video videos/car_chase.mp4 --output output/car_chase.avi

    # Webcam Detection (press 'q' to quit)
    python yolo_improved.py

    # Using GPU for faster processing
    python yolo_improved.py --video videos/car_chase.mp4 --gpu
"""

import numpy as np
import argparse
import time
import cv2
import os
from pathlib import Path
from typing import List, Tuple

def parse_arguments() -> dict:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Object Detection (Images, Videos, Webcam)")
    parser.add_argument("-i", "--image", type=str,
                        help="path to input image")
    parser.add_argument("-v", "--video", type=str,
                        help="path to input video")
    parser.add_argument("-o", "--output", type=str,
                        help="path to output image/video")
    parser.add_argument("-y", "--yolo", type=str, default="yolo-coco",
                        help="base path to YOLO directory")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.3,
                        help="threshold when applying non-maxima suppression")
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="boolean flag to use GPU")
    return vars(parser.parse_args())

def load_yolo_model(yolo_path: Path, use_gpu: bool) -> cv2.dnn.Net:
    """Load YOLO model from disk and configure for CPU or GPU."""
    config_path = yolo_path / "yolov3.cfg"
    weights_path = yolo_path / "yolov3.weights"

    print("[INFO] Loading YOLO from disk...")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))

    if use_gpu:
        print("[INFO] Setting preferable backend and target to CUDA...")
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except cv2.error as e:
            print(f"[ERROR] Failed to set CUDA backend: {e}")
            print("[INFO] Falling back to CPU. Ensure you have OpenCV compiled with CUDA support.")

    return net

def get_output_layer_names(net: cv2.dnn.Net) -> List[str]:
    """Get the names of the output layers from the network."""
    layer_names = net.getLayerNames()
    # The new function getUnconnectedOutLayers() returns indices which may not be contiguous.
    # We need to use these indices to get the actual layer names.
    try:
        # For OpenCV 4.x and later
        out_layer_indices = net.getUnconnectedOutLayers()
        return [layer_names[i - 1] for i in out_layer_indices.flatten()]
    except AttributeError:
        # For older OpenCV versions
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def process_frame(frame: np.ndarray, net: cv2.dnn.Net, ln: List[str], labels: List[str], colors: np.ndarray, conf_thresh: float, nms_thresh: float) -> np.ndarray:
    """Process a single frame for object detection."""
    (H, W) = frame.shape[:2]

    # Construct a blob from the input frame and then perform a forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    # Initialize lists of detected bounding boxes, confidences, and class IDs
    boxes, confidences, class_ids = [], [], []

    # Loop over each of the layer outputs
    for output in layer_outputs:
        # Loop over each of the detections
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_thresh:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    # Draw detections if any exist
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def main():
    args = parse_arguments()
    yolo_path = Path(args["yolo"])

    # Load COCO class labels
    labels_path = yolo_path / "coco.names"
    try:
        LABELS = open(labels_path).read().strip().split("\n")
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Initialize colors for each class
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # Load YOLO model
    net = load_yolo_model(yolo_path, args["gpu"])
    ln = get_output_layer_names(net)

    # Determine input source
    if args.get("image"):
        print(f"[INFO] Processing image: {args['image']}")
        image = cv2.imread(args["image"])
        if image is None:
            raise ValueError(f"Could not read image: {args['image']}")
        
        start_time = time.time()
        processed_image = process_frame(image, net, ln, LABELS, COLORS, args["confidence"], args["threshold"])
        end_time = time.time()
        print(f"[INFO] Image processing took {end_time - start_time:.6f} seconds")

        # Save or show the output
        if args.get("output"):
            cv2.imwrite(args["output"], processed_image)
            print(f"[INFO] Saved output image to {args['output']}")
        else:
            cv2.imshow("Object Detection", processed_image)
            cv2.waitKey(0)

    elif args.get("video"):
        vs = cv2.VideoCapture(args["video"])
        print(f"[INFO] Processing video: {args['video']}")
    else:
        print("[INFO] Starting webcam feed...")
        vs = cv2.VideoCapture(0) # Use 0 for default webcam

    writer = None
    if args.get("video") or not args.get("image"): # This condition handles video and webcam
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break

            start_time = time.time()
            processed_frame = process_frame(frame, net, ln, LABELS, COLORS, args["confidence"], args["threshold"])
            end_time = time.time()
            
            fps = 1 / (end_time - start_time)
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Check if the video writer is None
            if writer is None and args.get("output"):
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
                print(f"[INFO] Saving output video to {args['output']}")
            
            if writer is not None:
                writer.write(processed_frame)

            cv2.imshow("Object Detection", processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        
        # Release file pointers
        print("[INFO] Cleaning up...")
        if writer is not None:
            writer.release()
        vs.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()