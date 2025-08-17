#!/usr/bin/env python
"""
YOLO Object Detection Implementation

Usage:
    python yolo.py --image images/baggage_claim.jpg [--confidence 0.5] [--threshold 0.3]
"""

import numpy as np
import argparse
import time
import cv2
import os
from pathlib import Path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("-i", "--image", required=True,
                        help="path to input image")
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.3,
                        help="threshold when applying non-maxima suppression")
    return vars(parser.parse_args())

def load_yolo_model(config_path, weights_path):
    """Load YOLO model from disk."""
    print("[INFO] loading YOLO from disk...")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    return net

def get_output_layers(net):
    """Get the output layer names from the network."""
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def process_detections(outputs, image_shape, confidence_threshold):
    """Process YOLO outputs and return filtered detections."""
    boxes = []
    confidences = []
    class_ids = []
    height, width = image_shape
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, box_width, box_height) = box.astype("int")
                
                x = int(center_x - (box_width / 2))
                y = int(center_y - (box_height / 2))
                
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    return boxes, confidences, class_ids

def draw_detections(image, boxes, confidences, class_ids, idxs, labels, colors):
    """Draw bounding boxes and labels on the image."""
    for i in idxs.flatten():
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.4f}"
        cv2.putText(image, text, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Verify input image exists
    if not os.path.exists(args["image"]):
        raise FileNotFoundError(f"Input image not found: {args['image']}")
    
    # Load COCO class labels
    labels_path = Path('yolo-coco') / 'coco.names'
    try:
        LABELS = open(labels_path).read().strip().split("\n")
    except FileNotFoundError:
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Initialize colors for each class
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    
    # Load YOLO model
    weights_path = Path('yolo-coco') / 'yolov3.weights'
    config_path = Path('yolo-coco') / 'yolov3.cfg'
    net = load_yolo_model(config_path, weights_path)
    
    # Load input image
    image = cv2.imread(args["image"])
    if image is None:
        raise ValueError(f"Could not read image: {args['image']}")
    H, W = image.shape[:2]
    
    # Get output layer names
    ln = get_output_layers(net)
    
    # Perform forward pass
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), 
                                swapRB=True, crop=False)
    net.setInput(blob)
    
    start_time = time.time()
    layer_outputs = net.forward(ln)
    inference_time = time.time() - start_time
    
    print(f"[INFO] YOLO took {inference_time:.6f} seconds")
    
    # Process detections
    boxes, confidences, class_ids = process_detections(
        layer_outputs, (H, W), args["confidence"])
    
    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 
                           args["confidence"], args["threshold"])
    
    # Draw detections if any exist
    if len(idxs) > 0:
        draw_detections(image, boxes, confidences, class_ids, idxs, LABELS, COLORS)
    
    # Display results
    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()