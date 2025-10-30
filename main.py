"""
Enhanced YOLO Object Detection Implementation for Images, Videos, and Webcams.

Features:
- Support for multiple YOLO versions (v3, v4, v4-tiny)
- Real-time FPS counter and performance metrics
- Configurable input/output resolutions
- Better error handling and logging
- Batch processing for images
- Progress tracking for videos

Usage:
    # Image Detection
    python yolo_enhanced.py --image images/baggage_claim.jpg --output output/baggage_claim.jpg

    # Video Detection
    python yolo_enhanced.py --video videos/car_chase.mp4 --output output/car_chase.avi

    # Webcam Detection (press 'q' to quit)
    python yolo_enhanced.py

    # Using GPU for faster processing
    python yolo_enhanced.py --video videos/car_chase.mp4 --gpu

    # Using YOLOv4-tiny for faster inference
    python yolo_enhanced.py --video videos/car_chase.mp4 --model yolov4-tiny
"""

import numpy as np
import argparse
import time
import cv2
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
import json

@dataclass
class DetectionResult:
    """Data class to store detection results."""
    frame: np.ndarray
    boxes: List[List[int]]
    confidences: List[float]
    class_ids: List[int]
    class_names: List[str]
    processing_time: float

class YOLODetector:
    """YOLO Object Detector class with enhanced functionality."""
    
    def __init__(self, yolo_path: Path, model_type: str = "yolov3", use_gpu: bool = False):
        self.yolo_path = yolo_path
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.net = None
        self.ln = None
        self.labels = []
        self.colors = []
        self.input_width = 416
        self.input_height = 416
        
        self._setup_logging()
        self._load_model()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_model_paths(self) -> Tuple[Path, Path]:
        """Get model configuration and weights paths based on model type."""
        model_configs = {
            "yolov3": ("yolov3.cfg", "yolov3.weights"),
            "yolov3-tiny": ("yolov3-tiny.cfg", "yolov3-tiny.weights"),
            "yolov4": ("yolov4.cfg", "yolov4.weights"),
            "yolov4-tiny": ("yolov4-tiny.cfg", "yolov4-tiny.weights"),
        }
        
        if self.model_type not in model_configs:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Available: {list(model_configs.keys())}")
        
        cfg_file, weights_file = model_configs[self.model_type]
        config_path = self.yolo_path / cfg_file
        weights_path = self.yolo_path / weights_file
        
        return config_path, weights_path
    
    def _load_model(self):
        """Load YOLO model and configure for CPU/GPU."""
        config_path, weights_path = self._get_model_paths()
        
        # Validate model files exist
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        self.logger.info(f"Loading {self.model_type} model from disk...")
        
        # Load model
        self.net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))
        
        # Configure GPU if requested and available
        if self.use_gpu:
            self._configure_gpu()
        else:
            self.logger.info("Using CPU for inference")
        
        # Get output layer names
        self.ln = self._get_output_layer_names()
        
        # Load class labels
        self._load_labels()
        
        # Generate colors for classes
        self._generate_colors()
    
    def _configure_gpu(self):
        """Configure the model to use GPU if available."""
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.logger.info("Successfully configured CUDA backend for GPU acceleration")
        except Exception as e:
            self.logger.warning(f"Failed to configure GPU: {e}. Falling back to CPU.")
            self.use_gpu = False
    
    def _get_output_layer_names(self) -> List[str]:
        """Get the names of the output layers from the network."""
        layer_names = self.net.getLayerNames()
        try:
            # OpenCV 4.x
            out_layer_indices = self.net.getUnconnectedOutLayers()
            return [layer_names[i - 1] for i in out_layer_indices.flatten()]
        except (AttributeError, TypeError):
            # OpenCV 3.x or different format
            return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def _load_labels(self):
        """Load COCO class labels."""
        labels_path = self.yolo_path / "coco.names"
        try:
            self.labels = open(labels_path).read().strip().split("\n")
            self.logger.info(f"Loaded {len(self.labels)} class labels")
        except FileNotFoundError:
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    def _generate_colors(self):
        """Generate distinct colors for each class."""
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
    
    def set_input_size(self, width: int, height: int):
        """Set the input size for the model."""
        self.input_width = width
        self.input_height = height
        self.logger.info(f"Input size set to {width}x{height}")
    
    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5, 
               nms_threshold: float = 0.3) -> DetectionResult:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            
        Returns:
            DetectionResult object containing detection results
        """
        start_time = time.time()
        (H, W) = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (self.input_width, self.input_height), 
            swapRB=True, crop=False
        )
        
        # Forward pass
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)
        
        # Process detections
        boxes, confidences, class_ids = self._process_detections(
            layer_outputs, W, H, confidence_threshold
        )
        
        # Apply non-maximum suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        # Filter results and get class names
        final_boxes, final_confidences, final_class_ids, class_names = \
            self._filter_detections(idxs, boxes, confidences, class_ids)
        
        # Draw detections on frame
        output_frame = self._draw_detections(
            frame.copy(), final_boxes, final_confidences, final_class_ids, class_names
        )
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            frame=output_frame,
            boxes=final_boxes,
            confidences=final_confidences,
            class_ids=final_class_ids,
            class_names=class_names,
            processing_time=processing_time
        )
    
    def _process_detections(self, layer_outputs: List[np.ndarray], W: int, H: int, 
                           confidence_threshold: float) -> Tuple[List, List, List]:
        """Process layer outputs to extract detections."""
        boxes, confidences, class_ids = [], [], []
        
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confidences, class_ids
    
    def _filter_detections(self, idxs: List, boxes: List, confidences: List, 
                          class_ids: List) -> Tuple[List, List, List, List]:
        """Filter detections using NMS results."""
        if len(idxs) == 0:
            return [], [], [], []
        
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        class_names = []
        
        for i in idxs.flatten():
            final_boxes.append(boxes[i])
            final_confidences.append(confidences[i])
            final_class_ids.append(class_ids[i])
            class_names.append(self.labels[class_ids[i]])
        
        return final_boxes, final_confidences, final_class_ids, class_names
    
    def _draw_detections(self, frame: np.ndarray, boxes: List, confidences: List, 
                        class_ids: List, class_names: List) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        for (box, confidence, class_id, class_name) in zip(boxes, confidences, class_ids, class_names):
            (x, y, w, h) = box
            color = [int(c) for c in self.colors[class_id]]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw text background
            cv2.rectangle(
                frame, (x, y - text_height - baseline - 5),
                (x + text_width, y), color, -1
            )
            
            # Draw text
            cv2.putText(
                frame, label, (x, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
        
        return frame

class PerformanceMetrics:
    """Class to track and calculate performance metrics."""
    
    def __init__(self):
        self.frame_times = []
        self.detection_counts = []
        self.start_time = time.time()
    
    def update(self, processing_time: float, detection_count: int):
        """Update metrics with new frame data."""
        self.frame_times.append(processing_time)
        self.detection_counts.append(detection_count)
    
    def get_current_fps(self) -> float:
        """Get current FPS based on recent frames."""
        if len(self.frame_times) < 2:
            return 0.0
        recent_times = self.frame_times[-10:]  # Last 10 frames
        return len(recent_times) / sum(recent_times) if recent_times else 0.0
    
    def get_average_fps(self) -> float:
        """Get average FPS since start."""
        total_time = time.time() - self.start_time
        return len(self.frame_times) / total_time if total_time > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "total_frames": len(self.frame_times),
            "average_fps": self.get_average_fps(),
            "current_fps": self.get_current_fps(),
            "average_processing_time": np.mean(self.frame_times) if self.frame_times else 0,
            "average_detections_per_frame": np.mean(self.detection_counts) if self.detection_counts else 0,
            "total_processing_time": sum(self.frame_times)
        }

def parse_arguments() -> dict:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced YOLO Object Detection")
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("-i", "--image", type=str, help="Path to input image")
    input_group.add_argument("-v", "--video", type=str, help="Path to input video")
    input_group.add_argument("--images-dir", type=str, help="Directory of images for batch processing")
    
    # Output
    parser.add_argument("-o", "--output", type=str, help="Path to output image/video")
    
    # Model configuration
    parser.add_argument("-y", "--yolo", type=str, default="yolo-coco", help="Base path to YOLO directory")
    parser.add_argument("-m", "--model", type=str, default="yolov3", 
                       choices=["yolov3", "yolov3-tiny", "yolov4", "yolov4-tiny"],
                       help="YOLO model type")
    
    # Detection parameters
    parser.add_argument("-c", "--confidence", type=float, default=0.5,
                       help="Minimum probability to filter weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.3,
                       help="Threshold when applying non-maxima suppression")
    
    # Performance options
    parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--input-size", type=int, nargs=2, default=[416, 416],
                       metavar=("WIDTH", "HEIGHT"), help="Input size for the model")
    parser.add_argument("--no-display", action="store_true", help="Run without display window")
    
    # Additional features
    parser.add_argument("--save-json", type=str, help="Save detection results to JSON file")
    parser.add_argument("--log-file", type=str, help="Save logs to file")
    
    return vars(parser.parse_args())

def setup_output_directory(output_path: Optional[str]) -> Optional[Path]:
    """Create output directory if it doesn't exist."""
    if not output_path:
        return None
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return Path(output_path)

def process_image(detector: YOLODetector, args: dict, metrics: PerformanceMetrics):
    """Process a single image."""
    image_path = args["image"]
    output_path = setup_output_directory(args.get("output"))
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Perform detection
    result = detector.detect(image, args["confidence"], args["threshold"])
    metrics.update(result.processing_time, len(result.boxes))
    
    # Display results
    summary = metrics.get_summary()
    cv2.putText(result.frame, f"FPS: {summary['current_fps']:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(result.frame, f"Detections: {len(result.boxes)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save or display
    if output_path:
        cv2.imwrite(str(output_path), result.frame)
        detector.logger.info(f"Saved output image to {output_path}")
    
    if not args.get("no_display"):
        cv2.imshow("Object Detection", result.frame)
        cv2.waitKey(0)
    
    # Save JSON results if requested
    if args.get("save_json"):
        save_detection_results(result, output_path.with_suffix('.json') if output_path else Path("detections.json"))

def process_video(detector: YOLODetector, args: dict, metrics: PerformanceMetrics):
    """Process video file or webcam stream."""
    video_source = args.get("video", 0)  # 0 for webcam
    output_path = setup_output_directory(args.get("output"))
    no_display = args.get("no_display", False)
    
    # Open video source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_source}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if video_source != 0 else 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        detector.logger.info(f"Writing output to: {output_path}")
    
    detector.logger.info(f"Starting video processing (Source: {video_source})")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            result = detector.detect(frame, args["confidence"], args["threshold"])
            metrics.update(result.processing_time, len(result.boxes))
            
            # Add overlay information
            summary = metrics.get_summary()
            overlay_text = [
                f"FPS: {summary['current_fps']:.1f}",
                f"Detections: {len(result.boxes)}",
                f"Frame: {frame_count}/{total_frames}" if total_frames > 0 else f"Frame: {frame_count}"
            ]
            
            for i, text in enumerate(overlay_text):
                cv2.putText(result.frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write frame if output specified
            if writer:
                writer.write(result.frame)
            
            # Display frame
            if not no_display:
                cv2.imshow("Object Detection", result.frame)
            
            frame_count += 1
            
            # Progress update for files
            if total_frames > 0 and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                detector.logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or (no_display and frame_count >= total_frames and total_frames > 0):
                break
                
    except KeyboardInterrupt:
        detector.logger.info("Processing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if not no_display:
            cv2.destroyAllWindows()
        
        # Print final summary
        summary = metrics.get_summary()
        detector.logger.info("Processing completed")
        detector.logger.info(f"Total frames processed: {summary['total_frames']}")
        detector.logger.info(f"Average FPS: {summary['average_fps']:.2f}")
        detector.logger.info(f"Average processing time: {summary['average_processing_time']*1000:.2f}ms")

def save_detection_results(result: DetectionResult, output_path: Path):
    """Save detection results to JSON file."""
    data = {
        "timestamp": time.time(),
        "processing_time": result.processing_time,
        "detections": [
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "confidence": float(confidence),
                "bbox": {
                    "x": int(box[0]),
                    "y": int(box[1]),
                    "width": int(box[2]),
                    "height": int(box[3])
                }
            }
            for box, confidence, class_id, class_name in zip(
                result.boxes, result.confidences, result.class_ids, result.class_names
            )
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging to file if specified
    if args.get("log_file"):
        file_handler = logging.FileHandler(args["log_file"])
        file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    try:
        # Initialize detector
        detector = YOLODetector(
            yolo_path=Path(args["yolo"]),
            model_type=args["model"],
            use_gpu=args["gpu"]
        )
        
        # Set input size if specified
        if args["input_size"]:
            detector.set_input_size(args["input_size"][0], args["input_size"][1])
        
        # Initialize metrics tracker
        metrics = PerformanceMetrics()
        
        # Process based on input type
        if args.get("image"):
            process_image(detector, args, metrics)
        elif args.get("images_dir"):
            # Batch image processing (simplified implementation)
            images_dir = Path(args["images_dir"])
            for image_path in images_dir.glob("*.*"):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    args["image"] = str(image_path)
                    args["output"] = str(images_dir / "output" / f"detected_{image_path.name}")
                    process_image(detector, args, metrics)
        else:
            process_video(detector, args, metrics)
            
    except Exception as e:
        logging.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()