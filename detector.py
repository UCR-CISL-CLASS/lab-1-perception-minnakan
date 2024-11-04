import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import gc
import json
import os
import time
from datetime import datetime

class Detector:
    def __init__(self):
        """Initialize detector with memory management and logging"""
        # Setup model and device
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        except RuntimeError:
            print("Warning: GPU memory insufficient, falling back to CPU")
            self.device = torch.device("cpu")
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()

        # Image transform pipeline with resizingS
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((360, 640)),  # Reduced resolution
            T.ToTensor()
        ])

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging directory and files"""
        # Create logs directory if it doesn't exist
        self.log_dir = "detection_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create new log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"detection_log_{timestamp}.json")
        
        # Initialize log data structure
        self.log_data = {
            "metadata": {
                "start_time": timestamp,
                "device": str(self.device),
                "image_size": (360, 640)
            },
            "detections": []
        }

    def log_detection(self, frame_id, detections, processing_time):
        """Log detection results for a single frame"""
        detection_entry = {
            "frame_id": int(frame_id),
            "timestamp": time.time(),
            "processing_time": processing_time,
            "detections": []
        }

        # Log each detected object
        if "det_boxes" in detections and len(detections["det_boxes"]) > 0:
            for i in range(len(detections["det_boxes"])):
                detection_entry["detections"].append({
                    "box_3d": detections["det_boxes"][i].tolist(),
                    "class": int(detections["det_class"][i]),
                    "score": float(detections["det_score"][i])
                })

        self.log_data["detections"].append(detection_entry)

        # Periodically save to file (every 100 frames)
        if len(self.log_data["detections"]) % 100 == 0:
            self.save_logs()

    def save_logs(self):
        """Save current logs to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2)
        except Exception as e:
            print(f"Error saving logs: {str(e)}")

    def sensors(self):
        """Sensor configurations for CARLA environment"""
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.5, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.5, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'range': 50, 'rotation_frequency': 20, 'channels': 64,
             'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000, 'id': 'LIDAR'}
        ]
        return sensors

    def detect(self, sensor_data):
        """Run detection with memory optimization and logging"""
        start_time = time.time()
        try:
            # Extract data from the 'Center' camera sensor
            camera_data = sensor_data.get('Left')
            if camera_data is None:
                return {}

            frame_id, image = camera_data
            image = image[:, :, :3]  # Remove alpha channel

            # Transform and prepare image
            try:
                # Clear GPU memory before processing
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

                # Process image
                image_tensor = self.transform(image)
                input_tensor = image_tensor.unsqueeze(0).to(self.device)

                # Run inference
                with torch.no_grad():
                    outputs = self.model(input_tensor)

                # Move results to CPU immediately to free GPU memory
                if self.device.type == 'cuda':
                    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

                # Process detections
                boxes = outputs[0]['boxes'].numpy()
                labels = outputs[0]['labels'].numpy()
                scores = outputs[0]['scores'].numpy()

                # Filter by confidence
                mask = scores > 0.01
                boxes = boxes[mask]
                labels = labels[mask]
                scores = scores[mask]

                # Convert 2D boxes to 3D format
                boxes_3d = []
                for box in boxes:
                    box_3d = self.convert_to_3d(box)
                    boxes_3d.append(box_3d)

                results = {
                    "det_boxes": np.array(boxes_3d) if boxes_3d else np.array([]),
                    "det_class": labels - 1,  # Adjust class indices
                    "det_score": scores
                }

                # Log the results
                processing_time = time.time() - start_time
                self.log_detection(frame_id, results, processing_time)

                return results

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                    print("Warning: GPU memory error, skipping frame")
                return {}

        except Exception as e:
            print(f"Error in detection: {str(e)}")
            return {}

        finally:
            # Ensure memory is cleared
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

    def convert_to_3d(self, box):
        """Convert 2D bounding box to 3D format"""
        x1, y1, x2, y2 = box
        # Scale factor to account for resolution change
        scale_x = 1280 / 640
        scale_y = 720 / 360
        
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y
        
        z = 1.0  # Assume default depth
        return np.array([
            [x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0],  # Bottom face
            [x1, y1, z], [x2, y1, z], [x2, y2, z], [x1, y2, z]   # Top face
        ])

    def __del__(self):
        """Cleanup when detector is destroyed"""
        self.save_logs()  # Save any remaining logs