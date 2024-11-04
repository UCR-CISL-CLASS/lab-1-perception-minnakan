from detector import Detector
import numpy as np
import carla
from agents.navigation.basic_agent import BasicAgent as OriginalBasicAgent

class BasicAgent(OriginalBasicAgent):
    """
    BasicAgent with added detection capabilities using Faster R-CNN.
    """
    def __init__(self, vehicle, target_speed=20, opt_dict=None, map_inst=None, grp_inst=None):
        """Constructor method"""
        super().__init__(vehicle, target_speed, opt_dict, map_inst, grp_inst)
        
        # Initialize detector
        self._detector = Detector()
        
        # Store bounding box data for visualization
        self.bbox = {}
        
        # Store vehicle dimensions for sensor positioning
        self.bound_x = vehicle.bounding_box.extent.x
        self.bound_y = vehicle.bounding_box.extent.y
        self.bound_z = vehicle.bounding_box.extent.z

    def sensors(self):
        """Get list of sensors required by the agent"""
        sensors = self._detector.sensors()
        # Scale sensor positions based on vehicle dimensions
        for s in sensors:
            s['x'] = s['x'] * self.bound_x
            s['y'] = s['y'] * self.bound_y
            s['z'] = s['z'] * self.bound_z
        return sensors

    def gt_box_vertice_sequence(self, box):
        """Reorder box vertices to match expected sequence"""
        box = [box[1], box[3], box[7], box[5], 
               box[0], box[2], box[6], box[4]]
        return np.array(box)

    def gt_actors(self):
        """Get ground truth actors in the scene"""
        actor_list = self._world.get_actors()
        vehicles = actor_list.filter("*vehicle*")
        walkers = actor_list.filter("*walker.pedestrian*")
        
        detection_results = {
            "det_boxes": [],
            "det_class": [],
            "det_score": []
        }
        
        transform = self._vehicle.get_transform()
        
        # Calculate distance to ego vehicle
        def dist(l):
            return np.sqrt((l.x - transform.location.x)**2 + 
                         (l.y - transform.location.y)**2 + 
                         (l.z - transform.location.z)**2)
        
        # Process vehicles
        for v in vehicles:
            if dist(v.get_location()) > 50 or v.id == self._vehicle.id:
                continue
            bbox = [[v.x, v.y, v.z] for v in v.bounding_box.get_world_vertices(v.get_transform())]
            bbox = self.gt_box_vertice_sequence(bbox)
            detection_results["det_boxes"].append(bbox)
            detection_results["det_class"].append(0)  # 0 for vehicles
            detection_results["det_score"].append(1.0)
        
        # Process pedestrians
        for w in walkers:
            if dist(w.get_location()) > 50:
                continue
            bbox = [[w.x, w.y, w.z] for w in w.bounding_box.get_world_vertices(w.get_transform())]
            bbox = self.gt_box_vertice_sequence(bbox)
            detection_results["det_boxes"].append(bbox)
            detection_results["det_class"].append(1)  # 1 for pedestrians
            detection_results["det_score"].append(1.0)
        
        # Convert lists to numpy arrays
        detection_results["det_boxes"] = np.array(detection_results["det_boxes"])
        detection_results["det_class"] = np.array(detection_results["det_class"])
        detection_results["det_score"] = np.array(detection_results["det_score"])
        
        return detection_results

    def run_step(self, debug=False):
        """Execute one step of navigation"""
        # Get sensor data and run detection
        sensor_data = {
            'Center': self.get_sensor_data('Center')
        }
        detections = self._detector.detect(sensor_data)
        
        # Store detection results for visualization
        self.bbox = {
            'gt_det': detections,  # Ground truth bounding boxes
            'det': detections      # Detected bounding boxes
        }
        
        # Run original navigation logic
        return super().run_step(debug=debug)

    def destroy(self):
        """Clean up when agent is destroyed"""
        super().destroy()
