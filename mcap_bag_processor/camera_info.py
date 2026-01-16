"""
Camera intrinsics loader and CameraInfo message generator.

Loads camera calibration data from YAML files and generates
sensor_msgs/CameraInfo messages.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import yaml


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    camera_name: str
    image_width: int
    image_height: int
    camera_matrix: List[float]  # 3x3 flattened (K matrix)
    distortion_model: str
    distortion_coefficients: List[float]
    rectification_matrix: List[float]  # 3x3 flattened (R matrix)
    projection_matrix: List[float]  # 3x4 flattened (P matrix)
    frame_id: str  # TF frame for this camera


def load_intrinsics_yaml(yaml_path: str) -> Optional[CameraIntrinsics]:
    """
    Load camera intrinsics from a YAML file.
    
    Supports the format used by camera_calibration_parsers.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        CameraIntrinsics object or None if parsing fails
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML {yaml_path}: {e}")
        return None
    
    if data is None:
        return None
    
    # Extract camera name from filename if not in YAML
    filename = os.path.basename(yaml_path)
    # Pattern: camera_rear_left.intrinsics.yaml -> rear_left
    match = re.match(r'camera_(.+)\.intrinsics\.yaml', filename)
    if match:
        camera_id = match.group(1)
    else:
        camera_id = data.get('camera_name', 'unknown')
    
    camera_name = data.get('camera_name', camera_id)
    
    # Get dimensions
    width = data.get('image_width', 1280)
    height = data.get('image_height', 720)
    
    # Get camera matrix (K)
    camera_matrix_data = data.get('camera_matrix', {})
    if isinstance(camera_matrix_data, dict):
        camera_matrix = camera_matrix_data.get('data', [1, 0, 0, 0, 1, 0, 0, 0, 1])
    else:
        camera_matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    
    # Get distortion model and coefficients
    distortion_model = data.get('distortion_model', 'plumb_bob')
    
    # Normalize distortion model for Foxglove compatibility
    if distortion_model.lower() in ('equidistant', 'fisheye'):
        distortion_model = 'kannala_brandt'
    
    distortion_data = data.get('distortion_coefficients', {})
    if isinstance(distortion_data, dict):
        distortion_coefficients = distortion_data.get('data', [0, 0, 0, 0, 0])
    else:
        distortion_coefficients = [0, 0, 0, 0, 0]
    
    # For kannala_brandt, only use first 4 coefficients
    if distortion_model == 'kannala_brandt' and len(distortion_coefficients) > 4:
        distortion_coefficients = distortion_coefficients[:4]
    
    # Get rectification matrix (R)
    rectification_data = data.get('rectification_matrix', {})
    if isinstance(rectification_data, dict):
        rectification_matrix = rectification_data.get('data', [1, 0, 0, 0, 1, 0, 0, 0, 1])
    else:
        rectification_matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    
    # Get projection matrix (P)
    projection_data = data.get('projection_matrix', {})
    if isinstance(projection_data, dict):
        projection_matrix = projection_data.get('data', [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    else:
        projection_matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    
    # Frame ID follows ROS convention: camera_<id>
    frame_id = f"camera_{camera_id}"
    
    return CameraIntrinsics(
        camera_name=camera_name,
        image_width=width,
        image_height=height,
        camera_matrix=camera_matrix,
        distortion_model=distortion_model,
        distortion_coefficients=distortion_coefficients,
        rectification_matrix=rectification_matrix,
        projection_matrix=projection_matrix,
        frame_id=frame_id
    )


def load_all_intrinsics(calibration_dir: str) -> Dict[str, CameraIntrinsics]:
    """
    Load all camera intrinsics from a calibration directory.
    
    Args:
        calibration_dir: Directory containing *.intrinsics.yaml files
        
    Returns:
        Dictionary mapping topic base name to CameraIntrinsics
    """
    intrinsics = {}
    
    if not os.path.isdir(calibration_dir):
        print(f"Calibration directory not found: {calibration_dir}")
        return intrinsics
    
    for filename in os.listdir(calibration_dir):
        if filename.endswith('.intrinsics.yaml'):
            filepath = os.path.join(calibration_dir, filename)
            camera = load_intrinsics_yaml(filepath)
            
            if camera:
                # Determine topic base name from filename
                # camera_rear_left.intrinsics.yaml -> rear_left
                # zed_left_camera.intrinsics.yaml -> zed_left_camera
                
                # Try camera_*.intrinsics.yaml pattern first
                match = re.match(r'camera_(.+)\.intrinsics\.yaml', filename)
                if match:
                    camera_id = match.group(1)
                    topic_base = f"/camera/{camera_id}/image_raw"
                    intrinsics[topic_base] = camera
                    print(f"Loaded intrinsics for {camera_id}: {filepath}")
                else:
                    # Try *.intrinsics.yaml pattern (e.g., zed_left_camera.intrinsics.yaml)
                    match = re.match(r'(.+)\.intrinsics\.yaml', filename)
                    if match:
                        camera_id = match.group(1)
                        topic_base = f"/camera/{camera_id}/image_raw"
                        intrinsics[topic_base] = camera
                        print(f"Loaded intrinsics for {camera_id}: {filepath}")
    
    return intrinsics


def create_camera_info_message(intrinsics: CameraIntrinsics, timestamp_ns: int) -> dict:
    """
    Create a sensor_msgs/CameraInfo dictionary for MCAP serialization.
    
    Args:
        intrinsics: CameraIntrinsics object
        timestamp_ns: Timestamp in nanoseconds
        
    Returns:
        Dictionary representing a CameraInfo message
    """
    sec = timestamp_ns // 1_000_000_000
    nanosec = timestamp_ns % 1_000_000_000
    
    return {
        'header': {
            'stamp': {
                'sec': sec,
                'nanosec': nanosec
            },
            'frame_id': intrinsics.frame_id
        },
        'height': intrinsics.image_height,
        'width': intrinsics.image_width,
        'distortion_model': intrinsics.distortion_model,
        'd': intrinsics.distortion_coefficients,
        'k': intrinsics.camera_matrix,
        'r': intrinsics.rectification_matrix,
        'p': intrinsics.projection_matrix,
        'binning_x': 0,
        'binning_y': 0,
        'roi': {
            'x_offset': 0,
            'y_offset': 0,
            'height': 0,
            'width': 0,
            'do_rectify': False
        }
    }


# Mapping from image topics to camera_info topics
IMAGE_TO_CAMERA_INFO = {
    '/camera/rear_left/image_raw/compressed': '/camera/rear_left/image_raw/camera_info',
    '/camera/rear_mid/image_raw/compressed': '/camera/rear_mid/image_raw/camera_info',
    '/camera/rear_right/image_raw/compressed': '/camera/rear_right/image_raw/camera_info',
    '/camera/side_left/image_raw/compressed': '/camera/side_left/image_raw/camera_info',
    '/camera/side_right/image_raw/compressed': '/camera/side_right/image_raw/camera_info',
    '/zed/zed_node/left_raw/image_raw_color/compressed': '/camera/zed_left_camera/image_raw/camera_info',
    '/zed/zed_node/right_raw/image_raw_color/compressed': '/camera/zed_right_camera/image_raw/camera_info',
}


def get_camera_info_topic(image_topic: str) -> Optional[str]:
    """Get the corresponding camera_info topic for an image topic."""
    return IMAGE_TO_CAMERA_INFO.get(image_topic)


def get_intrinsics_for_image_topic(image_topic: str, all_intrinsics: Dict[str, CameraIntrinsics]) -> Optional[CameraIntrinsics]:
    """
    Get the camera intrinsics for a given image topic.
    
    Args:
        image_topic: The image topic (e.g., /camera/rear_left/image_raw/compressed)
        all_intrinsics: Dictionary of all loaded intrinsics
        
    Returns:
        CameraIntrinsics for this camera or None
    """
    # Extract camera ID from topic
    # /camera/rear_left/image_raw/compressed -> rear_left
    # /zed/zed_node/left_raw/image_raw_color/compressed -> zed_left_camera
    
    if image_topic.startswith('/camera/'):
        parts = image_topic.split('/')
        if len(parts) >= 3:
            camera_id = parts[2]  # rear_left, rear_mid, etc.
            topic_base = f"/camera/{camera_id}/image_raw"
            return all_intrinsics.get(topic_base)
    
    elif 'zed' in image_topic:
        if 'left' in image_topic:
            return all_intrinsics.get('/camera/zed_left_camera/image_raw')
        elif 'right' in image_topic:
            return all_intrinsics.get('/camera/zed_right_camera/image_raw')
    
    return None


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        intrinsics = load_all_intrinsics(sys.argv[1])
        print(f"\nLoaded {len(intrinsics)} camera intrinsics:")
        for topic, cam in intrinsics.items():
            print(f"  {topic}: {cam.camera_name} ({cam.image_width}x{cam.image_height})")

