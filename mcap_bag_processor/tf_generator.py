"""
URDF parser and tf_static message generator.

Parses URDF files and extracts all fixed joint transforms to generate
tf2_msgs/TFMessage for /tf_static topic.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class Transform:
    """Represents a single transform from parent to child frame."""
    parent_frame: str
    child_frame: str
    translation: Tuple[float, float, float]  # x, y, z
    rotation: Tuple[float, float, float, float]  # x, y, z, w (quaternion)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).
    Uses the same convention as ROS: roll around X, pitch around Y, yaw around Z.
    """
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (x, y, z, w)


def parse_urdf(urdf_path: str) -> List[Transform]:
    """
    Parse a URDF file and extract all fixed joint transforms.
    
    Args:
        urdf_path: Path to the URDF file
        
    Returns:
        List of Transform objects for all fixed joints
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    transforms = []
    
    for joint in root.findall('joint'):
        joint_type = joint.get('type', '')
        
        # Only process fixed joints for tf_static
        if joint_type != 'fixed':
            continue
            
        # Get parent and child links
        parent_elem = joint.find('parent')
        child_elem = joint.find('child')
        
        if parent_elem is None or child_elem is None:
            continue
            
        parent_link = parent_elem.get('link', '')
        child_link = child_elem.get('link', '')
        
        if not parent_link or not child_link:
            continue
        
        # Get origin (default to identity if not specified)
        origin_elem = joint.find('origin')
        
        if origin_elem is not None:
            xyz_str = origin_elem.get('xyz', '0 0 0')
            rpy_str = origin_elem.get('rpy', '0 0 0')
            
            xyz = tuple(float(v) for v in xyz_str.split())
            rpy = tuple(float(v) for v in rpy_str.split())
        else:
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
        
        # Convert RPY to quaternion
        quat = euler_to_quaternion(rpy[0], rpy[1], rpy[2])
        
        transforms.append(Transform(
            parent_frame=parent_link,
            child_frame=child_link,
            translation=xyz,
            rotation=quat
        ))
    
    return transforms


def create_tf_static_message(transforms: List[Transform], timestamp_ns: int) -> dict:
    """
    Create a tf2_msgs/TFMessage dictionary suitable for MCAP serialization.
    
    Args:
        transforms: List of Transform objects
        timestamp_ns: Timestamp in nanoseconds
        
    Returns:
        Dictionary representing a TFMessage
    """
    sec = timestamp_ns // 1_000_000_000
    nanosec = timestamp_ns % 1_000_000_000
    
    transform_stamped_list = []
    
    for tf in transforms:
        transform_stamped = {
            'header': {
                'stamp': {
                    'sec': sec,
                    'nanosec': nanosec
                },
                'frame_id': tf.parent_frame
            },
            'child_frame_id': tf.child_frame,
            'transform': {
                'translation': {
                    'x': tf.translation[0],
                    'y': tf.translation[1],
                    'z': tf.translation[2]
                },
                'rotation': {
                    'x': tf.rotation[0],
                    'y': tf.rotation[1],
                    'z': tf.rotation[2],
                    'w': tf.rotation[3]
                }
            }
        }
        transform_stamped_list.append(transform_stamped)
    
    return {'transforms': transform_stamped_list}


def add_static_transforms(transforms: List[Transform]) -> List[Transform]:
    """
    Add additional static transforms commonly needed (map->odom, odom->base_link).
    
    Args:
        transforms: Existing list of transforms from URDF
        
    Returns:
        Extended list of transforms
    """
    # Add map -> odom (identity)
    transforms.append(Transform(
        parent_frame='map',
        child_frame='odom',
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0)
    ))
    
    # Add odom -> base_link (identity - will be overwritten by actual odometry)
    transforms.append(Transform(
        parent_frame='odom',
        child_frame='base_link',
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0)
    ))
    
    return transforms


if __name__ == '__main__':
    # Test with a URDF file
    import sys
    if len(sys.argv) > 1:
        transforms = parse_urdf(sys.argv[1])
        transforms = add_static_transforms(transforms)
        print(f"Found {len(transforms)} transforms:")
        for tf in transforms:
            print(f"  {tf.parent_frame} -> {tf.child_frame}")
            print(f"    xyz: {tf.translation}")
            print(f"    quat: {tf.rotation}")

