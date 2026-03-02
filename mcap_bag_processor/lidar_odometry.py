"""
LiDAR odometry using KISS-ICP.

Processes PointCloud2 messages (e.g. from Ouster) and produces
nav_msgs/Odometry by running scan-to-map ICP registration.
"""

import math
from typing import Optional, Tuple

import numpy as np

try:
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    KISS_ICP_AVAILABLE = True
except ImportError:
    KISS_ICP_AVAILABLE = False


def extract_xyz_from_pointcloud2(msg_data) -> np.ndarray:
    """Extract Nx3 float64 XYZ array from a PointCloud2 message (dict or object)."""
    if isinstance(msg_data, dict):
        data = msg_data.get('data', b'')
        point_step = msg_data.get('point_step', 0)
        fields = msg_data.get('fields', [])
        width = msg_data.get('width', 0)
        height = msg_data.get('height', 1)
    else:
        data = getattr(msg_data, 'data', b'')
        point_step = getattr(msg_data, 'point_step', 0)
        fields = getattr(msg_data, 'fields', [])
        width = getattr(msg_data, 'width', 0)
        height = getattr(msg_data, 'height', 1)

    if isinstance(data, (list, tuple)):
        data = bytes(data)

    num_points = width * height
    if num_points == 0 or point_step == 0:
        return np.empty((0, 3), dtype=np.float64)

    offsets = {}
    for field in fields:
        if isinstance(field, dict):
            name, offset = field['name'], field['offset']
        else:
            name, offset = field.name, field.offset
        if name in ('x', 'y', 'z'):
            offsets[name] = offset

    if not all(k in offsets for k in ('x', 'y', 'z')):
        return np.empty((0, 3), dtype=np.float64)

    raw = np.frombuffer(data, dtype=np.uint8).reshape(num_points, point_step)
    x = np.frombuffer(raw[:, offsets['x']:offsets['x'] + 4].tobytes(), dtype=np.float32)
    y = np.frombuffer(raw[:, offsets['y']:offsets['y'] + 4].tobytes(), dtype=np.float32)
    z = np.frombuffer(raw[:, offsets['z']:offsets['z'] + 4].tobytes(), dtype=np.float32)

    points = np.column_stack([x, y, z]).astype(np.float64)
    valid = np.isfinite(points).all(axis=1)
    return points[valid]


def _rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to (x, y, z, w) quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def _pose_to_stamp(timestamp_ns: int) -> dict:
    return {
        'sec': int(timestamp_ns // 1_000_000_000),
        'nanosec': int(timestamp_ns % 1_000_000_000),
    }


def _pose_to_position_orientation(pose: np.ndarray) -> tuple:
    """Return (position_dict, orientation_dict) from a 4x4 SE3 matrix."""
    t = pose[:3, 3]
    qx, qy, qz, qw = _rotation_matrix_to_quaternion(pose[:3, :3])
    position = {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])}
    orientation = {'x': float(qx), 'y': float(qy), 'z': float(qz), 'w': float(qw)}
    return position, orientation


def pose_to_odometry_dict(
    pose: np.ndarray,
    timestamp_ns: int,
    frame_id: str = 'odom',
    child_frame_id: str = 'os_sensor',
) -> dict:
    """Convert a 4x4 SE3 pose matrix to a nav_msgs/Odometry dict."""
    stamp = _pose_to_stamp(timestamp_ns)
    position, orientation = _pose_to_position_orientation(pose)

    return {
        'header': {'stamp': stamp, 'frame_id': frame_id},
        'child_frame_id': child_frame_id,
        'pose': {
            'pose': {'position': position, 'orientation': orientation},
            'covariance': [0.0] * 36,
        },
        'twist': {
            'twist': {
                'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            },
            'covariance': [0.0] * 36,
        },
    }


def pose_to_tf_msg_dict(
    pose: np.ndarray,
    timestamp_ns: int,
    frame_id: str = 'odom',
    child_frame_id: str = 'base_link',
) -> dict:
    """Convert a 4x4 SE3 pose matrix to a tf2_msgs/TFMessage dict (single transform)."""
    stamp = _pose_to_stamp(timestamp_ns)
    position, orientation = _pose_to_position_orientation(pose)

    return {
        'transforms': [{
            'header': {'stamp': stamp, 'frame_id': frame_id},
            'child_frame_id': child_frame_id,
            'transform': {
                'translation': position,
                'rotation': orientation,
            },
        }],
    }


def compute_odom_base_link(
    T_kiss: np.ndarray,
    T_base_sensor: np.ndarray,
) -> np.ndarray:
    """Compute odom->base_link from the raw KISS-ICP pose and the URDF extrinsic.

    KISS-ICP poses live in a frame aligned with the sensor at t=0.
    Pre-multiplying by T_base_sensor rotates them into base_link-aligned
    coordinates, then post-multiplying by inv(T_base_sensor) gives the
    base_link pose in the odom frame:

        T_odom_base = T_base_sensor @ T_kiss @ inv(T_base_sensor)
    """
    return T_base_sensor @ T_kiss @ np.linalg.inv(T_base_sensor)


def compute_odom_sensor(
    T_kiss: np.ndarray,
    T_base_sensor: np.ndarray,
) -> np.ndarray:
    """Compute odom->os_sensor from the raw KISS-ICP pose and the URDF extrinsic.

        T_odom_sensor = T_base_sensor @ T_kiss
    """
    return T_base_sensor @ T_kiss


def build_sensor_transform(
    translation: tuple, rotation: tuple,
) -> np.ndarray:
    """Build a 4x4 SE3 matrix from (x,y,z) translation and (qx,qy,qz,qw) quaternion."""
    qx, qy, qz, qw = rotation
    R = np.array([
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


class LidarOdometry:
    """KISS-ICP based LiDAR odometry wrapper."""

    def __init__(self, max_range: float = 100.0, min_range: float = 1.0):
        if not KISS_ICP_AVAILABLE:
            raise ImportError(
                "kiss-icp is not installed. Install it with: pip install kiss-icp"
            )
        config = KISSConfig()
        config.data.max_range = max_range
        config.data.min_range = min_range
        config.data.deskew = False
        if config.mapping.voxel_size is None:
            config.mapping.voxel_size = float(max_range / 100.0)
        self._odom = KissICP(config=config)
        self._frame_count = 0

    @property
    def last_pose(self) -> np.ndarray:
        """Current 4x4 SE3 pose (odom->sensor) from the most recent registration."""
        return self._odom.last_pose

    def process_pointcloud(
        self,
        msg_data,
        timestamp_ns: int,
        frame_id: str = 'odom',
        child_frame_id: str = 'os_sensor',
    ) -> Optional[dict]:
        """
        Run KISS-ICP on a PointCloud2 message and return an Odometry dict.

        Returns None if the point cloud is empty or cannot be processed.
        """
        points = extract_xyz_from_pointcloud2(msg_data)
        if points.shape[0] == 0:
            return None

        timestamps = np.zeros(points.shape[0])
        self._odom.register_frame(points, timestamps=timestamps)
        self._frame_count += 1

        return pose_to_odometry_dict(self._odom.last_pose, timestamp_ns, frame_id, child_frame_id)
