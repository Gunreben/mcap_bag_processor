"""
Pointcloud filtering for ZED stereo camera data.

Implements:
1. Alpha channel filter - keeps only fully confident points (alpha == 255)
2. Voxel-based outlier filter - removes isolated points in sparse voxels
"""

import struct
from typing import Tuple, Optional
import numpy as np


def parse_pointcloud2_fields(fields: list) -> dict:
    """
    Parse PointCloud2 field definitions to get offsets and types.
    
    Args:
        fields: List of field objects or dictionaries from PointCloud2 message
        
    Returns:
        Dictionary mapping field names to (offset, datatype, count)
    """
    field_info = {}
    for field in fields:
        # Handle both dict and object access patterns
        if isinstance(field, dict):
            name = field['name']
            offset = field['offset']
            datatype = field['datatype']
            count = field.get('count', 1)
        else:
            # Handle PointField objects from mcap_ros2 decoder
            name = field.name
            offset = field.offset
            datatype = field.datatype
            count = getattr(field, 'count', 1)
        field_info[name] = (offset, datatype, count)
    return field_info


# PointCloud2 datatype constants
DATATYPE_TO_DTYPE = {
    1: np.int8,    # INT8
    2: np.uint8,   # UINT8
    3: np.int16,   # INT16
    4: np.uint16,  # UINT16
    5: np.int32,   # INT32
    6: np.uint32,  # UINT32
    7: np.float32, # FLOAT32
    8: np.float64, # FLOAT64
}

DATATYPE_TO_SIZE = {
    1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8
}


def filter_pointcloud_alpha(
    data: bytes,
    point_step: int,
    fields: list,
    alpha_threshold: int = 255
) -> Tuple[bytes, int]:
    """
    Filter pointcloud by alpha channel value.
    
    ZED cameras encode confidence in the alpha channel. Points with alpha < 255
    have lower confidence and are typically artifacts.
    
    Args:
        data: Raw pointcloud data bytes
        point_step: Size of each point in bytes
        fields: List of field definitions
        alpha_threshold: Keep only points with this alpha value (default 255)
        
    Returns:
        Tuple of (filtered_data, num_points)
    """
    field_info = parse_pointcloud2_fields(fields)
    
    # Check if we have RGBA or just RGB
    if 'rgba' in field_info:
        rgba_offset = field_info['rgba'][0]
        has_rgba = True
    elif 'rgb' in field_info:
        # Some formats pack RGBA into 'rgb' field
        rgba_offset = field_info['rgb'][0]
        has_rgba = True
    else:
        # No color info, return unchanged
        num_points = len(data) // point_step
        return data, num_points
    
    # Convert to numpy for fast processing
    num_points = len(data) // point_step
    if num_points == 0:
        return data, 0
    
    # Read raw data as byte array
    data_array = np.frombuffer(data, dtype=np.uint8).reshape(num_points, point_step)
    
    # Extract alpha channel (4th byte of RGBA at rgba_offset)
    # RGBA is packed as [R, G, B, A] in little-endian
    alpha_values = data_array[:, rgba_offset + 3]
    
    # Filter by alpha
    mask = alpha_values == alpha_threshold
    filtered_data = data_array[mask]
    
    return filtered_data.tobytes(), int(np.sum(mask))


def filter_pointcloud_outliers(
    data: bytes,
    point_step: int,
    fields: list,
    voxel_size: float = 0.2,
    min_points_per_voxel: int = 2,
    max_valid_range: float = 100.0
) -> Tuple[bytes, int]:
    """
    Filter outlier points using voxel-based density check.
    
    Points in voxels with fewer than min_points_per_voxel neighbors are removed.
    Also removes NaN/Inf points and points beyond max_valid_range.
    
    Args:
        data: Raw pointcloud data bytes
        point_step: Size of each point in bytes
        fields: List of field definitions
        voxel_size: Size of voxel grid cells in meters
        min_points_per_voxel: Minimum points required in a voxel
        max_valid_range: Maximum coordinate value to consider valid
        
    Returns:
        Tuple of (filtered_data, num_points)
    """
    field_info = parse_pointcloud2_fields(fields)
    
    # Get x, y, z field offsets
    if 'x' not in field_info or 'y' not in field_info or 'z' not in field_info:
        num_points = len(data) // point_step
        return data, num_points
    
    x_offset = field_info['x'][0]
    y_offset = field_info['y'][0]
    z_offset = field_info['z'][0]
    
    # Convert to numpy
    num_points = len(data) // point_step
    if num_points == 0:
        return data, 0
    
    data_array = np.frombuffer(data, dtype=np.uint8).reshape(num_points, point_step)
    
    # Extract x, y, z as float32
    x = np.frombuffer(data_array[:, x_offset:x_offset+4].tobytes(), dtype=np.float32)
    y = np.frombuffer(data_array[:, y_offset:y_offset+4].tobytes(), dtype=np.float32)
    z = np.frombuffer(data_array[:, z_offset:z_offset+4].tobytes(), dtype=np.float32)
    
    # Remove NaN/Inf
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    
    # Remove points outside range
    max_coord = np.maximum(np.abs(x), np.maximum(np.abs(y), np.abs(z)))
    valid_mask &= max_coord <= max_valid_range
    
    # Voxel-based filtering
    if np.sum(valid_mask) > 0:
        # Get valid points
        valid_indices = np.where(valid_mask)[0]
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        
        # Compute voxel indices
        inv_voxel_size = 1.0 / voxel_size
        voxel_x = np.floor(x_valid * inv_voxel_size).astype(np.int32)
        voxel_y = np.floor(y_valid * inv_voxel_size).astype(np.int32)
        voxel_z = np.floor(z_valid * inv_voxel_size).astype(np.int32)
        
        # Create voxel keys (combine into single int64 for hashing)
        # Use a large prime to minimize collisions
        voxel_keys = (voxel_x.astype(np.int64) * 73856093 + 
                      voxel_y.astype(np.int64) * 19349663 + 
                      voxel_z.astype(np.int64) * 83492791)
        
        # Count points per voxel
        unique_keys, inverse, counts = np.unique(voxel_keys, return_inverse=True, return_counts=True)
        point_counts = counts[inverse]
        
        # Keep only points in sufficiently populated voxels
        density_mask = point_counts >= min_points_per_voxel
        
        # Map back to original indices
        final_indices = valid_indices[density_mask]
        filtered_data = data_array[final_indices]
        
        return filtered_data.tobytes(), len(final_indices)
    
    return b'', 0


def _get_attr(obj, key, default=None):
    """Get attribute from dict or object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def filter_zed_pointcloud(
    msg_data,
    alpha_threshold: int = 255,
    voxel_size: float = 0.2,
    min_points_per_voxel: int = 2,
    max_valid_range: float = 100.0
) -> dict:
    """
    Apply full ZED pointcloud filtering pipeline.
    
    Args:
        msg_data: PointCloud2 message (dict or object)
        alpha_threshold: Alpha value to keep (255 = fully confident)
        voxel_size: Voxel grid size for outlier filtering
        min_points_per_voxel: Minimum neighbors to keep point
        max_valid_range: Maximum coordinate range
        
    Returns:
        Filtered PointCloud2 message dictionary
    """
    data = _get_attr(msg_data, 'data', b'')
    if isinstance(data, (list, tuple)):
        data = bytes(data)
    
    point_step = _get_attr(msg_data, 'point_step', 32)
    fields = _get_attr(msg_data, 'fields', [])
    
    # Stage 1: Alpha filter
    filtered_data, num_points = filter_pointcloud_alpha(
        data, point_step, fields, alpha_threshold
    )
    
    # Stage 2: Outlier filter
    filtered_data, num_points = filter_pointcloud_outliers(
        filtered_data, point_step, fields,
        voxel_size, min_points_per_voxel, max_valid_range
    )
    
    # Create filtered message as dict for encoding
    # Copy original message fields
    header = _get_attr(msg_data, 'header', {})
    if not isinstance(header, dict):
        header = {
            'stamp': {
                'sec': _get_attr(header.stamp, 'sec', 0) if hasattr(header, 'stamp') else 0,
                'nanosec': _get_attr(header.stamp, 'nanosec', 0) if hasattr(header, 'stamp') else 0,
            },
            'frame_id': _get_attr(header, 'frame_id', '')
        }
    
    # Convert fields to dict format
    fields_dict = []
    for field in fields:
        if isinstance(field, dict):
            fields_dict.append(field)
        else:
            fields_dict.append({
                'name': field.name,
                'offset': field.offset,
                'datatype': field.datatype,
                'count': getattr(field, 'count', 1)
            })
    
    filtered_msg = {
        'header': header,
        'height': 1,
        'width': num_points,
        'fields': fields_dict,
        'is_bigendian': _get_attr(msg_data, 'is_bigendian', False),
        'point_step': point_step,
        'row_step': num_points * point_step,
        'data': list(filtered_data) if filtered_data else [],
        'is_dense': False
    }
    
    return filtered_msg


if __name__ == '__main__':
    # Simple test
    print("Pointcloud filter module loaded successfully")
    print("Functions available:")
    print("  - filter_pointcloud_alpha()")
    print("  - filter_pointcloud_outliers()")
    print("  - filter_zed_pointcloud()")

