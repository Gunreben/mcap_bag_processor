"""
Core MCAP processor for offline bag manipulation.

Handles reading input MCAP, processing messages, and writing output MCAP
with added tf_static, camera_info, and filtered pointclouds.
"""

import os
import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Any

from mcap.reader import make_reader
from mcap.writer import Writer
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2.writer import serialize_dynamic

from .tf_generator import parse_urdf, add_static_transforms, create_tf_static_message
from .camera_info import (
    load_all_intrinsics, create_camera_info_message,
    get_camera_info_topic, get_intrinsics_for_image_topic
)
from .pointcloud import filter_zed_pointcloud
from .lidar_odometry import (
    LidarOdometry, KISS_ICP_AVAILABLE,
    pose_to_odometry_dict, pose_to_tf_msg_dict,
    compute_odom_base_link, compute_odom_sensor, build_sensor_transform,
)


# LiDAR odometry constants
LIDAR_POINTS_TOPIC = '/ouster/points'
LIDAR_ODOM_OUTPUT_TOPIC = '/kiss_icp/odom'

# Topics whose header.stamp may use the Ouster internal oscillator (boot time)
# instead of Unix epoch time. When fix_ouster_timestamps is enabled, their
# header.stamp is replaced with the MCAP log_time (wall-clock).
OUSTER_TIMESTAMP_TOPICS = {'/ouster/points', '/ouster/imu'}

# Default topics to include in output
DEFAULT_TOPICS = [
    '/camera/rear_left/image_raw/compressed',
    '/camera/rear_mid/image_raw/compressed',
    '/camera/rear_right/image_raw/compressed',
    '/camera/side_left/image_raw/compressed',
    '/camera/side_right/image_raw/compressed',
    '/camera/rear_left/image_raw/camera_info',
    '/camera/rear_mid/image_raw/camera_info',
    '/camera/rear_right/image_raw/camera_info',
    '/camera/side_left/image_raw/camera_info',
    '/camera/side_right/image_raw/camera_info',
    '/camera/zed_left_camera/image_raw/camera_info',
    '/kiss_icp/odom',
    '/novatel/oem7/bestpos',
    '/novatel/oem7/fix',
    '/novatel/oem7/odom',
    '/ouster/imu',
    '/ouster/points',
    '/robot_description',
    '/tf_static',
    '/zed/zed_node/left_raw/image_raw_color/compressed',
    '/zed/zed_node/odom',
    '/zed/zed_node/point_cloud/cloud_registered',
    '/zed/zed_node/point_cloud/cloud_registered/filtered',
    '/zed/zed_node/right_raw/image_raw_color/compressed',
]

# Topics that are passthrough (copy raw bytes as-is)
PASSTHROUGH_TOPICS = {
    '/camera/rear_left/image_raw/compressed',
    '/camera/rear_mid/image_raw/compressed',
    '/camera/rear_right/image_raw/compressed',
    '/camera/side_left/image_raw/compressed',
    '/camera/side_right/image_raw/compressed',
    '/novatel/oem7/bestpos',
    '/novatel/oem7/fix',
    '/novatel/oem7/odom',
    '/ouster/imu',
    '/ouster/points',
    '/robot_description',
    '/zed/zed_node/left_raw/image_raw_color/compressed',
    '/zed/zed_node/odom',
    '/zed/zed_node/right_raw/image_raw_color/compressed',
}

# Image topics that need camera_info generation
IMAGE_TOPICS_FOR_CAMERA_INFO = {
    '/camera/rear_left/image_raw/compressed',
    '/camera/rear_mid/image_raw/compressed',
    '/camera/rear_right/image_raw/compressed',
    '/camera/side_left/image_raw/compressed',
    '/camera/side_right/image_raw/compressed',
    '/zed/zed_node/left_raw/image_raw_color/compressed',
}

# Pointcloud topics to filter
POINTCLOUD_TOPICS_TO_FILTER = {
    '/zed/zed_node/point_cloud/cloud_registered',
}


def patch_cdr_header_stamp(raw_data: bytes, timestamp_ns: int) -> bytes:
    """Patch header.stamp in a CDR-encoded ROS 2 message.

    Standard ROS 2 CDR layout for messages starting with std_msgs/Header:
      bytes 0-3 : CDR encapsulation header (e.g. 0x00 0x01 0x00 0x00)
      bytes 4-7 : header.stamp.sec   (int32,  little-endian)
      bytes 8-11: header.stamp.nanosec (uint32, little-endian)

    Works for sensor_msgs/PointCloud2, sensor_msgs/Imu, and any message
    whose first field is a std_msgs/Header.
    """
    sec = int(timestamp_ns // 1_000_000_000)
    nanosec = int(timestamp_ns % 1_000_000_000)
    patched = bytearray(raw_data)
    struct.pack_into('<iI', patched, 4, sec, nanosec)
    return bytes(patched)


@dataclass
class ProcessingStats:
    """Statistics from bag processing."""
    total_messages: int = 0
    processed_messages: int = 0
    passthrough_messages: int = 0
    generated_camera_info: int = 0
    filtered_pointclouds: int = 0
    generated_lidar_odom: int = 0
    patched_ouster_timestamps: int = 0
    skipped_messages: int = 0
    topics_found: Set[str] = field(default_factory=set)


@dataclass 
class ProcessorConfig:
    """Configuration for bag processing."""
    input_path: str
    output_path: str
    urdf_path: Optional[str] = None
    calibration_dir: Optional[str] = None
    topics_to_include: List[str] = field(default_factory=lambda: DEFAULT_TOPICS.copy())
    filter_pointcloud: bool = True
    generate_camera_info: bool = True
    generate_tf_static: bool = True
    add_map_odom_tf: bool = True
    generate_lidar_odom: bool = False
    fix_ouster_timestamps: bool = False
    pointcloud_voxel_size: float = 0.2
    pointcloud_min_points: int = 2
    pointcloud_max_range: float = 100.0
    lidar_odom_max_range: float = 100.0
    lidar_odom_min_range: float = 1.0


class McapBagProcessor:
    """
    Offline MCAP bag processor.
    
    Processes MCAP bags to add:
    - tf_static from URDF
    - camera_info matched to image timestamps
    - filtered ZED pointclouds
    """
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        
        # Load URDF transforms
        self.transforms = []
        if config.generate_tf_static and config.urdf_path:
            self.transforms = parse_urdf(config.urdf_path)
            if config.add_map_odom_tf:
                self.transforms = add_static_transforms(self.transforms)
        
        # Load camera intrinsics
        self.camera_intrinsics = {}
        if config.generate_camera_info and config.calibration_dir:
            self.camera_intrinsics = load_all_intrinsics(config.calibration_dir)
        
        # Initialise LiDAR odometry (KISS-ICP)
        self.lidar_odom: Optional[LidarOdometry] = None
        self._T_base_sensor: Optional[Any] = None
        if config.generate_lidar_odom:
            if not KISS_ICP_AVAILABLE:
                raise ImportError(
                    "kiss-icp is required for LiDAR odometry. "
                    "Install it with: pip install kiss-icp"
                )
            self.lidar_odom = LidarOdometry(
                max_range=config.lidar_odom_max_range,
                min_range=config.lidar_odom_min_range,
            )
            # Look up base_link->os_sensor from URDF for proper TF computation
            for tf in self.transforms:
                if tf.child_frame == 'os_sensor' and tf.parent_frame == 'base_link':
                    self._T_base_sensor = build_sensor_transform(
                        tf.translation, tf.rotation,
                    )
                    break
            # Remove odom->base_link identity from static TFs (dynamic TF replaces it)
            self.transforms = [
                tf for tf in self.transforms
                if not (tf.parent_frame == 'odom' and tf.child_frame == 'base_link')
            ]
        
        # Encoder cache for dynamically generated messages
        self._encoder_cache: Dict[str, Callable] = {}
    
    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set a callback for progress updates: callback(progress_fraction, status_message)"""
        self.progress_callback = callback
    
    def _update_progress(self, progress: float, message: str):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(progress, message)
    
    def get_input_topics(self) -> List[str]:
        """Get list of topics in the input bag."""
        topics = []
        with open(self.config.input_path, 'rb') as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary:
                for channel_id, channel in summary.channels.items():
                    topics.append(channel.topic)
        return sorted(topics)
    
    def get_input_message_count(self) -> int:
        """Get total number of messages in input bag."""
        with open(self.config.input_path, 'rb') as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary and summary.statistics:
                return summary.statistics.message_count
        return 0
    
    def _get_encoder(self, schema_name: str, schema_text: str) -> Callable:
        """Get or create an encoder for a message type."""
        if schema_name not in self._encoder_cache:
            encoders = serialize_dynamic(schema_name, schema_text)
            self._encoder_cache[schema_name] = encoders[schema_name]
        return self._encoder_cache[schema_name]
    
    def process(self) -> ProcessingStats:
        """
        Process the input bag and write to output.
        
        Returns:
            ProcessingStats with processing results
        """
        self.stats = ProcessingStats()
        
        # Ensure output directory exists
        output_dir = os.path.dirname(self.config.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self._update_progress(0.0, "Opening input bag...")
        
        total_messages = self.get_input_message_count()
        
        with open(self.config.input_path, 'rb') as input_file:
            reader = make_reader(input_file, decoder_factories=[DecoderFactory()])
            
            with open(self.config.output_path, 'wb') as output_file:
                writer = Writer(output_file)
                writer.start(profile='ros2', library='mcap_bag_processor')
                
                # Track which topics we've registered
                registered_channels: Dict[str, int] = {}
                schemas: Dict[str, int] = {}
                schema_texts: Dict[str, str] = {}  # Keep schema text for encoding
                
                # Track timestamps for tf_static (write once at start)
                tf_static_written = False
                first_timestamp = None
                
                # Read and process all messages
                message_count = 0
                for schema, channel, message in reader.iter_messages():
                    message_count += 1
                    self.stats.total_messages += 1
                    self.stats.topics_found.add(channel.topic)
                    
                    # Track first timestamp
                    if first_timestamp is None:
                        first_timestamp = message.log_time
                    
                    # Write tf_static at start
                    if not tf_static_written and self.config.generate_tf_static and self.transforms:
                        self._write_tf_static(writer, first_timestamp, registered_channels, schemas, schema_texts)
                        tf_static_written = True
                    
                    # Progress update
                    if message_count % 1000 == 0:
                        progress = message_count / max(total_messages, 1)
                        self._update_progress(progress, f"Processing message {message_count}/{total_messages}")
                    
                    # Skip tf and tf_static from original bag (we generate our own)
                    if channel.topic in ('/tf', '/tf_static'):
                        self.stats.skipped_messages += 1
                        continue
                    
                    # Check if topic should be included
                    topic_included = channel.topic in self.config.topics_to_include
                    
                    # Patch Ouster timestamps: replace header.stamp with log_time
                    if (topic_included
                            and self.config.fix_ouster_timestamps
                            and channel.topic in OUSTER_TIMESTAMP_TOPICS):
                        patched_data = patch_cdr_header_stamp(message.data, message.log_time)
                        self._write_raw_message(
                            writer, channel.topic,
                            schema.name, schema.data.decode('utf-8'),
                            patched_data, message.log_time,
                            registered_channels, schemas, schema_texts
                        )
                        self.stats.patched_ouster_timestamps += 1
                        self.stats.processed_messages += 1
                    
                    # Handle passthrough topics - copy raw bytes directly
                    elif topic_included and channel.topic in PASSTHROUGH_TOPICS:
                        self._write_raw_message(
                            writer, channel.topic, 
                            schema.name, schema.data.decode('utf-8'),
                            message.data, message.log_time,
                            registered_channels, schemas, schema_texts
                        )
                        self.stats.passthrough_messages += 1
                        self.stats.processed_messages += 1
                    
                    # Handle pointcloud filtering
                    elif channel.topic in POINTCLOUD_TOPICS_TO_FILTER:
                        schema_text = schema.data.decode('utf-8')
                        
                        if self.config.filter_pointcloud:
                            # Decode the message for filtering
                            decoder = DecoderFactory()
                            decoder_fn = decoder.decoder_for('cdr', schema)
                            if decoder_fn:
                                decoded_msg = decoder_fn(message.data)
                                msg_dict = self._msg_to_dict(decoded_msg)
                                
                                filtered_msg = filter_zed_pointcloud(
                                    msg_dict,
                                    voxel_size=self.config.pointcloud_voxel_size,
                                    min_points_per_voxel=self.config.pointcloud_min_points,
                                    max_valid_range=self.config.pointcloud_max_range
                                )
                                
                                # Write filtered pointcloud
                                output_topic = channel.topic + '/filtered'
                                self._write_encoded_message(
                                    writer, output_topic,
                                    schema.name, schema_text,
                                    filtered_msg, message.log_time,
                                    registered_channels, schemas, schema_texts
                                )
                                self.stats.filtered_pointclouds += 1
                                self.stats.processed_messages += 1
                        
                        # Also write original if requested
                        if topic_included:
                            self._write_raw_message(
                                writer, channel.topic,
                                schema.name, schema_text,
                                message.data, message.log_time,
                                registered_channels, schemas, schema_texts
                            )
                            self.stats.passthrough_messages += 1
                    
                    # Generate camera_info for image topics
                    if (self.config.generate_camera_info and 
                        channel.topic in IMAGE_TOPICS_FOR_CAMERA_INFO):
                        
                        intrinsics = get_intrinsics_for_image_topic(
                            channel.topic, self.camera_intrinsics
                        )
                        
                        if intrinsics:
                            camera_info_topic = get_camera_info_topic(channel.topic)
                            if camera_info_topic:
                                camera_info_msg = create_camera_info_message(
                                    intrinsics, message.log_time
                                )
                                
                                self._write_encoded_message(
                                    writer, camera_info_topic,
                                    'sensor_msgs/msg/CameraInfo',
                                    CAMERA_INFO_SCHEMA,
                                    camera_info_msg, message.log_time,
                                    registered_channels, schemas, schema_texts
                                )
                                self.stats.generated_camera_info += 1
                    
                    # Generate LiDAR odometry from Ouster pointclouds
                    if (self.lidar_odom is not None and
                            channel.topic == LIDAR_POINTS_TOPIC):
                        try:
                            decoder = DecoderFactory()
                            decoder_fn = decoder.decoder_for('cdr', schema)
                            if decoder_fn:
                                decoded_msg = decoder_fn(message.data)
                                odom_msg = self.lidar_odom.process_pointcloud(
                                    decoded_msg, message.log_time
                                )
                                if odom_msg is not None:
                                    T_kiss = self.lidar_odom.last_pose

                                    if self._T_base_sensor is not None:
                                        T_odom_base = compute_odom_base_link(
                                            T_kiss, self._T_base_sensor,
                                        )
                                        T_odom_sensor = compute_odom_sensor(
                                            T_kiss, self._T_base_sensor,
                                        )
                                    else:
                                        T_odom_base = T_kiss
                                        T_odom_sensor = T_kiss

                                    # Write corrected nav_msgs/Odometry
                                    corrected_odom = pose_to_odometry_dict(
                                        T_odom_sensor, message.log_time,
                                        frame_id='odom',
                                        child_frame_id='os_sensor',
                                    )
                                    self._write_encoded_message(
                                        writer, LIDAR_ODOM_OUTPUT_TOPIC,
                                        'nav_msgs/msg/Odometry',
                                        ODOMETRY_SCHEMA,
                                        corrected_odom, message.log_time,
                                        registered_channels, schemas, schema_texts
                                    )

                                    # Write odom->base_link on /tf
                                    tf_msg = pose_to_tf_msg_dict(
                                        T_odom_base, message.log_time,
                                        frame_id='odom',
                                        child_frame_id='base_link',
                                    )
                                    self._write_encoded_message(
                                        writer, '/tf',
                                        'tf2_msgs/msg/TFMessage',
                                        TF_MESSAGE_SCHEMA,
                                        tf_msg, message.log_time,
                                        registered_channels, schemas, schema_texts
                                    )
                                    self.stats.generated_lidar_odom += 1
                        except Exception as e:
                            print(f"Warning: LiDAR odometry failed on frame: {e}")
                
                writer.finish()
        
        self._update_progress(1.0, "Processing complete!")
        return self.stats
    
    def _msg_to_dict(self, msg) -> dict:
        """Convert a ROS message object to a dictionary."""
        if isinstance(msg, dict):
            return msg
        
        # Handle dataclass-like objects from mcap_ros2
        result = {}
        for key in dir(msg):
            if key.startswith('_'):
                continue
            value = getattr(msg, key)
            if callable(value):
                continue
            if hasattr(value, '__dict__') or hasattr(value, '__slots__'):
                result[key] = self._msg_to_dict(value)
            else:
                result[key] = value
        return result
    
    def _ensure_schema_channel(
        self, writer: Writer, topic: str, schema_name: str, schema_text: str,
        registered_channels: Dict[str, int], schemas: Dict[str, int],
        schema_texts: Dict[str, str]
    ) -> int:
        """Ensure schema and channel are registered, return channel_id."""
        # Register schema if needed
        if schema_name not in schemas:
            schema_id = writer.register_schema(
                name=schema_name,
                encoding='ros2msg',
                data=schema_text.encode('utf-8')
            )
            schemas[schema_name] = schema_id
            schema_texts[schema_name] = schema_text
        
        # Register channel if needed
        if topic not in registered_channels:
            channel_id = writer.register_channel(
                topic=topic,
                message_encoding='cdr',
                schema_id=schemas[schema_name]
            )
            registered_channels[topic] = channel_id
        
        return registered_channels[topic]
    
    def _write_raw_message(
        self, writer: Writer, topic: str, schema_name: str, schema_text: str,
        raw_data: bytes, timestamp: int,
        registered_channels: Dict[str, int], schemas: Dict[str, int],
        schema_texts: Dict[str, str]
    ):
        """Write a raw message (passthrough) to the output bag."""
        channel_id = self._ensure_schema_channel(
            writer, topic, schema_name, schema_text,
            registered_channels, schemas, schema_texts
        )
        
        writer.add_message(
            channel_id=channel_id,
            log_time=timestamp,
            data=raw_data,
            publish_time=timestamp
        )
    
    def _write_encoded_message(
        self, writer: Writer, topic: str, schema_name: str, schema_text: str,
        msg_data: Any, timestamp: int,
        registered_channels: Dict[str, int], schemas: Dict[str, int],
        schema_texts: Dict[str, str]
    ):
        """Encode and write a message to the output bag."""
        channel_id = self._ensure_schema_channel(
            writer, topic, schema_name, schema_text,
            registered_channels, schemas, schema_texts
        )
        
        try:
            encoder = self._get_encoder(schema_name, schema_text)
            encoded = encoder(msg_data)
            
            writer.add_message(
                channel_id=channel_id,
                log_time=timestamp,
                data=encoded,
                publish_time=timestamp
            )
        except Exception as e:
            print(f"Warning: Failed to encode message for {topic}: {e}")
    
    def _write_tf_static(
        self, writer: Writer, timestamp: int,
        registered_channels: Dict[str, int], schemas: Dict[str, int],
        schema_texts: Dict[str, str]
    ):
        """Write tf_static message with all transforms from URDF."""
        tf_msg = create_tf_static_message(self.transforms, timestamp)
        
        self._write_encoded_message(
            writer, '/tf_static',
            'tf2_msgs/msg/TFMessage', TF_MESSAGE_SCHEMA,
            tf_msg, timestamp,
            registered_channels, schemas, schema_texts
        )


# Schema definitions for generated messages
TF_MESSAGE_SCHEMA = """
geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
""".strip()

CAMERA_INFO_SCHEMA = """
std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: sensor_msgs/RegionOfInterest
uint32 x_offset
uint32 y_offset
uint32 height
uint32 width
bool do_rectify
""".strip()

ODOMETRY_SCHEMA = """
std_msgs/Header header
string child_frame_id
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/PoseWithCovariance
geometry_msgs/Pose pose
float64[36] covariance
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
================================================================================
MSG: geometry_msgs/TwistWithCovariance
geometry_msgs/Twist twist
float64[36] covariance
================================================================================
MSG: geometry_msgs/Twist
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
""".strip()


if __name__ == '__main__':
    print("MCAP Bag Processor module loaded successfully")
