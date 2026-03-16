import torch

from aerial_gym.sensors.warp.warp_sensor import WarpSensor
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import quat_mul, tf_apply

logger = CustomLogger("RecordingWarpSensor")


class RecordingWarpSensor(WarpSensor):
    """
    Recording-only warp sensor that stabilizes camera orientation by following
    robot yaw (vehicle frame orientation) and ignoring roll/pitch jitter.
    """

    def __init__(self, sensor_config, num_envs, mesh_id_list, device):
        super().__init__(sensor_config, num_envs, mesh_id_list, device)
        self.stable_robot_orientation = None

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)
        vehicle_orientation = global_tensor_dict.get("robot_vehicle_orientation")
        if vehicle_orientation is None:
            logger.warning(
                "robot_vehicle_orientation not found. Falling back to full robot orientation."
            )
            self.stable_robot_orientation = None
            return
        self.stable_robot_orientation = vehicle_orientation.unsqueeze(1).expand(
            -1, self.num_sensors, -1
        )

    def update(self):
        orientation_source = (
            self.stable_robot_orientation
            if self.stable_robot_orientation is not None
            else self.robot_orientation
        )
        self.sensor_position[:] = tf_apply(
            orientation_source, self.robot_position, self.sensor_local_position
        )
        self.sensor_orientation[:] = quat_mul(
            orientation_source,
            quat_mul(self.sensor_local_orientation, self.sensor_data_frame_quat),
        )

        self.sensor.capture()
        self.apply_noise()
        if self.cfg.sensor_type in ["camera", "lidar", "stereo_camera"]:
            self.apply_range_limits()
            self.normalize_observation()
