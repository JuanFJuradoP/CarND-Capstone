import rospy
from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        self.vehicle_mass = kwargs["vehicle_mass"]
        self.fuel_capacity = kwargs["fuel_capacity"]
        self.brake_deadband = kwargs["brake_deadband"]
        self.decel_limit = kwargs["decel_limit"]
        self.accel_limit = kwargs["accel_limit"]
        self.wheel_radius = kwargs["wheel_radius"]
        self.wheel_base = kwargs["wheel_base"]
        self.steer_ratio = kwargs["steer_ratio"]
        self.max_lat_accel = kwargs["max_lat_accel"]
        self.max_steer_angle = kwargs["max_steer_angle"]

        self.last_timestamp = None

        # calculate brake torque
        self.torque_constant = (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.wheel_radius

        # PID controller
        pid_P = 0.7
        pid_I = 0.00015
        pid_D = 0.009
        self.pid_velocity = PID(pid_P, pid_I, pid_D, self.decel_limit, self.accel_limit)

        # Yaw controller
        self.yaw_controller = YawController(wheel_base = self.wheel_base,
                                            steer_ratio = self.steer_ratio,
                                            min_speed = 0.5,
                                            max_lat_accel = self.max_lat_accel,
                                            max_steer_angle = self.max_steer_angle)


    def control(self, cur_lin_vel, req_lin_vel, req_ang_vel):
        throttle, brake, steering = 0.0, 0.0, 0.0

        # Timing
        if self.last_timestamp is None:
            self.last_timestamp = rospy.get_time()
            return throttle, brake, steering

        timestamp = rospy.get_time()
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        # for car not to toggle itself at TL, reset PID and keep brake pushed
        if abs(req_lin_vel) < 0.5:
            self.pid_velocity.reset()
            return throttle, 100, steering

        desired_accel = self.pid_velocity.step(req_lin_vel - cur_lin_vel, dt)
        steering = self.yaw_controller.get_steering(req_lin_vel, req_ang_vel, cur_lin_vel)

        if desired_accel > 0.0:
            brake = 0.0
            throttle = desired_accel
        else:
            throttle = 0.0
            if abs(desired_accel) > self.brake_deadband:
                # let the car stop by itself if possible
                brake = abs(desired_accel) * self.torque_constant

        return throttle, brake, steering

    def reset(self):
        self.pid_velocity.reset()
