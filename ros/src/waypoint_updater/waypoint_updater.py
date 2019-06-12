#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32, Float32

import math
import numpy as np
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        rospy.Subscriber('/traffic_waypoint',  Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        self.publisher = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.final_waypoints = None
        self.traffic_waypoint = -1

        self.state = 0 # 0 for move, 1 for stop

        self.breaking_acceleration = None
        self.breaking_acceleration_limit = rospy.get_param('~decel_limit', -5)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()

    def pose_cb(self, msg): self.current_pose = msg
    def waypoints_cb(self, waypoints): self.base_waypoints = waypoints
    def traffic_cb(self, msg): self.traffic_waypoint = msg.data
    def current_velocity_cb(self, msg): self.current_velocity = msg.twist.linear.x

    def dl(self, a, b): return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
    def position(self, pose): return pose.pose.position
    def orientation(self, pose): return pose.pose.orientation

    def update(self):
        """
        Main activity
        """
        # terminate if we don't yet have start messages
        if not (hasattr(self, 'current_pose') and hasattr(self, 'base_waypoints')):
            return

        next_wp = self.get_next_waypoint()

        self.resolve_traffic_lights(next_wp)

        # calculate next waypoints
        self.calculate_final_waypoints(next_wp)

        # publish waypoints
        msg = Lane()
        msg.header.stamp = rospy.Time().now()
        msg.header.frame_id = 'world'
        msg.waypoints = self.final_waypoints[:LOOKAHEAD_WPS]
        self.publisher.publish(msg)

    def resolve_traffic_lights(self, next_wp):
        """
        Resolve what to do based on traffic lights 
        (either stop with decekeration or move)
        """
        # if car is moving
        if self.state == 0:
            # if there's a traffic light in sight
            if self.traffic_waypoint != -1:
                decel = lambda x: abs(self.current_velocity**2 / (2*x))

                traffic_light_distance = self.distance(self.base_waypoints.waypoints, next_wp, self.traffic_waypoint)
                min_distance = decel(self.breaking_acceleration_limit)

                if traffic_light_distance > min_distance:
                    self.state = 1
                    self.breaking_acceleration = decel(traffic_light_distance)
                else: 
                    self.state = 0
            else: 
                self.state = 0

        # if car is stopped but TL light is not red, switch to moving
        elif self.traffic_waypoint == -1: 
            self.state = 0

    def calculate_final_waypoints(self, start_wp):
        """
        Calculate the final waypoints to follow
        """

        self.final_waypoints = []

        if self.state == 0:
            for i in range(start_wp, start_wp + LOOKAHEAD_WPS):
                j = i % len(self.base_waypoints.waypoints)
                tmp = Waypoint()
                tmp.pose.pose = self.base_waypoints.waypoints[j].pose.pose
                tmp.twist.twist.linear.x = self.base_waypoints.waypoints[j].twist.twist.linear.x
                self.final_waypoints.append(tmp)

        elif self.state == 1:

            # waypoints before TL - base_waypoint's pose and speed
            for i in range(start_wp, self.traffic_waypoint):
                j = i % len(self.base_waypoints.waypoints)
                tmp = Waypoint()
                tmp.pose.pose = self.base_waypoints.waypoints[j].pose.pose
                tmp.twist.twist.linear.x = self.base_waypoints.waypoints[j].twist.twist.linear.x
                self.final_waypoints.append(tmp)

            # Brake to target
            target_wp = len(self.final_waypoints)

            # waypoints after TL -> base_waypoint's pose, speed = 0
            i_max = max(start_wp + LOOKAHEAD_WPS, self.traffic_waypoint + 1)
            for i in range(self.traffic_waypoint, i_max):
                j = i % len(self.base_waypoints.waypoints)
                tmp = Waypoint()
                tmp.pose.pose = self.base_waypoints.waypoints[j].pose.pose
                tmp.twist.twist.linear.x  = 0.0
                self.final_waypoints.append(tmp)

            # set speed to zero for last waypoint before TL
            last = self.final_waypoints[target_wp]
            last.twist.twist.linear.x = 0.0

            # Add deceleration to waypoints for braking smoothly
            for wp in self.final_waypoints[:target_wp][::-1]:
                x = wp.pose.pose.position.x - last.pose.pose.position.x
                y = wp.pose.pose.position.y - last.pose.pose.position.y
                z = wp.pose.pose.position.z - last.pose.pose.position.z
                dist = math.sqrt(x*x + y*y + z*z)
                vel = math.sqrt(2 * self.breaking_acceleration * max(0.0, dist - 5))
                if vel < 1.0: vel = 0.0
                wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)

    def get_closest_waypoint(self):
        """
        get waypoint closest to the current pose
        """
        min_dist = float('inf')
        wp = 0
        for i in range(len(self.base_waypoints.waypoints)):
            dist = self.dl(self.position(self.current_pose), self.position(self.base_waypoints.waypoints[i].pose))
            if dist < min_dist:
                min_dist = dist
                wp = i
        return wp

    def get_next_waypoint(self):
        """
        get waypoint ahead of the current pose
        """
        next_wp = self.get_closest_waypoint()
        self.position(self.current_pose)

        next_pose_position = self.position(self.base_waypoints.waypoints[next_wp].pose)
        cur_pose_position = self.position(self.current_pose)
        cur_pose_orient = self.orientation(self.current_pose)

        heading = math.atan2((next_pose_position.y - cur_pose_position.y), (next_pose_position.x - cur_pose_position.x))
        theta = tf.transformations.euler_from_quaternion([cur_pose_orient.x, cur_pose_orient.y, cur_pose_orient.z, cur_pose_orient.w])[-1]
        angle = math.fabs(theta - heading)

        return next_wp if angle <= math.pi / 4.0 else next_wp + 1
    
    def distance(self, waypoints, wp1, wp2):
        """
        Return distance between two waypoints
        """
        dist = 0
        for i in range(wp1, wp2 + 1):
            dist += self.dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
