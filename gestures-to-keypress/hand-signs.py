#!/usr/bin/env python
#modified from https://github.com/ros-teleop/teleop_twist_keyboard/blob/master/teleop_twist_keyboard.py

from __future__ import print_function

import roslib; roslib.load_manifest('spot_teleop') #
import rospy

from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from champ_msgs.msg import Pose as PoseLite
from geometry_msgs.msg import Pose as Pose
import tf

import sys, select, termios, tty
import numpy as np

class Teleop:
    def __init__(self):
        self.velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
        self.pose_lite_publisher = rospy.Publisher('body_pose/raw', PoseLite, queue_size = 1)
        self.pose_publisher = rospy.Publisher('body_pose', Pose, queue_size = 1)
        self.joy_subscriber = rospy.Subscriber('joy', Joy, self.joy_callback)
        self.swing_height = rospy.get_param("gait/swing_height", 0)
        self.nominal_height = rospy.get_param("gait/nominal_height", 0)

        self.speed = rospy.get_param("~speed", 0.5)
        self.turn = rospy.get_param("~turn", 1.0)

        self.msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
u    i    o
j         l
m    ,    .
For strafing, hold down the shift key:
---------------------------
U    I    O
           
M    <    >

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%
CTRL-C to quit
        """

        self.velocityBindings = {
		#normal
		'u':(1,0,0,1), #forward and left
                'i':(1,0,0,0), #forward 
                'o':(1,0,0,-1), #forward and right
                'j':(0,0,0,1), #turn left 
                'l':(0,0,0,-1), #turn right 
		'm':(-1,0,0,-1), #backwards and left
                ',':(-1,0,0,0), #backwards 
                '.':(-1,0,0,1), #backwards and right
                
		#strafing
		'U':(1,1,0,0), #forward and left
		'I':(1,0,0,0), #forward
                'O':(1,-1,0,0), #forward and right
		'M':(-1,1,0,0), #backwards and left
                '<':(-1,0,0,0), #backwards
                '>':(-1,-1,0,0), #backwards and right
            }


        self.speedBindings={
                'q':(1.1,1.1), #increase all speed
                'z':(.9,.9), #decrease all speed
                'w':(1.1,1), #increase linear speed
                'x':(.9,1), #decrease linear speed
                'e':(1,1.1), #increase angular speed
                'c':(1,.9), #decrease angular speed
            }
        
        self.poll_keys()

    def joy_callback(self, data):
        twist = Twist()
        twist.linear.x = data.axes[1] * self.speed
        twist.linear.y = data.buttons[4] * data.axes[0] * self.speed
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = (not data.buttons[4]) * data.axes[0] * self.turn
        self.velocity_publisher.publish(twist)

        body_pose_lite = PoseLite()
        body_pose_lite.x = 0
        body_pose_lite.y = 0
        body_pose_lite.roll = (not data.buttons[5]) *-data.axes[3] * 0.349066
        body_pose_lite.pitch = data.axes[4] * 0.174533
        body_pose_lite.yaw = data.buttons[5] * data.axes[3] * 0.436332
        if data.axes[5] < 0:
            body_pose_lite.z = data.axes[5] * 0.5

        self.pose_lite_publisher.publish(body_pose_lite)

        body_pose = Pose()
        body_pose.position.z = body_pose_lite.z

        quaternion = tf.transformations.quaternion_from_euler(body_pose_lite.roll, body_pose_lite.pitch, body_pose_lite.yaw)
        body_pose.orientation.x = quaternion[0]
        body_pose.orientation.y = quaternion[1]
        body_pose.orientation.z = quaternion[2]
        body_pose.orientation.w = quaternion[3]

        self.pose_publisher.publish(body_pose)

    def poll_keys(self):
        self.settings = termios.tcgetattr(sys.stdin)

        x = 0
        y = 0
        z = 0
        th = 0
        roll = 0
        pitch = 0
        yaw = 0
        status = 0
        cmd_attempts = 0

        try:
            print(self.msg)
            print(self.vels( self.speed, self.turn))
            
            while not rospy.is_shutdown():
                key = self.getKey()
                if key in self.velocityBindings.keys():
                    x = self.velocityBindings[key][0]
                    y = self.velocityBindings[key][1]
                    z = self.velocityBindings[key][2] 
                    th = self.velocityBindings[key][3]  
                    twist = Twist()
                    twist.linear.x = x *self.speed
                    twist.linear.y = y * self.speed
                    twist.linear.z = z * self.speed
                    twist.angular.x = 0
                    twist.angular.y = 0
                    twist.angular.z = th * self.turn
                    self.velocity_publisher.publish(twist)

                    cmd_attempts += 1
                    
                elif key in self.speedBindings.keys():
                    self.speed = self.speed * self.speedBindings[key][0]
                    self.turn = self.turn * self.speedBindings[key][1]
                    
                    print(self.vels(self.speed, self.turn))
                    if (status == 14):
                        print(self.msg)
                    status = (status + 1) % 15

                else:
                    cmd_attempts = 0
                    if (key == '\x03'):
                        break

        except Exception as e:
            print(e)

        finally:
            twist = Twist()
            twist.linear.x = 0
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = 0
            self.velocity_publisher.publish(twist)

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        
    def getKey(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def vels(self, speed, turn):
        return "currently:\tspeed %s\tturn %s " % (speed,turn)

    def map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

if __name__ == "__main__":
    rospy.init_node('spot_teleop')
    teleop = Teleop()
