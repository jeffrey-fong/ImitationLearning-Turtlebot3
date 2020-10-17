#!/usr/bin/env python

# This script resets the robot's position to A in both Gazebo and Rviz.
# You may wish to add some noise to the robot position.


import rospy
import tf
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped
from gazebo_msgs.srv import SetModelState, SpawnModel
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Bool


def resetRobotPositionCallback(msg):
	resetRobotPosition()

def resetRobotPosition():
	# Stop velocity
	cmd_msg = Twist()
	cmd_msg.linear.x = 0.0
	cmd_msg.angular.z = 0.0
	pub_cmd.publish(cmd_msg)

	# Reset in Gazebo
	state_msg = ModelState()
	state_msg.model_name = 'turtlebot3'
	state_msg.pose.position.x = -6.503
	state_msg.pose.position.y = -2.422
	state_msg.pose.position.z = 0.00
	state_msg.pose.orientation.x = 0
	state_msg.pose.orientation.y = 0
	state_msg.pose.orientation.z = -0.77
	state_msg.pose.orientation.w = -0.63

	state_msg.twist.linear.x = 0.0
	state_msg.twist.linear.y = 0.0
	state_msg.twist.linear.z = 0.0
	state_msg.twist.angular.x = 0.0
	state_msg.twist.angular.y = 0.0
	state_msg.twist.angular.z = 0.0

	rospy.wait_for_service('/gazebo/set_model_state')
	try:
		set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		resp = set_state( state_msg )

	except:
		print("Service call failed: %s" % e)

	# Reset in rviz
	resetrviz = PoseWithCovarianceStamped()
	resetrviz.header.frame_id = "map"
	resetrviz.header.stamp = rospy.Time.now()
	resetrviz.pose.pose.position.x = -6.503
	resetrviz.pose.pose.position.y = -2.422
	resetrviz.pose.pose.orientation.z = -0.77
	resetrviz.pose.pose.orientation.w = -0.63
	pub_resetrviz.publish(resetrviz)
	br.sendTransform((0, 0, 0),
					tf.transformations.quaternion_from_euler(0, 0, 0),
					rospy.Time.now(), "odom", "map")
	print('reset done')


if __name__=='__main__':
	rospy.init_node('reset_robot_pos')
	br = tf.TransformBroadcaster()
	pub_resetrviz = rospy.Publisher(
							"/initialpose", PoseWithCovarianceStamped, queue_size=10)
	pub_cmd = rospy.Publisher(
							"cmd_vel", Twist, queue_size=10)
	sub_reset = rospy.Subscriber('reset_robot_pos', Bool, resetRobotPositionCallback)
	rospy.sleep(0.8)
	resetRobotPosition()
	rospy.sleep(0.1)
	resetRobotPosition()

	rospy.spin()


# position: 
#       x: -6.503128366844999
#       y: -2.4221654755161817
#       z: -0.0010021019490881675
#     orientation: 
#       x: 0.0029722193889993587
#       y: -0.0024604578835837475
#       z: -0.7690043208045467
#       w: -0.6392319349366429
