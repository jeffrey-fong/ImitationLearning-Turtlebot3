#!/usr/bin/env python

# This script receives a goal pose, plans the path and executes it.

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal



def movebase_client():

    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = 6.0
    goal.target_pose.pose.position.y = -1.0
    goal.target_pose.pose.orientation.w = 1.0

    client.send_goal(goal)
    rospy.sleep(2.0)
    client.cancel_goal()
    '''wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        return client.get_result()'''


if __name__ == '__main__':
    try:
        rospy.init_node('movebase_client_py')
        result = movebase_client()
        if result:
            rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")


'''import rospy
from geometry_msgs.msg import Twist, Pose 
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import time


def moveToGoal():
	rospy.init_node('move_forward')
	pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
	rospy.sleep(0.1)
	move = Twist()
	move.linear.x, move.angular.z = 0.1, 0

	start = time.time()
	while time.time() - start < 5.0:
		pub.publish(move)
	move.linear.x, move.angular.z = 0, 0
	pub.publish(move)





if __name__=='__main__':
	#resetRobotPosition()
	moveForward()


# x: 6, y: -1'''