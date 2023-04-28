import rospy

from std_msgs.msg import Float64MultiArray


IMAGE_SUBSCRIBER_TOPIC = "/gopro_image"
DEPTH_SUBSCRIBER_TOPIC = "/depth_image"

TRANSLATIONAL_PUBLISHER_TOPIC = "/translation_tensor"
ROTATIONAL_PUBLISHER_TOPIC = "/rotational_tensor"
GRIPPER_PUBLISHER_TOPIC = "/gripper_tensor"
HOME_PUBLISHER_TOPIC = "/home_tensor"
HOME_PARAMS_TOPIC = "/home_params"


class PolicyPublisher:
    def __init__(self, cfg=None):
        self.cfg = cfg

        self.translational_publisher = rospy.Publisher(
            TRANSLATIONAL_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1
        )
        self.rotational_publisher = rospy.Publisher(
            ROTATIONAL_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1
        )
        self.gripper_publisher = rospy.Publisher(
            GRIPPER_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1
        )
        self.home_publisher = rospy.Publisher(
            HOME_PUBLISHER_TOPIC, Float64MultiArray, queue_size=1
        )
        self.home_params_publisher = rospy.Publisher(
            HOME_PARAMS_TOPIC, Float64MultiArray, queue_size=1
        )

    def publish(self, channel, data):
        publisher = getattr(self, channel)
        data_list = Float64MultiArray()
        data_list.layout.data_offset = self.uid
        data_list.data = data
        publisher.publish(data_list)

    def publish_action(self, action_robot, gripper):
        translation_tensor = action_robot[:3]
        rotation_tensor = action_robot[3:6]
        gripper_tensor = [gripper.item()]
        translation_publisher_list = Float64MultiArray()
        translation_publisher_list.layout.data_offset = self.uid
        translation_publisher_list.data = translation_tensor

        rotation_publisher_list = Float64MultiArray()
        rotation_publisher_list.layout.data_offset = self.uid
        rotation_publisher_list.data = rotation_tensor

        gripper_publisher_list = Float64MultiArray()
        gripper_publisher_list.layout.data_offset = self.uid
        gripper_publisher_list.data = gripper_tensor

        self.translational_publisher.publish(translation_publisher_list)
        self.rotational_publisher.publish(rotation_publisher_list)
        self.gripper_publisher.publish(gripper_publisher_list)
