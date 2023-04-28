import rospy
from sensor_msgs.msg import Image as Image_msg
from cv_bridge import CvBridgeError, CvBridge
from std_msgs.msg import Int64
from copy import copy


IMAGE_SUBSCRIBER_TOPIC = "/gopro_image"
DEPTH_SUBSCRIBER_TOPIC = "/depth_image"


PING_TOPIC = "run_model_ping"


class ImageSubscriber:
    def __init__(self, cfg=None):
        # Initializing a rosnode
        try:
            rospy.init_node("image_subscriber")
        except:
            pass

        self.cfg = cfg

        self.bridge = CvBridge()

        # Getting images from the rostopic
        self.image = None

        # Subscriber for images
        rospy.Subscriber(
            IMAGE_SUBSCRIBER_TOPIC, Image_msg, self._callback_image, queue_size=1
        )

        # rospy.Subscriber(DEPTH_SUBSCRIBER_TOPIC, Image_msg, self._callback_depth, queue_size=1)
        rospy.Subscriber(PING_TOPIC, Int64, self._callback_ping, queue_size=1)
        self.uid = -1
        self.prev_uid = -1
        self._registered_objects = []

    def register_for_uid(self, obj):
        self._registered_objects.append(obj)

    def _callback_image(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # print("Getting images")
        except CvBridgeError as e:
            print(e)

    def _callback_ping(self, data):
        self.uid = int(data.data)
        for obj in self._registered_objects:
            obj.uid = self.uid
        # print('Received uid {}'.format(self.uid))

    def _wait_for_image_ping(self):
        while self.image is None or self.uid == self.prev_uid:
            pass

    def get_image(self):
        self._wait_for_image_ping()
        self.prev_uid = copy(self.uid)
        return self.image
