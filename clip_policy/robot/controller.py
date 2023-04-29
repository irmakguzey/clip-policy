import rospy
from sensor_msgs.msg import Image as Image_msg


import rospy


from std_msgs.msg import Float64MultiArray, Int64


import torch
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R
from utils.action_transforms import *
import numpy as np
import cv2
from PIL import Image
import os

from clip_policy.robot.publisher import PolicyPublisher
from clip_policy.robot.subscriber import ImageSubscriber


class Controller:
    def __init__(self, cfg=None):
        self.cfg = cfg

        self.publisher = PolicyPublisher(cfg=cfg)
        self.subscriber = ImageSubscriber(cfg=cfg)

        self.subscriber.register_for_uid(self.publisher)
        self.subscriber.register_for_uid(self)

        self.img_transforms = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.h = 0.91

    def setup_model(self, model):
        self.model = model
        self.model.to(self.device)

    def img_to_tensor(self, img):
        if type(img) is np.ndarray:
            # convert cv2 image to PIL image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        img_tensor = self.img_transforms(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def action_tensor_to_matrix(self, action_tensor):
        affine = np.eye(4)
        r = R.from_euler("xyz", action_tensor[3:6], degrees=False)
        affine[:3, :3] = r.as_matrix()
        affine[:3, -1] = action_tensor[:3]

        return affine

    def matrix_to_action_tensor(self, matrix):
        r = R.from_matrix(matrix[:3, :3])
        action_tensor = np.concatenate(
            (matrix[:3, -1], r.as_euler("xyz", degrees=False))
        )
        return action_tensor

    def cam_to_robot_frame(self, matrix):
        return invert_permutation_transform(matrix)

    # normal = 0.07
    # drawer corridor max_h = 0.055, h = 0.65
    # door kitchen cabinet h = 0.52
    # drawer closing kitchen h = 0.71, max_h = 0.055
    # door closing kitchen h = 0.91, max_h = 0.055

    def schedule_init(self, sched_no, max_h=0.055, max_base=0.08):
        if sched_no <= 2:
            base = -max_base
            if sched_no == 1:
                h = self.h + max_h * 2 / 3
            else:
                h = self.h - max_h * 2 / 3
        elif sched_no <= 5:
            base = -max_base / 3
            if sched_no == 3:
                h = self.h + max_h
            elif sched_no == 4:
                h = self.h + max_h / 3.5
            else:
                h = self.h - max_h
        elif sched_no <= 8:
            base = max_base / 3
            if sched_no == 6:
                h = self.h + max_h
            elif sched_no == 7:
                h = self.h - max_h / 3.5
            else:
                h = self.h - max_h
        elif sched_no <= 10:
            base = max_base
            if sched_no == 9:
                h = self.h + max_h * 2 / 3
            else:
                h = self.h - max_h * 2 / 3

        return base, h

    def _process_instruction(self, instruction):
        if instruction == "h" or instruction == "H":
            self.publisher.publish("home_publisher", [1])
            self.run_for -= 1
            return None

        elif instruction == "r" or instruction == "R":
            h = input("Enter height:")

            self.publisher.publish(
                "home_params_publisher", [float(h), 0.02, 0.0, 0.0, 0.07, 0.03, 1.0]
            )
            self.run_for -= 1
            return None

        elif instruction == "s" or instruction == "S":
            sched_no = input("Enter schedule number:")
            base, h = self.schedule_init(int(sched_no))
            print(h, base)
            self.publisher.publish(
                "home_params_publisher", [h, 0.02, base, 0.0, 0.07, 0.03, 1.0]
            )

            self.run_for -= 1
            return None

        elif instruction.isdigit():
            self.run_for = int(instruction)
            return "-"

        return instruction

    def run(self):
        rate = rospy.Rate(10)

        self.run_for = 0
        traj_pos = 0
        while True:
            rate.sleep()

            img = self.subscriber.get_image()

            if self.run_for == 0:
                instruction = input("Enter instruction:")
                self.run_for = 1

            instruction = self._process_instruction(instruction)

            if instruction is None:
                continue

            self.run_for -= 1

            if self.cfg is not None and self.cfg["save_img"]:
                os.makedirs(self.cfg["save_img_path"], exist_ok=True)
                img_small = cv2.resize(img, (224, 224))
                cv2.imwrite(
                    os.path.join(
                        self.cfg["save_img_path"], "img_{}.png".format(traj_pos)
                    ),
                    img_small,
                )

            # TODO: Maybe add the following code to model or seperate interface for more flexibility

            if self.cfg["model_type"] == "vinn" and self.cfg["save_nbhrs"]:
                action_tensor, indices = self.model.get_action(
                    self.img_to_tensor(img), return_indices=True
                )

                action_tensor = action_tensor.cpu().detach()
                # print(indices.shape)
                for idx in indices[0][:3]:
                    # print("Neighbour:", idx)
                    img_nbhr = self.model.imgs[idx]
                    # img_nbhr of shape 1, 3, 224, 224 and type numpy
                    img_nbhr_unnorm = img_nbhr.squeeze(0).transpose(1, 2, 0) * np.array(
                        [0.229, 0.224, 0.225]
                    ) + np.array([0.485, 0.456, 0.406])
                    img_nbhr_unnorm = img_nbhr_unnorm * 255

                    # BGR to RGB
                    img_s = img_nbhr_unnorm[:, :, ::-1]
                    # print(img_s.shape)
                    # save numpy img to folder ./nbhrs. Make sure to create the folder first
                    os.makedirs("./nbhrs", exist_ok=True)
                    cv2.imwrite(
                        os.path.join("./nbhrs", "img_{}_{}.png".format(traj_pos, idx)),
                        img_s,
                    )

            action_tensor = (
                self.model.get_action(self.img_to_tensor(img)).cpu().detach()
            )

            #############

            action_matrix = self.action_tensor_to_matrix(action_tensor)
            action_robot_matrix = self.cam_to_robot_frame(action_matrix)
            action_robot = self.matrix_to_action_tensor(action_robot_matrix)

            gripper = action_tensor[-1]
            print("Gripper:", gripper)
            print("Action:", action_robot)

            self.publisher.publish_action(action_robot, gripper)

            traj_pos += 1
