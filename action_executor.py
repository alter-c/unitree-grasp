import sys
import time
import math
import numpy as np
import pinocchio as pin
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from control.g1_arm_sdk import Custom
from control.g1_arm_ik import G1_29_ArmIK
from control.dex_hand_sdk import Dex3_1_DirectController

from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


class ActionExecutor:
    def __init__(self):
        self.hand_ctrl = Dex3_1_DirectController()
        self.arm_ctrl = Custom()
        self.arm_ik_solver = G1_29_ArmIK()
        
        self.arm_ctrl.Init()

        ChannelFactoryInitialize(0, "eth0")
        self.sport_client = LocoClient()  
        self.sport_client.SetTimeout(5.0)
        self.sport_client.Init()

        self.kPi = math.pi

    def _arm_pos_control(self, target_left_pos, target_right_pos):
        """
        pos format: [x, y, z, roll, pitch, yaw]
        """
        L_tf_target = pin.SE3(
            pin.utils.rpyToMatrix(np.array(target_left_pos[3:])),
            np.array(target_left_pos[:3]),
        )
        R_tf_target = pin.SE3(
            pin.utils.rpyToMatrix(np.array(target_right_pos[3:])),
            np.array(target_right_pos[:3]),
        )
        try:
            sol_q, sol_tauff = self.arm_ik_solver.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
            target_q = sol_q.tolist()[0:14] + [0]*3
            self.arm_ctrl.Control(target_q)
        except Exception as e:
            print("[ActionExecutor] Arm pos control error:", e)

    def _single_arm_pos_control(self, target_pos, arm_flag):
        """
        pos format: [x, y, z, roll, pitch, yaw]
        arm flag: 'left' or 'right'
        """
        if arm_flag == 'left':
            flag = 1
        elif arm_flag == 'right':
            flag = -1
        else:
            print("Invalid arm. Use 'left' or 'right'.")
            return

        tf_target = pin.SE3(
            pin.utils.rpyToMatrix(np.array(target_pos[3:])),
            np.array(target_pos[:3]),
        )
        tf_another = pin.SE3(
            pin.utils.rpyToMatrix(np.array([0, 0, 0])),
            np.array([0.25, 0.25*flag, 0.1]),
        )
        q_another = [0, -self.kPi/9*flag, 0, self.kPi/2, 0, 0, 0, ]

        try:
            if flag == 1:
                sol_q, sol_tauff = self.arm_ik_solver.solve_ik(tf_target.homogeneous, tf_another.homogeneous)
                target_q = sol_q.tolist()[0:7] + q_another + [0]*3
            else:
                sol_q, sol_tauff = self.arm_ik_solver.solve_ik(tf_another.homogeneous, tf_target.homogeneous)
                target_q = q_another + sol_q.tolist()[7:14] + [0]*3
            self.arm_ctrl.Control(target_q)
        except Exception as e:
            print("[ActionExecutor] Single arm pos control error:", e)

    def _arm_joint_control(self, target_q):
        """
        q format: [q1, q2, ..., q14] without waist
        """
        try:
            target_q_full = target_q + [0]*3
            self.arm_ctrl.Control(target_q_full)
        except Exception as e:
            print("[ActionExecutor] Arm joint control error:", e)

    def move_forward(self, distance, speed=0.3):
        try:
            duration = distance / speed
            self.sport_client.SetVelocity(speed, 0, 0, duration)
            time.sleep(duration+2) # wait for the movement to complete
        except Exception as e:
            print("[ActionExecutor] Move forward error:", e)

    def grasp(self, target_coords):
        """
        coords format: [x, y, z]
        """
        if target_coords is None:
            print("[ActionExecutor] Grasp target is None!")
            return False
        elif target_coords[0] < 0:
            print("[ActionExecutor] Grasp target is negative in x direction!")
            return False
        elif target_coords[0] < 0.1:
            print("[ActionExecutor] Grasp target is too close!")
            return False
        elif target_coords[0] > 0.45:
            print("[ActionExecutor] Grasp target is too far!")
            return False

        arm_flag = "left" if target_coords[1] > 0 else "right"
        flag = 1 if arm_flag == "left" else -1
        
        try:
            print("[ActionExecutor] Grasping at coords: ", target_coords, " with", arm_flag, "arm.")

            # [Stage 1] Move arm to pre-grasp position and open hand
            pre_pos = [
                0.1, 0.25*flag, target_coords[2]+0.1,
                0.0, 0.0, 0.0
            ] 
            self._single_arm_pos_control(pre_pos, arm_flag)
            self.hand_ctrl.open_hand(arm_flag)

            # [Stage 2] Move arm to some mid pos
            mid_pos_1 = [
                (0.1+target_coords[0])/2, 0.25*flag, target_coords[2]+0.1,
                0.0, 0.0, 0.0
            ]
            self._single_arm_pos_control(mid_pos_1, arm_flag)

            # mid_pos_2 = [
            #     target_coords[0], target_coords[1]+0.05*flag, target_coords[2]+0.1,
            #     0.0, 0.0, 0.0
            # ]
            # self._single_arm_pos_control(mid_pos_2, arm_flag)

            # [Stage 3] Move arm to grasp position and close hand
            grasp_pos = [
                target_coords[0]+0.05, target_coords[1], target_coords[2]+0.05,
                0.0, 0.0, 0.0
            ]
            self._single_arm_pos_control(grasp_pos, arm_flag)
            self.hand_ctrl.close_hand(arm_flag)

            print("[ActionExecutor] Grasp completed.")
            return True

        except Exception as e:
            print("[ActionExecutor] Grasp error:", e)
            return False

    def retract(self, arm_flag):
        if arm_flag == 'left':
            flag = 1
        elif arm_flag == 'right':
            flag = -1
        else:
            print("Invalid arm. Use 'left' or 'right'.")
            return
        
        try:
            print("[ActionExecutor] Retracting", arm_flag, "arm.")

            # [Stage 1] Close hand and move arm to mid pos
            self.hand_ctrl.close_hand(arm_flag)
            mid_pos = [
                0.3, 0.25*flag, 0.25,
                0.0, 0.0, 0.0
            ]
            self._single_arm_pos_control(mid_pos, arm_flag)

            # [Stage 2] Move arm to prepare position
            pre_pos = [
                0.05, 0.3*flag, 0.15,
                0.0, 0.0, 0.0
            ] 
            self._single_arm_pos_control(pre_pos, arm_flag)

            # [Stage 3] Put arm down and release hand and arm
            self.arm_ctrl.Release()

            print("[ActionExecutor] Retract completed.")

        except Exception as e:
            print("[ActionExecutor] Retract error:", e)

    def release(self):
        self.hand_ctrl.release_hand()
        self.arm_ctrl.Release()

        
if __name__ == "__main__":
    executor = ActionExecutor()
    while True:
        executor.move_forward(0.5)
        target_coords = [0.35, -0.1, 0.1]
        executor.grasp(target_coords)
        executor.retract("right")
        input()


