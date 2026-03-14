import sys
import math
import numpy as np
import pinocchio as pin
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from control.g1_arm_sdk import Custom
from control.g1_arm_ik import G1_29_ArmIK
from control.dex_hand_sdk import Dex3_1_DirectController

from yolo_detector import YOLODetector
import time

kPi = math.pi


if __name__ == "__main__":
    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    # ---------------- Initialize ----------------
    custom = Custom()
    custom.Init()

    arm_ik = G1_29_ArmIK()

    detector = YOLODetector("./models/yolov8s-seg.pt", True)
    detector.start()

    hand_ctrl = Dex3_1_DirectController()

    # ---------------- Control ----------------
    # Target pose
    interested_classes = ["bottle"]
    pos = None
    while True:
        res = detector.get_latest_detection()
        for item in res:
            if item['class'] in interested_classes:
                print(item)
                pos = item["world"]
                print(f"Detected bottle")
        if pos is not None:
            break

    # Fisrt pose
    L_tf_target = pin.SE3(
        pin.utils.rpyToMatrix(np.array([0, 0, 0])),
        np.array([0.25, 0.25, 0.1]),
    )
    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.1, -0.2, pos[2]+0.1]),
    )    
    sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
    first_pos_ik =  [
            0.0, kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0, 
        ] + sol_q.tolist()[7:14] + [
            0.0, 0.0, 0.0
        ]
    custom.Control(first_pos_ik)
    print(sol_q)
    #input("First position reached. Press Enter to continue...")
    hand_ctrl.open_hand("right")

    # Second pose
    L_tf_target = pin.SE3(
        pin.utils.rpyToMatrix(np.array([0, 0, 0])),
        np.array([0.25, 0.25, 0.1]),
    )
    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([(0.1+pos[0])/2, -0.2, pos[2]+0.1]),
    )    
    sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
    target_pos_ik =  [
            0.0, kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0, 
        ] + sol_q.tolist()[7:14] + [
            0.0, 0.0, 0.0
        ]
    custom.Control(target_pos_ik)

    # Third pose
    L_tf_target = pin.SE3(
        pin.utils.rpyToMatrix(np.array([0, 0, 0])),
        np.array([0.25, 0.25, 0.1]),
    )
    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([pos[0], pos[1]-0.1, pos[2]+0.15]),
    )    
    sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
    target_pos_ik =  [
            0.0, kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0, 
        ] + sol_q.tolist()[7:14] + [
            0.0, 0.0, 0.0
        ]
    custom.Control(target_pos_ik)

    # new_arm_ik = G1_29_ArmIK()
    # Target pose for IK
    L_tf_target = pin.SE3(
        pin.utils.rpyToMatrix(np.array([0, 0, 0])),
        np.array([0.25, 0.25, 0.1]),
    )
    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([pos[0]+0.05, pos[1]+0.05, pos[2]+0.05]),
    )

    sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)

    # input()
    target_pos_ik = [
            0.0, kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0, 
        ] + sol_q.tolist()[7:14] + [
            0.0, 0.0, 0.0
        ]
    custom.Control(target_pos_ik)

    # 第四个关节


    hand_ctrl.close_hand("right")
    #input("IK position reached. Press Enter to continue...")
    time.sleep(2)
    
    hand_ctrl.open_hand("right")


    # Final pose
    target_joint_prepare = [
            0.0, kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0, 
            0.0, -kPi/2.5, 0.0, kPi/3, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0
        ]
    custom.Control(target_joint_prepare)
    hand_ctrl.open_hand("right")
    time.sleep(2)
    hand_ctrl.close_hand("right")
    hand_ctrl.release_hand()
    # input("Final position reached. Press Enter to continue...")

    # ----------------- Release ----------------
    custom.Release()
    detector.stop()