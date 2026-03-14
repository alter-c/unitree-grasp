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
    # Fisrt pose
    target_joint_spread = [
            0.0, kPi/2.5, 0.0, kPi/2, 0.0, 0.0, 0.0, 
            0.0, -kPi/10, 0.0, kPi/2, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0
        ]
    custom.Control(target_joint_spread)
    #input("First position reached. Press Enter to continue...")


    hand_ctrl.open_hand("left")

    # Target pose
    interested_classes = ["bottle"]
    pos =None
    while True:
        res = detector.get_latest_detection()
        for item in res:
            if item['class'] in interested_classes:
                print(item)
                pos = item["world"]
                print(f"Detected bottle")
        if pos is not None:
            break
    pos[2] += 0
    L_tf_target = pin.SE3(
        pin.utils.rpyToMatrix(np.array([0, 0, 0])),
        np.array(pos),
    )
    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)
    print(type(sol_q))

    input()
    target_pos_ik = sol_q.tolist()[0:7] + [
            0.0, -kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0, 
        ] + [
            0.0, 0.0, 0.0
        ]
    tmp_1 = target_pos_ik[1]
    tmp_3 = target_pos_ik[3]
    target_pos_ik[1] = kPi/3
    target_pos_ik[3] = kPi/6    
    custom.Control(target_pos_ik)
    target_pos_ik[1] = tmp_1
    custom.Control(target_pos_ik)
    target_pos_ik[3] = tmp_3
    custom.Control(target_pos_ik)

    # 第四个关节


    hand_ctrl.close_hand("left")
    #input("IK position reached. Press Enter to continue...")


    # Final pose
    target_joint_prepare = [
            0.0, kPi/3, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, -kPi/9, 0.0, kPi/2, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0
        ]
    custom.Control(target_joint_prepare)
    hand_ctrl.open_hand("left")
    time.sleep(2)
    hand_ctrl.close_hand("left")
    hand_ctrl.release_hand()
    input("Final position reached. Press Enter to continue...")

    # ----------------- Release ----------------
    custom.Release()
    detector.stop()