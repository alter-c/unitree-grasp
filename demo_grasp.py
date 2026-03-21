import time
import sys
import argparse


from action_executor import ActionExecutor
from yolo_detector import YOLODetector

def parse_arg():
    parser = argparse.ArgumentParser(description="use sdk to grasp")
    parser.add_argument(
        'target',
        type=str,
        help='target object class to grasp'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    executor = ActionExecutor()
    detector = YOLODetector("./models/yolov8s-seg.pt", False)
    detector.start()

    try:    
        args = parse_arg()
        target = args.target
        if target is None:
            print("[Unitree] Grasp failed: No target provided")
            raise

        # Step 1: Detect objects
        while True:
            detection = detector.get_interested_detection(target)
            if detection:
                coords = detection["world"]
                if coords[0] > 1.0:
                    print(f"Detected object is too far for grasp.")
                    continue
                else:
                    break
            time.sleep(1)

        # Step 2: Move to expected distance
        cur_dis = coords[0]
        expect_dis = 0.4
        if cur_dis > expect_dis:
            executor.move_forward(cur_dis-expect_dis+0.1)
        else:
            print("Already within expected distance.")

        # Step 3: Execute grasping action
        while True:
            detection = detector.get_interested_detection(target)
            if detection:
                coords = detection["world"]
                if coords[0] > 1.0:
                    print(f"Detected object is too far for grasp.")
                    continue
                else:
                    break
            time.sleep(1)

        suc = executor.grasp(coords)
        if suc:
            print(f"[Unitree] Grasp success: Use {executor.hand_ctrl.object_hand} hand")
        else:
            print("[Unitree] Grasp failed")

    except Exception as e:
        print(f"[Unitree] Grasp failed: An error occurred: {e}")

    finally:
        detector.stop()
