import time
import sys
import argparse

from action_executor import ActionExecutor
from yolo_detector import YOLODetector

def parse_arg():
    parser = argparse.ArgumentParser(description="use sdk to handover")
    parser.add_argument(
        'hand',
        type=str,
        help='which hand to handover, left or right'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    executor = ActionExecutor()
    detector = YOLODetector("./models/yolov8s-seg.pt", False)
    detector.start()

    try:
        args = parse_arg()
        hand_flag = args.hand
        
        # Step 1: Detect person
        while True:
            detection = detector.get_interested_detection("person")
            if detection:
                coords = detection["world"]
                if coords[0] > 3.0:
                    print(f"Detected huamn is too far for handover.")
                    continue
                else:
                    break
            time.sleep(1)

        # Step 2: Move to expected distance
        cur_dis = coords[0]
        expect_dis = 0.6
        if cur_dis > expect_dis:
            executor.move_forward(cur_dis-expect_dis)
        else:
            print("Already within expected distance.")

        # Step 3: Execute handover action
        suc = executor.hand_over(hand_flag)
        if suc:
            print(f"[Unitree] Handover success")
        else:
            print("[Unitree] Handover failed")
            
    except Exception as e:
        print(f"[Unitree] Handover failed: An error occurred: {e}")
