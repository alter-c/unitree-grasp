import time

from action_executor import ActionExecutor
from yolo_detector import YOLODetector

executor = ActionExecutor()
detector = YOLODetector("./models/yolov8s-seg.pt", True)
detector.start()

interested_classes = ["bottle"]

if __name__ == "__main__":
    try:
        while True:
            while True:
                # Step 1: Detect objects
                detections = detector.get_latest_detection()

                # Step 2: Filter detections for interested classes
                coords = None
                for detection in detections:
                    if detection["class"] in interested_classes:
                        coords = detection["world"]
                        print(f"Detected {detection['class']}.")
                if coords is not None:
                    break
                else:
                    print("No interested objects detected. Retrying...")
                    time.sleep(1)

            input("Press any key to continue.")

            # Step 3: Execute grasping action
            executor.grasp(coords)

            # Step 4: Execute other action
            arm_flag = "left" if coords[1] > 0 else "right"
            executor.hand_ctrl.open_hand(arm_flag) # here just open hand

            # Step 5: Retract arm
            executor.retract(arm_flag)

            user_input = input("Press 'n' to grasp next object, otherwise exit.")
            if user_input.lower() != 'n':
                break

    except Exception as e:
        user_input = input("Demo interrupted. Press 'r' to release arm.")
        if user_input.lower() == 'r':
            executor.release()
    finally:
        detector.stop()
