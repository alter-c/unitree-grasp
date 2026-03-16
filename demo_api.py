import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from action_executor import ActionExecutor
from yolo_detector import YOLODetector


executor = ActionExecutor()
detector = YOLODetector("./models/yolov8s-seg.pt", True)
detector.start()

# ---------------- FastAPI App ----------------
app = FastAPI()

@app.get("/api/unitree/grasp")
def unitree_grasp(target: str=None):
    try:
        if target is None:
            return JSONResponse(status_code=500, content={"error": "No target provided."})

        # Step 1: Detect objects
        while True:
            detection = detector.get_interested_detection(target)
            if detection:
                coords = detection["world"]
                break
            time.sleep(1)

        input("Check and Press any key to continue.")

        # Step 2: Move to expected distance
        cur_dis = coords[0]
        expect_dis = 0.4
        if cur_dis > expect_dis:
            executor.move_forward(cur_dis-expect_dis)
        else:
            print("Already within expected distance.")

        # Step 3: Execute grasping action
        while True:
            # Refresh detection to get updated coords
            detection = detector.get_interested_detection(target)
            if detection:
                coords = detection["world"]
                break
            time.sleep(1)

        input("Check and Press any key to continue.")
        suc = executor.grasp(coords)

        if suc:
            # Step 4: Execute other action
            arm_flag = "left" if coords[1] > 0 else "right"
            # executor.hand_ctrl.open_hand(arm_flag) # here just open hand

            # Step 5: Retract arm
            executor.retract(arm_flag)
            return JSONResponse(status_code=200, content={"message": "Grasp success."})
        else:
            print("Grasping failed.")
            return JSONResponse(status_code=500, content={"error": "Grasp failed."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/unitree/release")
def unitree_release():
    try:
        executor.release()
        return JSONResponse(status_code=200, content={"message": "Release success."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        executor.release() 
        detector.stop()