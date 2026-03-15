import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from action_executor import ActionExecutor
from yolo_detector import YOLODetector

interested_classes = ["bottle"]

executor = ActionExecutor()
detector = YOLODetector("./models/yolov8s-seg.pt", False)
detector.start()


# ---------------- FastAPI App ----------------
app = FastAPI()

@app.get("/api/unitree/grasp")
def unitree_grasp():
    try:
        while True:
            # Step 1: Detect objects
            detections = detector.get_latest_detection()

            # Step 2: Filter detections for interested classes
            coords = None
            for detection in detections:
                if detection["class"] in interested_classes:
                    if detection["world"][0] > 1: # invalid detection, skip
                        continue
                    coords = detection["world"]
                    print(f"Detected {detection['class']} at coords: {coords}.")
            if coords is not None:
                break
            else:
                print("No interested objects detected. Retrying...")
                time.sleep(1)

        input("Check and Press any key to continue.")

        # Step 3: Execute grasping action
        suc = executor.grasp(coords)

        if suc:
            # Step 4: Execute other action
            arm_flag = "left" if coords[1] > 0 else "right"
            executor.hand_ctrl.open_hand(arm_flag) # here just open hand

            # Step 5: Retract arm
            executor.retract(arm_flag)
            return JSONResponse(status_code=200, content={"message": "Grasp success."})
        else:
            print("Grasping failed. Retrying...")
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
        executor.release() 
        detector.stop()