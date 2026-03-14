import time
import cv2
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import base64
from multiprocessing import Process, Value, Array, shared_memory
from ultralytics import YOLO

from utils.coordinate_transform import *

class YOLODetector:
    def __init__(self, model_path, visualize=False):
        self.model_path = model_path
        self.visualize = visualize

        # shared memory
        self.color_shm = None
        self.depth_shm = None

        self.width = 640
        self.height = 480
        self.fps = 30

        self.color_shape = (self.height, self.width, 3)
        self.depth_shape = (self.height, self.width)
        self.color_dtype = np.uint8
        self.depth_dtype = np.float32

        # latest detection
        self.result_shm = mp.Manager().list()  # detection results

        # camera info
        self.intr_shm = Array('f', 9)  # intrinsics matrix

        self.capture_proc = None
        self.infer_proc = None
        self.vis_proc = None
        self.is_running = Value('b', False)
        self.capture_ready = mp.Event()
        self.depth_scale = 0.001

    def start(self):
        if self.is_running.value:
            return "Already running."

        # Create shared memory
        self.color_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.color_shape) * self.color_dtype().nbytes)
        self.depth_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.depth_shape) * self.depth_dtype().nbytes)

        self.is_running.value = True

        # Start processes
        self.capture_proc = Process(target=self._capture_loop, daemon=True)
        self.infer_proc = Process(target=self._inference_loop, daemon=True)
        self.capture_proc.start()
        self.infer_proc.start()
        if self.visualize:
            self.vis_proc = Process(target=self._visualize_loop, daemon=True)
            self.vis_proc.start()
        return "Pipeline started."

    def stop(self):
        if not self.is_running.value:
            return "Not running."
        self.is_running.value = False
        time.sleep(1)
        if self.capture_proc.is_alive():
            self.capture_proc.terminate()
        if self.infer_proc.is_alive():
            self.infer_proc.terminate()
        if self.visualize and self.vis_proc.is_alive():
            self.vis_proc.terminate()
        # Clean shared memory
        if self.color_shm:
            self.color_shm.close()
            self.color_shm.unlink()
        if self.depth_shm:
            self.depth_shm.close()
            self.depth_shm.unlink()
        return "Pipeline stopped."

    def get_latest_detection(self):
        return list(self.result_shm)  # copy to avoid race

    # ---------------- Capture Loop ----------------
    def _capture_loop(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color) # align depth to color

        # get intrinsics
        intr = get_realsense_intrinsics(profile)
        self.intr_shm[:] = intr.flatten()
        self.capture_ready.set()
        print("[YOLODetector] Capture started.")
        
        # Wrap shared memory as numpy arrays
        color_buf = np.ndarray(self.color_shape, dtype=self.color_dtype, buffer=self.color_shm.buf)
        depth_buf = np.ndarray(self.depth_shape, dtype=self.depth_dtype, buffer=self.depth_shm.buf)

        try:
            while self.is_running.value:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
            
                # Copy data into shared memory
                np.copyto(color_buf, np.asanyarray(color_frame.get_data()))
                np.copyto(depth_buf, np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale)

        except Exception as e:
            print(f"[YOLODetector] Capture error: {e}")
        finally:
            pipeline.stop()
            print("[YOLODetector] Capture stopped.")

    # ---------------- Inference Loop ----------------
    def _inference_loop(self):
        model = YOLO(self.model_path)
        print("[YOLODetector] YOLO model loaded.")

        self.capture_ready.wait()
        intr = np.frombuffer(self.intr_shm.get_obj(), dtype=np.float32).reshape((3, 3))
        extr = get_default_extrinsics()
        self.tf = CoordinateTransformer(intr, extr)
        print("[YOLODetector] Inference started.")

        color_buf = np.ndarray(self.color_shape, dtype=self.color_dtype, buffer=self.color_shm.buf)
        depth_buf = np.ndarray(self.depth_shape, dtype=self.depth_dtype, buffer=self.depth_shm.buf)

        try:
            while self.is_running.value:
                # copy latest frame (to avoid race while inference)
                color_img = color_buf.copy()
                depth_img = depth_buf.copy()

                # yolo inference
                results = model(color_img, conf=0.5, verbose=False)
                result = results[0]
                class_names = result.names
                detections = []
                for box in result.boxes:
                    class_name = class_names[int(box.cls[0])]
                    u, v = map(int, box.xywh[0][:2])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # depth = depth_frame.get_distance(u, v)
                    # mean filter
                    roi_size = 3
                    roi = depth_img[v-roi_size:v+roi_size+1, u-roi_size:u+roi_size+1] # note index order
                    depth = float(np.median(roi))
                    x_w, y_w, z_w = self.tf.pixel_to_world([u, v], depth)
                    # print(f"[YOLODetector] Detected {class_name} at ({x_w:.3f}, {y_w:.3f}, {z_w:.3f})")

                    detections.append({
                        "class": class_name,
                        "bbox": [x1, y1, x2, y2],
                        "pixel": [u, v],
                        "world": [x_w, y_w, z_w]
                    })

                self.result_shm[:] = detections
                time.sleep(0.1)  # small sleep to reduce CPU usage

        except Exception as e:
            print(f"[YOLODetector] Inference error: {e}")
            time.sleep(0.1)
        finally:
            print("[YOLODetector] Inference stopped.")

    # ---------------- Visualizer Loop ----------------
    def _visualize_loop(self):
        print("[YOLODetector] Visualizer started.")

        color_buf = np.ndarray(self.color_shape, dtype=self.color_dtype, buffer=self.color_shm.buf)

        try:
            while self.is_running.value:
                color_frame = color_buf.copy()
                detections = self.get_latest_detection()

                for detection in detections:
                    class_name = detection["class"]
                    x1, y1, x2, y2 = detection["bbox"]
                    u, v = detection["pixel"]
                    x_w, y_w, z_w = detection["world"]

                    cv2.rectangle(color_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(color_frame, (u, v), 6, (0, 0, 255), -1)
                    text = f"{class_name}: ({x_w:.2f}, {y_w:.2f}, {z_w:.2f})"
                    cv2.putText(color_frame, text, (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
                cv2.imshow("YOLO Visualizer", color_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"[YOLODetector] Visualizer error: {e}")
            time.sleep(0.1)
        finally:
            cv2.destroyAllWindows()
            print("[YOLODetector] Visualizer stopped.")


if __name__ == "__main__":
    try:
        detector = YOLODetector("./models/yolov8s-seg.pt", True)
        detector.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[YOLODetector] Interrupted by user, shutting down...")
    finally:
        detector.stop()