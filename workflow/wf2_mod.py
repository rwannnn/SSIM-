import cv2
import os
import json
import base64
import telegram
import io
import numpy as np
from datetime import datetime, timezone
from collections import Counter, deque
from typing import Any, List
from dotenv import load_dotenv
from ultralytics import YOLO
import asyncio
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

#---------- Configurations ----------

# Model initialization
ppe_model = YOLO("YOLO11n_PPE.pt")
vlm_model_name = "llava:7b-v1.6-mistral-q2_K"
USE_API = True  # Set to True to use Gemini API, False for local model
API_MODEL_NAME = "gemini-2.5-flash"

# Capture mode selection
USE_VIDEO_FILE = True  # Set to True to process a video file instead of webcam
VIDEO_PATH = "TEST_FOOTAGE.mp4"  # Path to your video file
WEBCAM_INDEX = 1  # Default webcam index

PPE_RATE = 1  # seconds between PPE detections
VLM_RATE = 1  # minutes between VLM analyses
IMG_BUFFER_SIZE = 50  # Number of images to keep in buffer for context

#---------- Functions ----------
num_classes = 17
colors = []
for i in range(num_classes):
    hue = int(i * 180 / num_classes)
    hsv_color = np.uint8([[[hue, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
    colors.append(tuple(map(int, bgr_color[0][0])))

def draw_boxes(image, detections):
    for box in detections[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = box.conf[0]
        label = f"{detections[0].names[cls_id]} {conf:.2f}"
        color = colors[cls_id % len(colors)]
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def log_detections(class_counts, datetimestamp):
    log_entry = {
        'timestamp': datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"),
        'detections': dict(class_counts)
    }
    log_message = json.dumps(log_entry, indent=4)
    
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/ppe_detections_{datetimestamp}.jsonl", "a") as log_file:
        log_file.write(log_message + "\n")

class ImageBuffer:
    """A class to manage a fixed-size buffer that stores the latest 'n' images."""
    def __init__(self, max_size: int, save_dir: str):
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer.")
        if not isinstance(save_dir, str) or not save_dir:
            raise ValueError("save_dir must be a non-empty string.")

        self._max_size = max_size
        self._save_dir = save_dir
        os.makedirs(self._save_dir, exist_ok=True)
        self._managed_files = deque()

    def add(self, image: Any):
        """Saves a new image and deletes the oldest image if the buffer is full"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{timestamp}.jpg" 
        filepath = os.path.join(self._save_dir, filename)
        
        try:
            success = cv2.imwrite(filepath, image)
            if not success:
                print(f"Failed to save image to: {filepath}")
                return
        except Exception as e:
            print(f"Error saving image with OpenCV: {e}")
            return
        
        self._managed_files.append(filepath)
        
        if len(self._managed_files) > self._max_size:
            oldest_file = self._managed_files.popleft()
            try:
                os.remove(oldest_file)
            except Exception as e:
                print(f"Error deleting file {oldest_file}: {e}")

    def get_file_list(self) -> List[str]:
        """Returns a list of the current image file paths being managed."""
        return list(self._managed_files)

    def __len__(self) -> int:
        """Returns the current number of images being managed on disk."""
        return len(self._managed_files)

class ViolationTracker:
    """Tracks the persistence of specific safety violations based on object detection counts."""
    def __init__(self, thresholds: dict = None):
        if thresholds is None:
            self.thresholds = {}
        else:
            self.thresholds = thresholds
        
        self.violations = {
            v_type: {'active': False, 'start_time': None, 'alert_sent': False}
            for v_type in self.thresholds.keys()
        }

        self._violation_check_map = {
            'HEAD_UNPROTECTED': self._check_head_protection,
            'FACE_UNPROTECTED': self._check_face_protection,
            'EARS_UNPROTECTED': self._check_ear_protection,
            'HANDS_UNPROTECTED': self._check_hand_protection,
            'FEET_UNPROTECTED': self._check_foot_protection,
            'BODY_UNPROTECTED': self._check_body_protection
        }
        
        for v_type in self.thresholds:
            if v_type not in self._violation_check_map:
                raise ValueError(f"'{v_type}' is not a valid violation type.")
        
        print(f"ViolationTracker initialized. Tracking: {list(self.violations.keys()) or 'None'}")

    def _check_head_protection(self, counts: Counter) -> bool:
        return counts.get('head', 0) > counts.get('helmet', 0)

    def _check_face_protection(self, counts: Counter) -> bool:
        face_coverings = counts.get('face-guard', 0) + counts.get('face-mask', 0)
        return counts.get('face', 0) > face_coverings

    def _check_ear_protection(self, counts: Counter) -> bool:
        return counts.get('ear', 0) > counts.get('ear-mufs', 0)

    def _check_hand_protection(self, counts: Counter) -> bool:
        return counts.get('hands', 0) > counts.get('gloves', 0)

    def _check_foot_protection(self, counts: Counter) -> bool:
        return counts.get('foot', 0) > counts.get('shoes', 0)

    def _check_body_protection(self, counts: Counter) -> bool:
        persons = counts.get('person', 0)
        body_coverings = counts.get('safety-vest', 0) + counts.get('safety-suit', 0)
        return persons > 0 and persons > body_coverings

    def update(self, class_counts: Counter):
        """Updates the state of all tracked violations based on the latest detection counts."""
        current_time = datetime.now(timezone.utc).timestamp()

        for v_type, v_state in self.violations.items():
            check_function = self._violation_check_map[v_type]
            is_violated = check_function(class_counts)

            if is_violated:
                if not v_state['active']:
                    v_state['active'] = True
                    v_state['start_time'] = current_time
                    print(f"[{datetime.fromtimestamp(current_time, tz=timezone.utc).strftime('%H:%M:%S')}] New violation started: {v_type}")
            else:
                if v_state['active']:
                    duration = current_time - v_state['start_time']
                    print(f"[{datetime.fromtimestamp(current_time, tz=timezone.utc).strftime('%H:%M:%S')}] Violation ended: {v_type} (duration: {duration:.1f}s)")
                    v_state['active'] = False
                    v_state['start_time'] = None
                    v_state['alert_sent'] = False

    def check_for_alerts(self) -> list:
        """Checks if any active violations have exceeded their time threshold."""
        current_time = datetime.now(timezone.utc).timestamp()
        alerts_to_send = []
        for v_type, v_state in self.violations.items():
            if v_state['active'] and not v_state['alert_sent']:
                duration = current_time - v_state['start_time']
                if duration >= self.thresholds[v_type]:
                    alert_info = {
                        'type': v_type,
                        'duration': duration,
                        'threshold': self.thresholds[v_type],
                        'message': f"ALERT: {v_type} violation has persisted for over {self.thresholds[v_type]} seconds."
                    }
                    alerts_to_send.append(alert_info)
                    v_state['alert_sent'] = True
                    print(f"‼️  ALERT TRIGGERED: {alert_info['message']}")
        return alerts_to_send

async def send_telegram_alert(bot_token: str, chat_id: str, message: str = None, image_path: str = None, frame=None):
    """Sends a text message or an image to a Telegram chat."""
    try:
        bot = telegram.Bot(token=bot_token)
        if message:
            await bot.send_message(chat_id=chat_id, text=message)
            print(f"✅ Text message sent to Chat ID {chat_id}.")

        if image_path:
            with open(image_path, 'rb') as photo_file:
                await bot.send_photo(chat_id=chat_id, photo=photo_file)
            print(f"✅ Photo '{image_path}' sent to Chat ID {chat_id}.")
        
        # Send OpenCV frame if provided
        if frame is not None:
            # Encode the frame in JPEG format to a memory buffer
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                # Create an in-memory binary stream
                bio = io.BytesIO(buffer)
                bio.name = 'frame.jpg'
                bio.seek(0)
                await bot.send_photo(chat_id=chat_id, photo=bio)
                print(f"✅ OpenCV frame sent to Chat ID {chat_id}.")
            else:
                print(f"❌ Failed to encode frame for Chat ID {chat_id}.")

        return True

    except telegram.error.TelegramError as e:
        print(f"❌ Error sending Telegram message: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

# Worker function for PPE detection (CPU-bound - for multiprocessing)
def process_ppe_detection(frame):
    """Process PPE detection on a frame. Runs in separate process."""
    detections = ppe_model(frame, device='cpu', verbose=False, show=False)
    class_ids = detections[0].boxes.cls.int().tolist()
    class_names = detections[0].names
    detected_class_names = [class_names[i] for i in class_ids]
    class_counts = Counter(detected_class_names)
    
    return {
        'class_counts': class_counts,
        'detections': detections
    }

def create_vlm_message(image_array, yolo_findings):
    """Create VLM message from image and YOLO findings."""
    _, buffer = cv2.imencode(".jpeg", image_array)
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    
    prompt_text = f"""
You are an expert on-site safety inspector. A real-time PPE detection model has already reported the following objects: **{yolo_findings}**.

Your task is to analyze the provided image to identify all **other** safety violations. Focus on:
1. **Environmental Hazards:** Spills, trip hazards (cables, debris), blocked pathways, fire risks.
2. **Improper Equipment Use:** Misuse of tools, unsecured machinery, unsafe vehicle operation.
3. **Unsafe Practices:** Workers in unsafe positions (e.g., under suspended loads, near an edge without fall protection), improper lifting, lack of warning signs or barriers.
4. Filter your findings and only output ```high priority hazards that require immediate attention```.

**Output Instructions:**
- Provide a bulleted list of all identified hazards.
- Each bullet point must be concise (under 20 words).
- If no additional hazards are found, respond with the single word "None".
- Don't output any text other than the bulleted hazards or None.
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
            },
        ]
    )
    return [message]

# Worker function for VLM processing
def process_vlm_sync(image_array, yolo_findings, use_api, api_key=None):
    """Process VLM analysis. For local model (CPU-bound) or API (IO-bound)."""
    if use_api:
        llm = ChatGoogleGenerativeAI(model=API_MODEL_NAME, google_api_key=api_key)
    else:
        llm = ChatOllama(model=vlm_model_name, temperature=0.0)
    
    messages = create_vlm_message(image_array, yolo_findings)
    response = llm.invoke(messages)
    
    if hasattr(response, 'content'):
        return response.content
    return str(response)

# Environment setup
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not BOT_TOKEN or not CHAT_ID:
    print("Error: BOT_TOKEN and CHAT_ID must be set in the environment variables.")
    exit(1)

async def main(ppe_rate_s=PPE_RATE, vlm_rate_m=VLM_RATE, img_buffer_size=IMG_BUFFER_SIZE):
    """Main async function with parallelism support."""
    
    violation_thresholds = {
        'HEAD_UNPROTECTED': 10,
        'BODY_UNPROTECTED': 30
    }
    tracker = ViolationTracker(thresholds=violation_thresholds)

    # --- Initialize video capture ---
    if USE_VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{VIDEO_PATH}'")
            exit()
        print(f"🎬 Processing video file: {VIDEO_PATH}")
    else:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()
        print("📷 Using webcam stream...")
    
    vlm_rate_s = vlm_rate_m * 60
    
    initial_timestamp = datetime.now(timezone.utc).timestamp()
    last_processed_time_ppe = initial_timestamp
    last_processed_time_vlm = initial_timestamp

    print(f"Processing mode: {'API (Threading)' if USE_API else 'Local Model (Multiprocessing)'}")
    print(f"PPE detection: every {ppe_rate_s}s | VLM analysis: every {vlm_rate_m}m")
    
    start_datetime = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    previous_counts = Counter()
    img_buffer = ImageBuffer(max_size=img_buffer_size, save_dir=f"image_buffer/{start_datetime}")

    # Create executor pools
    ppe_executor = ProcessPoolExecutor(max_workers=2)
    
    if USE_API:
        vlm_executor = ThreadPoolExecutor(max_workers=2)
    else:
        vlm_executor = ProcessPoolExecutor(max_workers=2)
    
    loop = asyncio.get_event_loop()
    
    # Track pending tasks
    pending_ppe_task = None
    pending_vlm_task = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            current_time = datetime.now(timezone.utc).timestamp()
            
            # Submit PPE detection task if it's time and no task is pending
            if (current_time - last_processed_time_ppe) >= ppe_rate_s and pending_ppe_task is None:
                print("-----------------------------------")
                last_processed_time_ppe = current_time
                
                # Submit to process pool
                pending_ppe_task = loop.run_in_executor(
                    ppe_executor,
                    process_ppe_detection,
                    frame.copy()
                )
            
            # Check if PPE task is complete
            if pending_ppe_task is not None and pending_ppe_task.done():
                result = await pending_ppe_task
                pending_ppe_task = None
                
                class_counts = result['class_counts']
                detections = result['detections']
                
                # Update violation tracker
                tracker.update(class_counts)
                violation_alerts = tracker.check_for_alerts()
                
                # Send alerts
                for alert in violation_alerts:
                    alert_message = alert['message']
                    await send_telegram_alert(BOT_TOKEN, CHAT_ID, alert_message)
                
                if len(violation_alerts) > 0:
                    await send_telegram_alert(
                        BOT_TOKEN, CHAT_ID,
                        image_path = img_buffer.get_file_list()[-1]
                    )
                
                # Log and display changes
                if class_counts != previous_counts:
                    previous_counts = class_counts
                    print(f"Detected changes: {dict(class_counts)}")
                    log_detections(class_counts, initial_timestamp)
                    annotated_frame = draw_boxes(frame.copy(), detections)
                    img_buffer.add(annotated_frame)
                    
                    cv2.imshow("Detection Feed", annotated_frame)
            
            # Submit VLM task if it's time and no task is pending
            if (current_time - last_processed_time_vlm) >= vlm_rate_s and pending_vlm_task is None:
                last_processed_time_vlm = current_time
                print(f"Submitting VLM task at {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
                
                img_buffer.add(frame.copy())
                yolo_findings = [a['message'] for a in tracker.check_for_alerts()]
                
                # Submit to appropriate executor
                frame_vlm = frame.copy()
                pending_vlm_task = loop.run_in_executor(
                    vlm_executor,
                    process_vlm_sync,
                    frame_vlm,
                    yolo_findings,
                    USE_API,
                    GOOGLE_API_KEY if USE_API else None
                )
            
            # Check if VLM task is complete
            if pending_vlm_task is not None and pending_vlm_task.done():
                vlm_description = await pending_vlm_task
                pending_vlm_task = None
                
                print(f"VLM analysis complete: {vlm_description[:100]}...")
                
                if vlm_description != 'None':
                    await send_telegram_alert(
                        BOT_TOKEN, CHAT_ID,
                        vlm_description,
                        None,
                        frame_vlm
                    )
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.01)
    
    finally:
        # Cleanup
        print("Shutting down...")
        ppe_executor.shutdown(wait=True)
        vlm_executor.shutdown(wait=True)
        cap.release()
        cv2.destroyAllWindows()

# Global control variables
_stop_event = threading.Event()
_frame_queue = queue.Queue(maxsize=2)
_log_queue = queue.Queue(maxsize=100)

def streamlit_log(message: str):
    """Redirects logs to Streamlit queue instead of console."""
    try:
        _log_queue.put_nowait(message)
    except queue.Full:
        pass
    print(message)  # still keep console output

def streamlit_frame(image):
    """Sends annotated frame to Streamlit."""
    if _frame_queue.full():
        _frame_queue.get_nowait()
    _frame_queue.put_nowait(image)

def stop_main():
    _stop_event.set()

async def main_streamlit(ppe_rate_s = PPE_RATE, vlm_rate_m = VLM_RATE, img_buffer_size = IMG_BUFFER_SIZE):
    """Streamlit-integrated version of main()."""
    violation_thresholds = {
        'HEAD_UNPROTECTED': 10,
        'BODY_UNPROTECTED': 30
    }
    tracker = ViolationTracker(thresholds=violation_thresholds)

    # --- Initialize video capture ---
    if USE_VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            streamlit_log(f"❌ Error: Could not open video file '{VIDEO_PATH}'.")
            return
        streamlit_log(f"🎬 Processing video file: {VIDEO_PATH}")
    else:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            streamlit_log("❌ Error: Could not open webcam.")
            return
        streamlit_log("📷 Using webcam stream...")


    vlm_rate_s = vlm_rate_m * 60
    initial_timestamp = datetime.now(timezone.utc).timestamp()
    last_processed_time_ppe = initial_timestamp
    last_processed_time_vlm = initial_timestamp

    streamlit_log(f"Processing mode: {'API' if USE_API else 'Local Model'}")
    streamlit_log(f"PPE detection every {ppe_rate_s}s | VLM every {vlm_rate_m}m")

    previous_counts = Counter()
    img_buffer = ImageBuffer(max_size=img_buffer_size, save_dir=f"image_buffer/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}")

    ppe_executor = ProcessPoolExecutor(max_workers=2)
    vlm_executor = ProcessPoolExecutor(max_workers=2)

    loop = asyncio.get_event_loop()
    pending_ppe_task = None
    pending_vlm_task = None

    try:
        while not _stop_event.is_set():
            success, frame = cap.read()
            if not success:
                break

            current_time = datetime.now(timezone.utc).timestamp()

            # Submit PPE detection
            if (current_time - last_processed_time_ppe) >= ppe_rate_s and pending_ppe_task is None:
                last_processed_time_ppe = current_time
                pending_ppe_task = loop.run_in_executor(
                    ppe_executor,
                    process_ppe_detection,
                    frame.copy()
                )

            # PPE results
            if pending_ppe_task and pending_ppe_task.done():
                result = await pending_ppe_task
                pending_ppe_task = None
                class_counts = result['class_counts']
                detections = result['detections']
                tracker.update(class_counts)
                alerts = tracker.check_for_alerts()

                if alerts:
                    for alert in alerts:
                        streamlit_log(alert['message'])
                        await send_telegram_alert(BOT_TOKEN, CHAT_ID, alert['message'])
                        await send_telegram_alert(BOT_TOKEN, CHAT_ID, image_path=img_buffer.get_file_list()[-1])

                if class_counts != previous_counts:
                    previous_counts = class_counts
                    log_detections(class_counts, initial_timestamp)
                    streamlit_log(f"Detected: {dict(class_counts)}")
                    annotated = draw_boxes(frame.copy(), detections)
                    img_buffer.add(annotated)
                    streamlit_frame(annotated)

            # VLM task
            if (current_time - last_processed_time_vlm) >= vlm_rate_s and pending_vlm_task is None:
                last_processed_time_vlm = current_time
                
                frame_vlm = frame.copy()
                pending_vlm_task = loop.run_in_executor(
                    vlm_executor,
                    process_vlm_sync,
                    frame_vlm,
                    [a['message'] for a in tracker.check_for_alerts()],
                    USE_API,
                    GOOGLE_API_KEY if USE_API else None
                )

            if pending_vlm_task and pending_vlm_task.done():
                desc = await pending_vlm_task
                pending_vlm_task = None
                streamlit_log(f"VLM: {desc[:100]}...")
                if desc != "None":
                    img_buffer.add(frame.copy())
                    await send_telegram_alert(BOT_TOKEN, CHAT_ID, desc, None, frame_vlm)

            await asyncio.sleep(0.01)

    finally:
        streamlit_log("🛑 Stopping video capture...")
        ppe_executor.shutdown(wait=False)
        vlm_executor.shutdown(wait=False)
        cap.release()

def run_main_threaded():
    """Run the main async loop inside a background thread."""
    _stop_event.clear()
    def _runner():
        asyncio.run(main_streamlit())
    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t

def get_latest_frame():
    """Fetch latest frame (np.ndarray)."""
    try:
        return _frame_queue.get_nowait()
    except queue.Empty:
        return None

def get_latest_logs(max_lines=100):
    """Fetch recent logs."""
    items = []
    while not _log_queue.empty():
        try:
            items.append(_log_queue.get_nowait())
        except queue.Empty:
            break
    return items[-max_lines:]
