import cv2
import threading
import queue
import time
import os
import cv2.data
import subprocess
import datetime
import deepface
from deepface import Deepface
import os


orb = cv2.ORB_create()

frame_queue = queue.Queue(maxsize=50)
processed_frame_queue = queue.Queue(maxsize=50)

running = True
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def generate_output_paths():
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_dir = "content/video"
    output_audio_dir = "content/audio"

    output_video_filename = f"live_stream_content_{current_time}.mp4"
    output_audio_filename = f"live_stream_audio_{current_time}.wav"

    output_video_path = os.path.join(output_video_dir, output_video_filename)
    output_audio_path = os.path.join(output_audio_dir, output_audio_filename)

    return output_video_path, output_audio_path

output_path, output_audio_path = generate_output_paths()


def video_capture():
    global running
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc=video_writer, fps=fps, frameSize=(frame_width, frame_height))
    while running:
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                out.write(frame)
                frame_queue.put(frame)
        else:
            break
    out.release()
    cap.release()

def record_audio(audio_output_path, duration):
    ffmpeg_audio_command = [
        "ffmpeg",
        "-f", "avfoundation",  
        "-i", ":0",  # Input device, adjust according to your system
        "-t", str(duration),  # Duration to record in seconds
        "-y", audio_output_path  # Output file path
    ]
    process = subprocess.Popen(ffmpeg_audio_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()  # Wait for the process to finish

def merge_audio_video(video_path, audio_path, output_path):
    ffmpeg_merge_command = [
        "ffmpeg",
        "-i", video_path,  # Input video file
        "-i", audio_path,  # Input audio file
        "-c:v", "copy",    # Copy video stream without re-encoding
        "-c:a", "aac",     # Encode audio in AAC format (widely compatible for MP4)
        "-strict", "experimental",  # AAC compatibility flag
        output_path  # Output merged file
    ]
    process = subprocess.Popen(ffmpeg_merge_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()  # Wait for the process to finish

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 10)
    return faces

def extract_keypoints(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(frame, None)
    return kp, des

def match_frames(des1, des2, min_match_count):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    print(f"Number of matches: {len(matches)}")
    if len(matches) > min_match_count:
        return True
    return False

def save_frame(frame, output_directory, frame_count):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    cv2.imwrite(f"{output_directory}/keyframe_{frame_count}.jpg", frame)

def frame_processing():
    global running
    prev_descriptors = None
    frame_count = 0
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get(timeout=1)
            start_time = time.time()
            if frame is not None:
                faces = detect_bounding_box(frame)  
                processed_frame_queue.put(frame)
                print(f"Number of faces:{len(faces)}")
                if len(faces) > 0:
                    kp, des = extract_keypoints(frame=frame)
                    print(kp, des)
                    if prev_descriptors is None:
                        save_frame(frame=frame, output_directory="/Users/dhruvmehrottra007/Desktop/Transformers", frame_count=frame_count)
                    else:
                        if not match_frames(des1=prev_descriptors, des2=des, min_match_count=200):
                            save_frame(frame=frame, output_directory="/Users/dhruvmehrottra007/Desktop/Transformers", frame_count=frame_count)
                    prev_descriptors = des
                    frame_count += 1
                end_time = time.time()    # End time after processing
                print(f"Processing time: {end_time - start_time:.4f} seconds")
            frame_queue.task_done()

capture_thread = threading.Thread(target=video_capture)
#process_thread = threading.Thread(target=frame_processing)
audio_thread = threading.Thread(target=record_audio, args=(output_audio_path, 60))

capture_thread.start()
#process_thread.start()
audio_thread.start()

try:
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is not None:
                cv2.imshow("My Face Detection Project", frame)  # Show the modified frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False  # Exit on 'q' key press
            frame_queue.task_done()
        time.sleep(0.01)  # Avoid excessive CPU usage
except KeyboardInterrupt:
    running = False

capture_thread.join()
#process_thread.join()
audio_thread.join()

cv2.destroyAllWindows()