import cv2
import sqlite3
import datetime
import time
import logging

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detection logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Open a connection to SQLite database
with sqlite3.connect('faces_database.db') as conn:
    cursor = conn.cursor()
    
    # Create a table to store frame information
    cursor.execute('''CREATE TABLE IF NOT EXISTS my_test (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, x INTERGER, y INTEGER, frame BLOB)''')
    conn.commit()
    
    # Everything that is commented out is start and end time implementation (11:15 AM - 12:55 PM)
    #start_time = datetime.datetime.now().replace(hour = 11, minute = 15, second = 0, microsecond = 0)
    #end_time = datetime.datetime.now().replace(hour = 12, minute = 55, second = 0, microsecond = 0)
    
    cap = cv2.VideoCapture(0)
    
    # Capture images for x amount of time in seconds
    capture_duration = 25
    interval_seconds = 1/2
    start_time = datetime.datetime.now()
    
    #while datetime.datetime.now() < start_time:
        #time.sleep(60)
        
    #interval_minutes = 15
    #capture_duration = (end_time - start_time).seconds
    #capture_count =  capture_duration // (interval_minutes * 60)
    
    while (datetime.datetime.now() - start_time).seconds < capture_duration:
    #for _ in range(capture_count):
        ret, frame = cap.read()
        
        try:
            # Check if the frame is not empty
            if not ret or frame is None:
                break
        
            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)
            
            for (x, y, w, h) in faces:
                if w < 150 or h < 150:
                    # Resize the face if less than 150x150 pixels
                    face_resized = cv2.resize(frame[y:y+h, x:x+w], (512, 512))
                    _, img_encoded = cv2.imencode('.png', face_resized)

                else:
                    frame_to_crop = cv2.resize(frame[y:y+h, x:x+w], (512, 512))
                    _, img_encoded = cv2.imencode('.png', frame_to_crop)

                
                #_, img_encoded = cv2.imencode('.png', frame_to_crop)
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #x_coord = frame_to_crop[x:x+w]
                #y_coord = frame_to_crop[y:y+h]
                cursor.execute('INSERT INTO my_test (timestamp, x, y, frame) VALUES (?, ?, ?, ?)', (timestamp, int(x), int(y), img_encoded.tobytes()))
                conn.commit()
        
            cv2.imshow('Frame', frame)
        
            #time.sleep(interval_minutes * 60)
            time.sleep(interval_seconds)
        
            # Break the loop if 'q' is pressed, 's' starts Safe Shutdown
            key = cv2.waitKey(1)
            if key == ord('q'):
                logger.info("Starting safe shutdown")
                break
        
        except Exception as e:
            logger.error(f"Error: {e}")
            break

cap.release()
cv2.destroyAllWindows()
conn.close()
