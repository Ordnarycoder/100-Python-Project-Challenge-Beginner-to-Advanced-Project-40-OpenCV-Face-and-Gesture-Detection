import cv2
import mediapipe as mp


"""Warning: Mediapipe isn't working python 3.11 or more new versions. You can use python 3.10 with virtual environment"""

# Load Haar Cascade (Something was wrong with the path so I added this file to project.)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect hands
    hand_results = hands.process(rgb_frame)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Count fingers for each detected hand
    if hand_results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # List of fingertips and corresponding joints
            fingers = [
                (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
                (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
            ]

            finger_count = 0

            for tip, pip in fingers:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:  # Check if fingertip is above the joint
                    finger_count += 1

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display finger count for each hand
            h, w, _ = frame.shape
            cv2.putText(
                frame, f"Hand {hand_index + 1} Fingers: {finger_count}",
                (10, h - 20 - (hand_index * 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )


    cv2.imshow("Face and Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
