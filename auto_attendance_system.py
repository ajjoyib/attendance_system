import cv2
import face_recognition
import json

# Initialize variables to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Try to load an existing student library from a JSON file
try:
    with open('student_library.json', 'r') as file:
        student_library = json.load(file)
        known_face_encodings = student_library['encodings']
        known_face_names = student_library['names']
except FileNotFoundError:
    student_library = {'encodings': known_face_encodings, 'names': known_face_names}

# Initialize the video capture from the default camera (0)
video_capture = cv2.VideoCapture(0)

# Set registration_mode to True initially to start registration
registration_mode = True

# Enter the main loop for capturing and processing frames
while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # In registration_mode, capture faces and register them with names
    for face_encoding in face_encodings:
        if registration_mode:
            name = input("Enter the student's name: ")
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            # Update the student library with the new data and save it to a JSON file
            student_library = {'encodings': known_face_encodings, 'names': known_face_names}
            with open('student_library.json', 'w') as file:
                json.dump(student_library, file)
            registration_mode = False  # Switch to attendance mode after registration

    face_names = []
    for face_encoding in face_encodings:
        # Compare the face encodings to known students
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)

    # Draw rectangles and labels on the frame for recognized faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Red rectangle
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame with recognized faces
    cv2.imshow('Video', frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
