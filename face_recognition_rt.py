import cv2
import face_recognition

class FaceRecognition:
    def __init__(self):
        self.recognize_faces = {
            "ayanPanda": "AyanPanda.jpg",
            "rishavKumar": "RishavKumar.jpg",
            "Muneeswaran": "DrMuneeswaranV.jpg",
            "dhruvGupta": "DhruvGupta.jpg",
            "bhairavSharma": "BhairavSharma.jpg"
        }
        self.known_face_encodings = []
        self.known_face_names = []
        self.webcam_video_stream = cv2.VideoCapture(0)
        self.load_known_faces()

    def load_known_faces(self):
        for name, image_path in self.recognize_faces.items():
            image = cv2.imread(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
        print("Faces loaded successfully.")

    def run(self):
        while True:
            ret, current_frame = self.webcam_video_stream.read()
            if not ret:
                print("Failed to grab frame.")
                break

            current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
            all_face_locations = face_recognition.face_locations(current_frame_small, model='hog')
            all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

            for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
                top_pos, right_pos, bottom_pos, left_pos = current_face_location
                top_pos *= 4
                right_pos *= 4
                bottom_pos *= 4
                left_pos *= 4

                all_matches = face_recognition.compare_faces(self.known_face_encodings, current_face_encoding)
                name_of_person = 'Safe Person'

                if True in all_matches:
                    first_match_index = all_matches.index(True)
                    name_of_person = self.known_face_names[first_match_index]

                if name_of_person in ["ayanPanda", "Muneeswaran"]:
                    color = (0, 0, 255)
                elif name_of_person in ["rishavKumar", "bhairavSharma"]:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)

                cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), color, 2)
                cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            cv2.imshow("Webcam Video", current_frame)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        self.cleanup()

    def cleanup(self):
        self.webcam_video_stream.release()
        cv2.destroyAllWindows()
        print("Resources released.")


if __name__ == "__main__":
    face_recognition_app = FaceRecognition()
    face_recognition_app.run()
