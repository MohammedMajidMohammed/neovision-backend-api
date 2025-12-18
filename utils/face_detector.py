# utils/face_detector.py
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def remove_duplicates(faces):
    """
    Keep largest faces only (remove overlaps / duplicates)
    """
    if len(faces) <= 1:
        return faces

    # sort by area (w*h) descending
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)

    filtered = []
    for (x, y, w, h) in faces:
        keep = True
        for (fx, fy, fw, fh) in filtered:
            # simple overlap check
            if abs(x - fx) < fw * 0.3 and abs(y - fy) < fh * 0.3:
                keep = False
                break
        if keep:
            filtered.append((x, y, w, h))

    return filtered


def detect_faces(image):
    """
    image: RGB numpy array
    returns: list of (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,      # stricter than default
        minNeighbors=8,       # reduce false positives
        minSize=(100, 100)    # ignore small fake faces
    )

    faces = list(faces)
    faces = remove_duplicates(faces)

    return faces
