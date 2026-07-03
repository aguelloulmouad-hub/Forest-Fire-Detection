import numpy as np
import cv2
import json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = "best_fire_smoke_mobilenetv2.keras"
CLASS_INDICES_PATH = "class_indices.json"
VIDEO_PATH = "video_test_1.mp4"
OUTPUT_VIDEO_PATH = "video_demo.mp4"
IMG_SIZE = (224, 224)
THRESHOLD_NON_FIRE = 0.043

# Charger le modèle
print("Chargement du modèle...")
best_model = load_model(MODEL_PATH)

# Charger les indices de classes
print("Chargement des indices de classes...")
with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}


def predict_frame(frame, model, threshold=THRESHOLD_NON_FIRE):
    """Prédiction sur un frame de vidéo."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    probs = model.predict(arr, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])
    label = idx_to_class[best_idx]

    if best_prob < threshold:
        return "Non-fire", best_prob

    return label, best_prob


print(f"Ouverture de la vidéo: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Erreur: Impossible d'ouvrir la vidéo.")
    raise SystemExit(1)

# Propriétés de la vidéo
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✓ Vidéo chargée - Résolution: {width}x{height}, FPS: {fps:.2f}, Total frames: {total_frames}")
print(f"Sortie vidéo: {OUTPUT_VIDEO_PATH}")
print("\nTraitement de la vidéo en cours...")
print("Appuyez sur 'q' pour arrêter\n")

# Writer pour enregistrer la vidéo annotée
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    print("❌ Erreur: Impossible de créer le fichier vidéo de sortie.")
    cap.release()
    raise SystemExit(1)

frame_count = 0
fire_detections = 0
smoke_detections = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prédiction
        label, prob = predict_frame(frame, best_model)

        # Couleur d'affichage
        if label == "fire":
            fire_detections += 1
            color = (0, 0, 255)
        elif label == "smoke":
            smoke_detections += 1
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)

        # Annoter le frame
        text = f"{label} | {prob:.3f}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.rectangle(frame, (10, 10), (350, 60), color, 2)

        frame_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame, frame_text, (max(20, width - 300), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        # Écriture dans le fichier vidéo
        out.write(frame)

        # Affichage à l'écran
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Fire & Smoke Detection Demo", display_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️  Arrêt par l'utilisateur")
            break

except KeyboardInterrupt:
    print("\n⏹️  Arrêt par l'utilisateur (Ctrl+C)")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print("Traitement terminé!")
    print(f"{'='*50}")
    print(f"Total frames traités: {frame_count}")
    print(f"Détections de feu: {fire_detections}")
    print(f"Détections de fumée: {smoke_detections}")
    print(f"Pas de détection: {frame_count - fire_detections - smoke_detections}")
    print(f"Vidéo enregistrée sous: {OUTPUT_VIDEO_PATH}")
    print(f"{'='*50}\n")
