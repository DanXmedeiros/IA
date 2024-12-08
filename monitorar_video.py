import cv2
import numpy as np
from tensorflow.keras.models import load_model

# URL do stream do ESP32-CAM
esp32_stream_url = "http://192.168.1.100/stream"
model_path = "fall_detection_model.h5"

# Carregar o modelo treinado
model = load_model(model_path)

# Capturar o stream de vídeo
cap = cv2.VideoCapture(esp32_stream_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar o vídeo.")
        break
    
    # Redimensionar para 224x224
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0  # Normalizar
    input_data = np.expand_dims(normalized_frame, axis=0)
    
    # Prever queda
    prediction = model.predict(input_data)
    label = "Fall" if prediction > 0.5 else "No Fall"
    
    # Mostrar o resultado no frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ESP32-CAM Stream', frame)
    
    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
