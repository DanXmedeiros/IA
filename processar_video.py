import cv2

video_path = "C:\Users\aluno\Videos"  # Altere para o caminho real

# Abrir o vídeo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aqui você pode processar o frame para detecção
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
