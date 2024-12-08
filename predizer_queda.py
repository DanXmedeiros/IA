from tensorflow.keras.models import load_model
import cv2

# Carregar o modelo treinado
model = load_model('fall_detection_model.h5')

# Função para predizer queda
def predizer_queda(imagem):
    imagem = cv2.resize(imagem, (224, 224))
    imagem = np.expand_dims(imagem, axis=0)
    resultado = model.predict(imagem)
    return resultado[0][0]

# Testar o modelo com uma imagem
imagem_teste = cv2.imread('teste_foto.jpg')
print('Probabilidade de queda:', predizer_queda(imagem_teste))
