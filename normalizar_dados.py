from sklearn.preprocessing import MinMaxScaler

# Normalizar as imagens para [0,1]
scaler = MinMaxScaler()
frames_normalizados = scaler.fit_transform(frames)
