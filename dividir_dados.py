from sklearn.model_selection import train_test_split

# Dividir os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(frames_normalizados, labels, test_size=0.2)
