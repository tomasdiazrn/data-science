import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

print()
print('- Algorithm: K-Nearest Neighbours')
print('- The goal of the analysis is to forecast the outcome of a potential new loan: paid versus unpaid.')
print('')

# Función para generar un conjunto de datos extendido
def generate_extended_dataset(size=100):
    np.random.seed(42)

    # Generar datos aleatorios
    resultado = np.random.choice(['Pagado', 'No pagado'], size=size)
    total_prestamo = np.random.randint(5000, 30000, size=size)
    ingresos = np.random.randint(30000, 120000, size=size)
    proposito = np.random.choice(['Consolidacion de deuda', 'Traslado', 'Pequenio negocio', 'Compra de auto', 'Educacion', 'Vacaciones'], size=size)
    anios_empleo = np.random.randint(1, 15, size=size)
    prop_vivienda = np.random.choice(['HIPOTECA', 'ALQUILER'], size=size)
    estado = np.random.choice(['NV', 'TN', 'MD', 'CA', 'KS', 'NY', 'FL', 'TX', 'AZ'], size=size)

    # Crear DataFrame
    df = pd.DataFrame({
        'Resultado': resultado,
        'TotalPrestamo': total_prestamo,
        'Ingresos': ingresos,
        'Proposito': proposito,
        'AniosEmpleo': anios_empleo,
        'PropVivienda': prop_vivienda,
        'Estado': estado,
    })

    return df

loan200 = pd.DataFrame({
    'Resultado': ['Pagado', 'Pagado', 'Pagado', 'No pagado', 'Pagado', 'Pagado', 'No pagado', 'Pagado', 'No pagado', 'Pagado'],
    'TotalPrestamo': [10000, 9600, 18800, 15250, 17050, 5500, 20000, 12000, 8000, 15000],
    'Ingresos': [79100, 48000, 120036, 232000, 35000, 43000, 90000, 60000, 35000, 75000],
    'Proposito': ['Consolidacion de deuda', 'Traslado', 'Consolidacion de deuda', 'Pequenio negocio', 'Consolidacion de deuda', 'Consolidacion de deuda', 'Compra de auto', 'Educacion', 'Vacaciones', 'Compra de auto'],
    'AniosEmpleo': [11, 5, 11, 9, 4, 4, 8, 3, 2, 6],-
    'PropVivienda': ['HIPOTECA', 'HIPOTECA', 'HIPOTECA', 'HIPOTECA', 'ALQUILER', 'ALQUILER', 'HIPOTECA', 'ALQUILER', 'ALQUILER', 'HIPOTECA'],
    'Estado': ['NV', 'TN', 'MD', 'CA', 'MD', 'KS', 'NY', 'FL', 'TX', 'AZ'],
})

# Generar un DataFrame extendido con 200 filas
loan200 = generate_extended_dataset(size=200)

# Predictors is a list of column names used as features to predict the outcome.
# Outcome is the column we want to predict.
predictors = ['TotalPrestamo', 'Ingresos', 'AniosEmpleo']
outcome = 'Resultado'

# Newloan contains the features of a new loan that we want to predict.
# X contains the features of existing loans.
# y contains the corresponding outcomes of existing loans.

newloan = loan200.loc[0:0, predictors]
X = loan200.loc[1:, predictors]
y = loan200.loc[1:, outcome]

# n_neighbors is set to the minimum of 5 and the number of samples in the dataset.
# KNeighborsClassifier is instantiated with the specified number of neighbors.
n_neighbors = min(5, len(X))
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
# The model is trained with the existing loan data.
# prediction = knn.predict(newloan)
knn.fit(X, y)
prediction = knn.predict(newloan)

print("Prediction for the new loan:", prediction)
print()
print(loan200)
