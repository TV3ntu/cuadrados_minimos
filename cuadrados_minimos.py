import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Leer los datos del archivo Excel
df = pd.read_csv('ACUMULADOS vs DIAS.csv')

# Extraer las columnas de días y casos infectados
dias = df['dia'].values
infectados = df['acumulados'].values

# Crear el gráfico de dispersión de los datos
plt.scatter(dias, infectados, label='Datos', color='blue')

# Ajustar una línea recta (y = ax + b)
a, b, r, _, _ = linregress(dias, infectados)
linea_recta = a * dias + b
plt.plot(dias, linea_recta, label=f'Línea Recta: y = {a:.2f}x + {b:.2f} (r = {r:.2f})', color='red')

# Ajustar una curva cuadrática (y = bx^a)
p, cov = np.polyfit(np.log(dias), np.log(infectados), deg=1, cov=True)
a = p[0]
b = np.exp(p[1])
r = np.sqrt(1 - np.diag(cov)[1] / np.diag(cov)[0])
curva_cuadratica = b * np.power(dias, a)
plt.plot(dias, curva_cuadratica, label=f'Curva Cuadrática: y = {b:.2f}x^{a:.2f} (r = {r:.2f})', color='green')

# Ajustar una curva exponencial (y = be^(ax))
a, b, r, _, _ = linregress(dias, np.log(infectados))
curva_exponencial = b * np.exp(a * dias)
plt.plot(dias, curva_exponencial, label=f'Curva Exponencial: y = {b:.2f}e^({a:.2f}x) (r = {r:.2f})', color='purple')

# Configurar el gráfico
plt.xlabel('Días')
plt.ylabel('Infectados')
plt.title('Curvas de Tendencia de Infectados')
plt.legend()

# Mostrar el gráfico
plt.show()
