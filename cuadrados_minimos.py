import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
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

# Ajustar una ecuación de potencia (y = bx^a)
p, cov = np.polyfit(np.log(dias), np.log(infectados), deg=1, cov=True)
a = p[0]
b = np.exp(p[1])
#r = np.sqrt(1 - np.diag(cov)[1] / np.diag(cov)[0])
# Calcular el coeficiente de correlación
residuals = np.log(infectados) - (a * np.log(dias) + np.log(b))
ss_residuals = np.sum(residuals**2)
ss_total = np.sum((np.log(infectados) - np.mean(np.log(infectados)))**2)
r = 1 - (ss_residuals / ss_total)

curva_cuadratica = b * np.power(dias, a)
plt.plot(dias, curva_cuadratica, label=f'Curva Cuadrática: y = {b:.2f}x^{a:.2f} (r = {r:.2f})', color='green')

# Ajustar una curva exponencial (y = be^(ax))
a, b, r, _, _ = linregress(dias, np.log(infectados))
curva_exponencial = b * np.exp(a * dias)

plt.plot(dias, curva_exponencial, label=f'Curva Exponencial: y = {b:.2f}e^({a:.2f}x) (r = {r:.2f})', color='purple')

# Calcular el tiempo de duplicación
tiempo_duplicacion = np.log(2) / a

# Configurar el gráfico
plt.xlabel('Días')
plt.ylabel('Infectados')
plt.title('Curvas de Tendencia de Infectados')
plt.legend()

# Mostrar el gráfico
plt.show()


# Calcular la primera derivada numéricamente
primera_derivada = np.gradient(curva_cuadratica, dias)

# Calcular la segunda derivada numéricamente
segunda_derivada = np.gradient(primera_derivada, dias)

# Crear el gráfico de la primera derivada
plt.figure()
plt.plot(dias, primera_derivada)
plt.xlabel('Días')
plt.ylabel('Primera Derivada')
plt.title('Primera Derivada de la Curva Cuadrática')
plt.show()

# Crear el gráfico de la segunda derivada
plt.figure()
plt.plot(dias, segunda_derivada)
plt.xlabel('Días')
plt.ylabel('Segunda Derivada')
plt.title('Segunda Derivada de la Curva Cuadrática')
plt.show()

# Superponer la primera derivada en el gráfico existente
plt.plot(dias, curva_cuadratica, label='Curva Cuadrática', color='blue')
plt.plot(dias, primera_derivada, label='Primera Derivada', color='orange')

# Superponer la segunda derivada en el gráfico existente
plt.plot(dias, segunda_derivada, label='Segunda Derivada', color='yellow')

# Configurar el gráfico
plt.xlabel('Días')
plt.ylabel('Infectados')
plt.title('Curva Cuadrática con Derivadas')
plt.legend()

# Mostrar el gráfico
plt.show()

print("Trabajo Práctico Cuadrados mínimos y regresión lineal")
print("----------------------------------------------------")
print("Integrantes:")
print("-Tomás Venturini")
print("-Agustín Narvaez")
print("----------------------------------------------------")
print('Discusiones sobre los resultados obtenidos:')
print("----------------------------------------------------")
print("Tiempo de duplicación: {:.2f} días".format(tiempo_duplicacion))
print("----------------------------------------------------")
print('1. Las curvas nos proporciona distintas aproximaciones de la evolución de los casos infectados a lo largo del tiempo.')
print('2  Los gráficos de la primera y segunda derivada de la curva cuadrática nos permiten visualizar cómo cambia la tasa de cambio de los casos infectados a medida que pasa el tiempo.')
print('3. Al superponer las curvas podemos tener representación visual de la relación entre la curva de casos infectados, su tasa de cambio y su aceleración.')
print('4. Podemos entonces concluir con la observación de las curvas, que los infectados crecen de forma exponencial, pero que su tasa de cambio y aceleración se reducen con el tiempo, por lo que a medida que pase el tiempo la curva de infectados se irá aplanando. ')
print('5. Además, el tiempo de duplicación de los casos infectados es de 12.85 días, lo que significa que cada 12.85 días se duplica el número de casos infectados')
print('6. Finalmente, podemos concluiur que queda demostrado el poder del crecimiento exponencial y para este caso en particular, lo necesario que es para la salud pública actuar rápidamente en términos sanitarios para aplanar la curva y disminuir todos estos factores.')