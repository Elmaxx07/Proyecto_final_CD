import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

# ==============================================================
# Paso 1: CARGA DE DATOS
# ==============================================================
df = pd.read_csv('ejercicio4-inegi-MaximoRodriguez_procesado.csv')
print('Paso 1: datos cargados desde ejercicio4-inegi-MaximoRodriguez_procesado.csv')
print('-' * 80)

# ==============================================================
# Paso 2: PREPARACIÓN DE VARIABLES
# ==============================================================
# x1 = fecha_alta (años)
# x2 = latitud
# y  = per_ocu (personal ocupado)
df_clean = df[['fecha_alta', 'latitud', 'per_ocu']].dropna()
df_clean['per_ocu'] = pd.to_numeric(df_clean['per_ocu'], errors='coerce')
df_clean = df_clean.dropna(subset=['per_ocu'])
X = df_clean[['fecha_alta', 'latitud']].values
y = df_clean['per_ocu'].values
print('Paso 2: variables preparadas y valores no numéricos eliminados')
print('-' * 80)

# ==============================================================
# Paso 3: ENTRENAMIENTO DEL MODELO
# ==============================================================
modelo = LinearRegression()
modelo.fit(X, y)
print('Paso 3: modelo entrenado')
print('-' * 80)

# ==============================================================
# Paso 4: COEFICIENTES
# ==============================================================
a0 = modelo.intercept_          # Término independiente
a1 = modelo.coef_[0]            # Coeficiente para fecha_alta
a2 = modelo.coef_[1]            # Coeficiente para latitud
print(f'Paso 4: coeficientes calculados -> intercepto: {a0:.4f}, coef fecha_alta: {a1:.4f}, coef latitud: {a2:.4f}')
print('-' * 80)

# ==============================================================
# Paso 5: ECUACIÓN DE PREDICCIÓN
# ==============================================================
print('Paso 5: ecuación de predicción construida')
print(f'  per_ocu = {a0:.4f} + {a1:.4f}*(fecha_alta) + {a2:.4f}*(latitud)')
print('-' * 80)

# ==============================================================
# Paso 6: CÁLCULO DE PREDICCIONES INTERNAS
# ============================================================== 
y_pred = modelo.predict(X)
print('Paso 6: predicciones internas sobre los datos de entrenamiento calculadas')
print('-' * 80)

# ==============================================================
# Paso 7: CÁLCULO DE RESIDUALES
# ============================================================== 
residuals = y - y_pred
print('Paso 7: residuales calculados')
print('-' * 80)

# ==============================================================
# Paso 8: ORDENAR RESIDUALES
# ============================================================== 
residuals_sorted = np.sort(residuals)
print('Paso 8: residuales ordenados de menor a mayor')
print(f'  Total de residuales: {len(residuals_sorted)}')
print('  Primeros 10 residuales (más pequeños):')
for value in residuals_sorted[:10]:
    print(f'    {value:.4f}')
print('  Últimos 10 residuales (más grandes):')
for value in residuals_sorted[-10:]:
    print(f'    {value:.4f}')
print('-' * 80)

# ==============================================================
# Paso 9: PREDICCIONES SOLICITADAS
# ============================================================== 
print('Paso 9: predicciones solicitadas para valores nuevos')
predicciones = [
    (2010, 21.88),
    (2015, 22.14),
    (2020, 21.86)
]
for ano, latitud in predicciones:
    pred = modelo.predict([[ano, latitud]])[0]
    # Filtrar valores reales cercanos
    mask = (df_clean['fecha_alta'] == ano) & (df_clean['latitud'].between(latitud - 0.1, latitud + 0.1))
    real_avg = df_clean.loc[mask, 'per_ocu'].mean() if mask.any() else np.nan
    real_count = mask.sum()
    real_str = f"{real_avg:.2f}" if not pd.isna(real_avg) else "-"
    print(f'  Año: {ano:<4} | Latitud: {latitud:<6} | predicho: {pred:.2f} | real: {real_str} | n: {real_count}')
print('-' * 80)

# ==============================================================
# Paso 10: DASHBOARD DE GRÁFICAS
# ============================================================== 
print('Paso 10: creando dashboard de 4 gráficas')
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Paso 10: Dashboard de diagnóstico de regresión', fontsize=18, weight='bold')

# Gráfica 1: Residuales vs predichos 
axs[0, 0].scatter(y_pred, residuals, color='tab:blue', alpha=0.5, s=2)  
axs[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axs[0, 0].set_title('Residuales vs valores predichos')
axs[0, 0].set_xlabel('Valores predichos')
axs[0, 0].set_ylabel('Residuales')
axs[0, 0].grid(alpha=0.3)

# Gráfica 2: QQ plot de los residuales 
sm.qqplot(residuals, line='45', ax=axs[0, 1])
axs[0, 1].set_title('QQ plot de los residuales')
axs[0, 1].set_xlim(-5, 5)  
axs[0, 1].grid(alpha=0.3)

# Gráfica 3: Dispersión fecha_alta vs per_ocu
axs[1, 0].scatter(
    df_clean['fecha_alta'],
    df_clean['per_ocu'],
    color='tab:green',
    alpha=0.3,
    s=2
)

# Línea de tendencia
z1 = np.polyfit(df_clean['fecha_alta'], df_clean['per_ocu'], 1)
p1 = np.poly1d(z1)

axs[1, 0].plot(
    df_clean['fecha_alta'],
    p1(df_clean['fecha_alta']),
    color='red',
    linewidth=2
)

axs[1, 0].set_title('fecha_alta vs per_ocu')
axs[1, 0].set_xlabel('fecha_alta')
axs[1, 0].set_ylabel('per_ocu')
axs[1, 0].grid(alpha=0.3)

# Gráfica 4: Dispersión latitud vs per_ocu
axs[1, 1].scatter(
    df_clean['latitud'],
    df_clean['per_ocu'],
    color='tab:purple',
    alpha=0.3,
    s=2
)

# Línea de tendencia
z2 = np.polyfit(df_clean['latitud'], df_clean['per_ocu'], 1)
p2 = np.poly1d(z2)

axs[1, 1].plot(
    df_clean['latitud'],
    p2(df_clean['latitud']),
    color='red',
    linewidth=2
)

axs[1, 1].set_title('latitud vs per_ocu')
axs[1, 1].set_xlabel('latitud')
axs[1, 1].set_ylabel('per_ocu')
axs[1, 1].grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
print('Paso 10: dashboard listo, se mostrará en pantalla')
print('-' * 80)

# ==============================================================
# Paso 11: RESIDUALES ORDENADOS 
# ============================================================== 
print('Paso 11: residuales ordenados listos para revisión')
print(f'  Residuales mínimos: {residuals_sorted[:3]}')
print(f'  Residuales máximos: {residuals_sorted[-3:]}')
print('-' * 80)

# ==============================================================
# Paso 12: DESVIACIÓN ESTÁNDAR DE LOS RESIDUALES
# ============================================================== 
residuals_std = np.std(residuals, ddof=1)
print('Paso 12: desviación estándar de los residuales')
print(f'  Desviación estándar: {residuals_std:.4f}')
print('-' * 80)

# ==============================================================
# Paso 13: CORRELACIÓN DE PEARSON
# ============================================================== 
pearson_corr, pearson_pvalue = stats.pearsonr(y, y_pred)
if pearson_corr >= 0.7:
    pearson_desc = 'relación fuerte positiva'
elif pearson_corr >= 0.3:
    pearson_desc = 'relación positiva moderada'
elif pearson_corr >= 0.0:
    pearson_desc = 'relación débil positiva'
elif pearson_corr <= -0.7:
    pearson_desc = 'relación fuerte negativa'
elif pearson_corr <= -0.3:
    pearson_desc = 'relación negativa moderada'
else:
    pearson_desc = 'sin relación clara'
print('Paso 13: correlación de Pearson')
print(f'  Coeficiente de Pearson: {pearson_corr:.4f}')
print(f'  Interpretación: {pearson_desc}')
print(f'  p-valor: {pearson_pvalue:.6g}')
# ==============================================================
# CORRELACIONES INDIVIDUALES
# ==============================================================

corr_x1_y = df_clean['fecha_alta'].corr(df_clean['per_ocu'])
corr_x2_y = df_clean['latitud'].corr(df_clean['per_ocu'])
corr_x1_x2 = df_clean['fecha_alta'].corr(df_clean['latitud'])

print('Correlación X1 (fecha_alta) con Y:', corr_x1_y)
print('Correlación X2 (latitud) con Y:', corr_x2_y)
print('Correlación entre X1 y X2:', corr_x1_x2)
print('-' * 80)
print('-' * 80)

# ==============================================================
# Paso 14: R² (COEFICIENTE DE DETERMINACIÓN)
# ============================================================== 
r_squared = modelo.score(X, y)
print('Paso 14: coeficiente de determinación R²')
print(f'  R²: {r_squared:.4f}')
print('-' * 80)

# ==============================================================
# Paso 15: RMSE
# ============================================================== 
rmse = np.sqrt(np.mean(residuals ** 2))
print('Paso 15: error cuadrático medio (RMSE)')
print(f'  RMSE: {rmse:.4f}')
print('-' * 80)

# ==============================================================
# Paso 16: REPORTE OLS COMPLETO
# ============================================================== 
X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()
print('Paso 16: reporte OLS completo')
print(ols_model.summary())
print('-' * 80)

# ==============================================================
# Paso 17: RESUMEN ADICIONAL OLS
# ============================================================== 
print('Paso 17: estadísticas adicionales del modelo OLS')
print(f'  F-statistic: {ols_model.fvalue:.4f}')
print(f'  Prob(F-statistic): {ols_model.f_pvalue:.4g}')
print(f'  Durbin-Watson: {durbin_watson(residuals):.4f}')
print(f'  Número de observaciones: {int(ols_model.nobs)}')

# P-valores de los coeficientes
p_x1 = ols_model.pvalues[1]  # P-valor para fecha_alta (X1)
p_x2 = ols_model.pvalues[2]  # P-valor para latitud (X2)

print(f'  P-valor para X1 (fecha_alta): {p_x1:.4g}')
print(f'    ¿Es menor a 0.05? {"Sí" if p_x1 < 0.05 else "No"}')
print(f'    Significado: {"El coeficiente de fecha_alta es estadísticamente significativo" if p_x1 < 0.05 else "El coeficiente de fecha_alta no es estadísticamente significativo"}')

print(f'  P-valor para X2 (latitud): {p_x2:.4g}')
print(f'    ¿Es menor a 0.05? {"Sí" if p_x2 < 0.05 else "No"}')
print(f'    Significado: {"El coeficiente de latitud es estadísticamente significativo" if p_x2 < 0.05 else "El coeficiente de latitud no es estadísticamente significativo"}')

print(f'  P-valor del F-statistic: {ols_model.f_pvalue:.4g}')
print(f'  Modelo completo estadísticamente significativo: {"Sí" if ols_model.f_pvalue < 0.05 else "No"} (si p < 0.05, el modelo explica variabilidad significativa)')
print('-' * 80)

# ==============================================================
# Paso 18: DURBIN-WATSON
# ============================================================== 
dw_value = durbin_watson(residuals)
if 1.5 <= dw_value <= 2.5:
    dw_desc = 'indica ausencia de autocorrelación significativa'
else:
    dw_desc = 'indica posible autocorrelación'
print('Paso 18: prueba Durbin-Watson')
print(f'  Durbin-Watson: {dw_value:.4f}')
print(f'  Interpretación: {dw_desc}')
print('-' * 80)

# ==============================================================
# Paso 19: BREUSCH-PAGAN
# ============================================================== 
bp_test = het_breuschpagan(residuals, X_sm)
bp_pvalue = bp_test[1]
if bp_pvalue > 0.05:
    bp_desc = 'homocedasticidad aceptada (sin evidencia fuerte de heterocedasticidad)'
else:
    bp_desc = 'heterocedasticidad presente (evidencia de varianza no constante)'
print('Paso 19: prueba de Breusch-Pagan')
print(f'  p-valor: {bp_pvalue:.6g}')
print(f'  Interpretación: {bp_desc}')
print('-' * 80)

plt.show()
