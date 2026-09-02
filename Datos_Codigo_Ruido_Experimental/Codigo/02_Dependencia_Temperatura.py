# librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.lines import Line2D

#guardar datos

CARPETA_CODIGO = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA_CODIGO.parent / "Datos"
CARPETA_ENTREGA = CARPETA_CODIGO.parent

#Datos

kB_ref = 1.380649e-23
G1 = 600.0
G2 = 1000.0
delta_f = 9997.0
u_rel_delta_f = 0.05
K_squarer = 10.0
R_10k = 10000.0
R_100k = 100000.0
u_rel_R = 0.01
u_rel_G1 = np.sqrt(0.002**2 + 0.004**2)
u_rel_G2 = 0.002
u_rel_squarer = 0.002
u_rel_average = 0.001
V_cal = np.array([825, 835, 845, 855, 865, 875, 895, 900, 945, 955, 963], dtype=float)
T_cal = np.array(
    [149.6, 145.6, 141.6, 137.5, 133.4, 129.3, 121.1, 119.0, 99.8, 95.3, 91.8],
    dtype=float,
)
u_V_sensor = 1.0

#Ajuste de voltaje a temperatura
def voltaje_a_temperatura(V):
    return np.interp(V, V_cal, T_cal)


dTdV_cal = np.gradient(T_cal, V_cal)



def incertidumbre_temperatura(V):
    dTdV = np.interp(V, V_cal, dTdV_cal)
    return abs(dTdV) * u_V_sensor

#Datos

datos = {
    825: {
        "A": [640, 634, 622, 621, 617],
        "B": [658, 625, 622, 630, 629],
        "C": [629, 633, 620, 635, 622],
    },
    835: {
        "A": [153, 128, 128, 115, 110],
        "B": [482, 478, 476, 488, 500],
        "C": [579, 593, 607, 626, 637],
    },
    845: {
        "A": [600, 607, 606, 617, 619],
        "B": [640, 628, 622, 621, 633],
        "C": [640, 635, 634, 641, 635],
    },
    855: {
        "A": [417, 419, 424, 416, 418],
        "B": [535, 553, 551, 579, 567],
        "C": [642, 636, 636, 631, 637],
    },
    865: {
        "A": [370, 357, 347, 378, 363],
        "B": [497, 496, 498, 502, 519],
        "C": [529, 537, 550, 559, 541],
    },
    875: {
        "A": [250, 300, 283, 276, 273],
        "B": [554, 543, 544, 540, 490],
        "C": [599, 614, 619, 617, 622],
    },
    895: {
        "A": [205, 208, 214, 213, 211],
        "B": [237, 233, 238, 232, 248],
        "C": [295, 296, 324, 297, 330],
    },
    900: {
        "A": [322, 286, 288, 266],
        "B": [437, 429, 443, 448, 440],
        "C": [555, 574, 590, 580, 600],
    },
    945: {
        "A": [71.1, 76.5, 85.8, 81.2, 72.5],
        "B": [120.9, 117.2, 116.5, 109.0, 115.3],
        "C": [130.7, 139.5, 151.5, 144.0, 147.3],
    },
    955: {
        "A": [10.9, 11.9, 11.6, 11.0, 10.9],
        "B": [16.3, 16.5, 15.1, 17.4, 18.1],
        "C": [32.0, 31.5, 31.1, 31.7, 30.7],
    },
    963: {
        "A": [11.2, 11.2, 11.3, 11.4, 11.5],
        "B": [15.9, 14.4, 13.4, 13.3, 13.2],
        "C": [30.0, 29.0, 28.5, 28.6, 30.0],
    },
}


def u_instrumental_ruido(valor_mV):
    if abs(valor_mV) >= 100:
        return 1.0
    return 0.1

# Procesamiento de Datos
filas_crudas = []
filas_resumen = []
filas_procesadas = []
T_lista = []
uT_lista = []
V10_lista = []
uV10_lista = []
V100_lista = []
uV100_lista = []
for V_sensor, mediciones in datos.items():
    T = voltaje_a_temperatura(V_sensor)
    uT = incertidumbre_temperatura(V_sensor)
    resultados = {}
    for canal in ["A", "B", "C"]:
        x = np.array(mediciones[canal], dtype=float)
        N = len(x)
        promedio = np.mean(x)
        s = np.std(x, ddof=1)
        u_media = s / np.sqrt(N)
        u_inst = u_instrumental_ruido(promedio)
        u_total = np.sqrt(u_media**2 + u_inst**2)
        resultados[canal] = {
            "promedio": promedio,
            "s": s,
            "u_media": u_media,
            "u_inst": u_inst,
            "u_total": u_total,
        }
        for i, valor in enumerate(x, start=1):
            filas_crudas.append(
                {
                    "V_sensor (mV)": V_sensor,
                    "u(V_sensor) (mV)": u_V_sensor,
                    "T (K)": T,
                    "u(T) (K)": uT,
                    "Canal": canal,
                    "Medicion": i,
                    "V_ruido (mV)": valor,
                }
            )
        filas_resumen.append(
            {
                "V_sensor (mV)": V_sensor,
                "T (K)": T,
                "u(T) (K)": uT,
                "Canal": canal,
                "N": N,
                "Promedio (mV)": promedio,
                "s (mV)": s,
                "u_promedio (mV)": u_media,
                "u_inst (mV)": u_inst,
                "u_total (mV)": u_total,
            }
        )
    VA = resultados["A"]["promedio"]
    VB = resultados["B"]["promedio"]
    VC = resultados["C"]["promedio"]
    uA = resultados["A"]["u_total"]
    uB = resultados["B"]["u_total"]
    uC = resultados["C"]["u_total"]
    Vcorr_10k = (VB - VA) / 1000
    u_corr_10k = np.sqrt(uA**2 + uB**2) / 1000
    Vcorr_100k = (VC - VA) / 1000
    u_corr_100k = np.sqrt(uA**2 + uC**2) / 1000
    T_lista.append(T)
    uT_lista.append(uT)
    V10_lista.append(Vcorr_10k)
    uV10_lista.append(u_corr_10k)
    V100_lista.append(Vcorr_100k)
    uV100_lista.append(u_corr_100k)
    filas_procesadas.append(
        {
            "V_sensor (mV)": V_sensor,
            "T (K)": T,
            "u(T) (K)": uT,
            "Vcorr_10k (V)": Vcorr_10k,
            "u(Vcorr_10k) (V)": u_corr_10k,
            "Vcorr_100k (V)": Vcorr_100k,
            "u(Vcorr_100k) (V)": u_corr_100k,
        }
    )
T_array = np.array(T_lista)
uT_array = np.array(uT_lista)
V10 = np.array(V10_lista)
uV10 = np.array(uV10_lista)
V100 = np.array(V100_lista)
uV100 = np.array(uV100_lista)
idx = np.argsort(T_array)
T_array = T_array[idx]
uT_array = uT_array[idx]
V10 = V10[idx]
uV10 = uV10[idx]
V100 = V100[idx]
uV100 = uV100[idx]

#Ajuste de datos
def modelo(T, m, b):
    return m * T + b


def ajuste_xy(T, V, uT, uV, max_iter=100):
    m, b = np.polyfit(T, V, 1)
    for _ in range(max_iter):
        sigma_eff = np.sqrt(uV**2 + (m * uT) ** 2)
        popt, pcov = curve_fit(modelo, T, V, sigma=sigma_eff, absolute_sigma=True)
        m_nuevo, b_nuevo = popt
        if np.isclose(m_nuevo, m, rtol=1e-10, atol=1e-15):
            m = m_nuevo
            b = b_nuevo
            break
        m = m_nuevo
        b = b_nuevo
    sigma_eff = np.sqrt(uV**2 + (m * uT) ** 2)
    popt, pcov = curve_fit(modelo, T, V, sigma=sigma_eff, absolute_sigma=True)
    m, b = popt
    um_interno, ub_interno = np.sqrt(np.diag(pcov))
    V_pred = modelo(T, m, b)
    residuos = V - V_pred
    residuos_norm = residuos / sigma_eff
    SS_res = np.sum(residuos**2)
    SS_tot = np.sum((V - np.mean(V)) ** 2)
    R2 = 1 - SS_res / SS_tot
    chi2 = np.sum(residuos_norm**2)
    gl = len(T) - 2
    chi2_red = chi2 / gl
    factor_birge = np.sqrt(max(1.0, chi2_red))
    um = um_interno * factor_birge
    ub = ub_interno * factor_birge
    return {
        "m": m,
        "um": um,
        "um_interno": um_interno,
        "b": b,
        "ub": ub,
        "ub_interno": ub_interno,
        "factor_birge": factor_birge,
        "sigma_eff": sigma_eff,
        "residuos": residuos,
        "residuos_norm": residuos_norm,
        "R2": R2,
        "chi2_red": chi2_red,
    }


ajuste_10k = ajuste_xy(T_array, V10, uT_array, uV10)
ajuste_100k = ajuste_xy(T_array, V100, uT_array, uV100)

# calcular kB
def calcular_kB(ajuste, R):
    m = ajuste["m"]
    um = ajuste["um"]
    kB = K_squarer * m / (4 * R * delta_f * (G1 * G2) ** 2)
    u_rel_kB = np.sqrt(
        (um / m) ** 2
        + u_rel_R**2
        + u_rel_delta_f**2
        + (2 * u_rel_G1) ** 2
        + (2 * u_rel_G2) ** 2
        + u_rel_squarer**2
        + u_rel_average**2
    )
    u_kB = abs(kB) * u_rel_kB
    z = abs(kB - kB_ref) / u_kB
    return (kB, u_kB, z)

# Plotear
kB_10k, ukB_10k, z_10k = calcular_kB(ajuste_10k, R_10k)
kB_100k, ukB_100k, z_100k = calcular_kB(ajuste_100k, R_100k)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "Latin Modern Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
    }
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 8.3),
    sharex=True,
    gridspec_kw={
        "height_ratios": [3.0, 1.7],
        "hspace": 0.06,
    },
)

color_10k = "#0072B2"
color_100k = "#D55E00"

# Datos experimentales
ax1.errorbar(
    T_array,
    V10,
    xerr=uT_array,
    yerr=uV10,
    fmt="o",
    color=color_10k,
    ecolor=color_10k,
    markerfacecolor=color_10k,
    markersize=4,
    elinewidth=1.0,
    capsize=3,
    capthick=1.0,
    label=r"$10\,\mathrm{k}\Omega$",
    zorder=5,
)

ax1.errorbar(
    T_array,
    V100,
    xerr=uT_array,
    yerr=uV100,
    fmt="s",
    color=color_100k,
    ecolor=color_100k,
    markerfacecolor=color_100k,
    markersize=4,
    elinewidth=1.0,
    capsize=3,
    capthick=1.0,
    label=r"$100\,\mathrm{k}\Omega$",
    zorder=5,
)

T_linea = np.linspace(T_array.min(), T_array.max(), 500)

# Predicción teórica de Johnson--Nyquist con valores nominales
m_teorica_10k = (
    4 * kB_ref * R_10k * delta_f * (G1 * G2) ** 2 / K_squarer
)
m_teorica_100k = (
    4 * kB_ref * R_100k * delta_f * (G1 * G2) ** 2 / K_squarer
)

ax1.plot(
    T_linea,
    m_teorica_10k * T_linea,
    color=color_10k,
    linestyle=":",
    linewidth=1.2,
)
ax1.plot(
    T_linea,
    m_teorica_100k * T_linea,
    color=color_100k,
    linestyle=":",
    linewidth=1.2,
)

# Ajustes experimentales
ax1.plot(
    T_linea,
    modelo(T_linea, ajuste_10k["m"], ajuste_10k["b"]),
    color=color_10k,
    linestyle="-",
    linewidth=1.0,
)
ax1.plot(
    T_linea,
    modelo(T_linea, ajuste_100k["m"], ajuste_100k["b"]),
    color=color_100k,
    linestyle="-",
    linewidth=1.0,
)

# Residuales normalizados
ax2.scatter(
    T_array,
    ajuste_10k["residuos_norm"],
    color=color_10k,
    marker="o",
    s=22,
    zorder=5,
)
ax2.scatter(
    T_array,
    ajuste_100k["residuos_norm"],
    color=color_100k,
    marker="s",
    s=22,
    zorder=5,
)

ax1.set_ylabel(r"$V_{\mathrm{corr}}$ (V)")
ax1.grid(
    color="gray",
    alpha=0.14,
    linewidth=0.40,
    linestyle="-",
    which="both",
)

ax2.axhline(0, color="black", linewidth=0.8)
ax2.axhline(1, color="gray", linestyle="--", linewidth=0.7)
ax2.axhline(-1, color="gray", linestyle="--", linewidth=0.7)
ax2.set_yscale("symlog", linthresh=1)

residual_max = max(
    np.max(np.abs(ajuste_10k["residuos_norm"])),
    np.max(np.abs(ajuste_100k["residuos_norm"])),
)
ax2.set_ylim(-1.50 * residual_max, 1.50 * residual_max)
ax2.set_xlim(T_array.min() - 4, T_array.max() + 4)
ax2.set_xlabel(r"Temperatura $T$ (K)")
ax2.set_ylabel("Residual normalizado")
ax2.xaxis.set_major_locator(MultipleLocator(10))
ax2.grid(
    color="gray",
    alpha=0.14,
    linewidth=0.40,
    linestyle="-",
    which="both",
)

leyenda_modelos = [
    Line2D([0], [0], color="black", linestyle=":", linewidth=1.2, label="Teórico"),
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.0, label="Ajuste"),
]

legend_modelos = ax1.legend(
    handles=leyenda_modelos,
    fontsize=9,
    loc="upper left",
    bbox_to_anchor=(0.0, 0.83),
    frameon=True,
)
ax1.add_artist(legend_modelos)

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(
    handles,
    labels,
    fontsize=9,
    title=r"Resistencia $R$",
    title_fontsize=10,
    loc="upper left",
    frameon=True,
)

plt.savefig(CARPETA_ENTREGA / "figura2.png", dpi=300, bbox_inches="tight")
plt.savefig(CARPETA_ENTREGA / "figura2.pdf", bbox_inches="tight")
plt.show()
dataset_crudo = pd.DataFrame(filas_crudas)
tabla_resumen = pd.DataFrame(filas_resumen)
tabla_procesada = pd.DataFrame(filas_procesadas)
tabla_ajustes = pd.DataFrame(
    {
        "R": ["10 kOhm", "100 kOhm"],
        "ENBW (Hz)": [delta_f, delta_f],
        "m (V/K)": [ajuste_10k["m"], ajuste_100k["m"]],
        "u(m) (V/K)": [ajuste_10k["um"], ajuste_100k["um"]],
        "b (V)": [ajuste_10k["b"], ajuste_100k["b"]],
        "u(b) (V)": [ajuste_10k["ub"], ajuste_100k["ub"]],
        "kB (J/K)": [kB_10k, kB_100k],
        "u(kB) (J/K)": [ukB_10k, ukB_100k],
        "R2": [ajuste_10k["R2"], ajuste_100k["R2"]],
        "chi2_red": [ajuste_10k["chi2_red"], ajuste_100k["chi2_red"]],
        "z": [z_10k, z_100k],
    }
)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
print("\n\n========== CONVERSIÓN VOLTAJE -> TEMPERATURA ==========\n")
print(tabla_procesada[["V_sensor (mV)", "T (K)", "u(T) (K)"]].to_string(index=False))
print("\n\n========== DATOS PROCESADOS ==========\n")
print(tabla_procesada.to_string(index=False))
print("\n\n========== PROMEDIOS E INCERTIDUMBRES ==========\n")
print(tabla_resumen.to_string(index=False))
print("\n\n========== RESULTADOS DE LOS AJUSTES ==========\n")
print(tabla_ajustes.to_string(index=False))
print("\nValor de referencia:")
print(f"kB = {kB_ref:.6e} J/K")
dataset_crudo.to_csv(CARPETA_DATOS / "temperatura_01_dataset.csv", index=False)
tabla_resumen.to_csv(CARPETA_DATOS / "temperatura_02_promedios.csv", index=False)
tabla_procesada.to_csv(CARPETA_DATOS / "temperatura_03_datos_procesados.csv", index=False)
tabla_ajustes.to_csv(CARPETA_DATOS / "temperatura_04_ajustes_kB.csv", index=False)
