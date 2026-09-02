# Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from matplotlib.lines import Line2D


#Guardar datos

CARPETA_ENTREGA = Path(__file__).resolve().parent.parent
CARPETA_DATOS = CARPETA_ENTREGA / "Datos"

#Datos

kB_ref = 1.380649e-23
T = 300.0
u_T = 1.0
G1 = 600.0
G2 = 300.0
K_squarer = 10.0
u_rel_R = 0.001
u_rel_g6 = 0.002
u_rel_g100 = 0.004
u_rel_G1 = np.sqrt(u_rel_g6**2 + u_rel_g100**2)
u_rel_G2 = 0.002
u_rel_ENBW = 0.05
u_rel_squarer = 0.002
u_rel_average = 0.001

ENBW = {
    "1–3.3 kHz": 2576.0,
    "1–10 kHz": 9997.0,
    "1–33 kHz": 35543.0,
    "1–100 kHz": 109961.0,
    "100 Hz–100 kHz": 110961.0,
    "30 Hz–100 kHz": 111039.0,
}

datos = {
    "1–3.3 kHz": {
        1: [-3.6, -3.6, -3.6, -3.6, -3.6],
        10: [-3.6, -3.6, -3.6, -3.6, -3.5],
        100: [-3.5, -3.6, -3.5, -3.5, -3.5],
        1000.0: [-3.4, -3.5, -3.5, -3.3, -3.5],
        10000.0: [-2.2, -2.2, -2.2, -2.1, -2.3],
        100000.0: [10.3, 10.0, 10.1, 10.0, 10.1],
        1000000.0: [124.0, 124.4, 123.8, 125.0, 123.1],
    },
    "1–10 kHz": {
        1: [-2.0, -2.0, -2.0, -2.0, -2.0],
        10: [-2.0, -2.0, -2.0, -2.0, -2.0],
        100: [-1.9, -1.9, -1.9, -1.9, -1.9],
        1000.0: [-1.5, -1.5, -1.5, -1.5, -1.5],
        10000.0: [3.2, 3.2, 3.3, 3.2, 3.3],
        100000.0: [50.6, 50.8, 50.9, 50.5, 50.2],
        1000000.0: [368, 364, 367, 365, 360],
    },
    "1–33 kHz": {
        1: [3.4, 3.5, 3.4, 3.4, 3.4],
        10: [3.4, 3.5, 3.4, 3.4, 3.5],
        100: [3.7, 3.8, 4.0, 3.9, 3.8],
        1000.0: [5.4, 5.4, 5.3, 5.5, 5.4],
        10000.0: [22.8, 22.7, 22.7, 22.8, 22.7],
        100000.0: [184.2, 184.1, 184.2, 183.7, 184.2],
        1000000.0: [654, 655, 653, 650, 652],
    },
    "1–100 kHz": {
        1: [19.8, 19.8, 19.8, 19.8, 19.8],
        10: [19.8, 19.8, 19.9, 19.9, 19.8],
        100: [20.4, 20.4, 20.3, 20.4, 20.4],
        1000.0: [25.8, 25.9, 25.9, 25.8, 25.9],
        10000.0: [80.7, 80.6, 80.7, 80.4, 80.5],
        100000.0: [605, 598, 599, 601, 604],
        1000000.0: [987, 983, 978, 987, 992],
    },
    "100 Hz–100 kHz": {
        1: [20.1, 20.2, 20.2, 20.2, 20.2],
        10: [20.3, 20.3, 20.3, 20.3, 20.2],
        100: [20.8, 20.7, 20.8, 20.9, 20.8],
        1000.0: [26.2, 26.3, 26.2, 26.2, 26.3],
        10000.0: [81.3, 81.4, 81.2, 81.1, 81.2],
        100000.0: [478, 477, 477, 477, 475],
        1000000.0: [970, 986, 980, 982, 973],
    },
    "30 Hz–100 kHz": {
        1: [20.2, 20.2, 20.3, 20.2, 20.2],
        10: [20.3, 20.3, 20.3, 20.3, 20.2],
        100: [20.8, 20.8, 20.9, 20.8, 20.8],
        1000.0: [26.3, 26.2, 26.3, 26.3, 26.3],
        10000.0: [81.3, 81.2, 81.4, 81.4, 81.5],
        100000.0: [477, 478, 477, 478, 478],
        1000000.0: [969, 984, 987, 981, 985],
    },
}

u_inst = {
    "1–3.3 kHz": {
        1: 0.1,
        10: 0.1,
        100: 0.1,
        1000.0: 0.1,
        10000.0: 0.1,
        100000.0: 0.1,
        1000000.0: 0.1,
    },
    "1–10 kHz": {
        1: 0.1,
        10: 0.1,
        100: 0.1,
        1000.0: 0.1,
        10000.0: 0.1,
        100000.0: 0.1,
        1000000.0: 1.0,
    },
    "1–33 kHz": {
        1: 0.1,
        10: 0.1,
        100: 0.1,
        1000.0: 0.1,
        10000.0: 0.1,
        100000.0: 0.1,
        1000000.0: 1.0,
    },
    "1–100 kHz": {
        1: 0.1,
        10: 0.1,
        100: 0.1,
        1000.0: 0.1,
        10000.0: 0.1,
        100000.0: 1.0,
        1000000.0: 1.0,
    },
    "100 Hz–100 kHz": {
        1: 0.1,
        10: 0.1,
        100: 0.1,
        1000.0: 0.1,
        10000.0: 0.1,
        100000.0: 1.0,
        1000000.0: 1.0,
    },
    "30 Hz–100 kHz": {
        1: 0.1,
        10: 0.1,
        100: 0.1,
        1000.0: 0.1,
        10000.0: 0.1,
        100000.0: 1.0,
        1000000.0: 1.0,
    },
}

orden = [
    "1–3.3 kHz",
    "1–10 kHz",
    "1–33 kHz",
    "1–100 kHz",
    "100 Hz–100 kHz",
    "30 Hz–100 kHz",
]


# DATASET COMPLETO

filas_dataset = []

for banda in orden:
    for R, lecturas in datos[banda].items():
        for i, valor in enumerate(lecturas, start=1):
            filas_dataset.append(
                {
                    "Banda": banda,
                    "ENBW (Hz)": ENBW[banda],
                    "R (ohm)": R,
                    "Medicion": i,
                    "V (mV)": valor,
                    "u_inst (mV)": u_inst[banda][R],
                }
            )

dataset_completo = pd.DataFrame(filas_dataset)

# PROMEDIOS E INCERTIDUMBRES

filas_resumen = []
procesados = {}

for banda in orden:

    medias = {}
    incertidumbres = {}

    for R, lecturas in datos[banda].items():

        x = np.array(lecturas, dtype=float)
        N = len(x)

        promedio = np.mean(x)
        s = np.std(x, ddof=1)
        u_media = s / np.sqrt(N)

        ui = u_inst[banda][R]

        u_total = np.sqrt(
            u_media**2 + ui**2
        )

        medias[R] = promedio / 1000
        incertidumbres[R] = u_total / 1000

        filas_resumen.append(
            {
                "Banda": banda,
                "R (ohm)": R,
                "N": N,
                "Promedio (mV)": promedio,
                "s (mV)": s,
                "u_promedio (mV)": u_media,
                "u_inst (mV)": ui,
                "u_total (mV)": u_total,
            }
        )

    Vref = medias[10]
    uref = incertidumbres[10]

    R_array = np.array(
        [
            100,
            1000.0,
            10000.0,
            100000.0,
            1000000.0,
        ],
        dtype=float,
    )

    Vcorr = np.array(
        [
            medias[R] - Vref
            for R in R_array
        ]
    )

    n = len(R_array)

    C = np.full(
        (n, n),
        uref**2
    )

    for i, R in enumerate(R_array):
        C[i, i] += incertidumbres[R] ** 2

    uVcorr = np.sqrt(np.diag(C))
    uR = u_rel_R * R_array

    procesados[banda] = {
        "R": R_array,
        "u_R": uR,
        "V": Vcorr,
        "u_V": uVcorr,
        "C": C,
    }

tabla_resumen = pd.DataFrame(filas_resumen)

# DATOS CORREGIDOS

filas_corregidos = []

for banda in orden:

    R = procesados[banda]["R"]
    V = procesados[banda]["V"]
    uV = procesados[banda]["u_V"]
    uR = procesados[banda]["u_R"]

    for Ri, uiR, Vi, uiV in zip(
        R,
        uR,
        V,
        uV
    ):

        filas_corregidos.append(
            {
                "Banda": banda,
                "R (ohm)": Ri,
                "u(R) (ohm)": uiR,
                "Vcorr (V)": Vi,
                "u(Vcorr) (V)": uiV,
                "Usado en ajuste":
                    "Si" if Ri <= 100000.0 else "No",
            }
        )

tabla_corregidos = pd.DataFrame(
    filas_corregidos
)

# AJUSTE LINEAL PONDERADO CON SCIPY

def modelo_lineal(R, m, b):
    return m * R + b
def ajuste_curve_fit(
    R,
    V,
    C_y,
    u_R,
    max_iter=100
):

    m, b = np.polyfit(R, V, 1)

    for _ in range(max_iter):

        C_eff = (
            C_y
            + np.diag(
                (m * u_R) ** 2
            )
        )

        popt, pcov = curve_fit(
            modelo_lineal,
            R,
            V,
            p0=(m, b),
            sigma=C_eff,
            absolute_sigma=True,
        )

        m_nuevo, b_nuevo = popt

        if np.isclose(
            m_nuevo,
            m,
            rtol=1e-12,
            atol=1e-20
        ):
            m = m_nuevo
            b = b_nuevo
            break

        m = m_nuevo
        b = b_nuevo

    C_eff = (
        C_y
        + np.diag(
            (m * u_R) ** 2
        )
    )

    beta, cov_beta = curve_fit(
        modelo_lineal,
        R,
        V,
        p0=(m, b),
        sigma=C_eff,
        absolute_sigma=True,
    )

    return (
        beta,
        cov_beta,
        C_eff
    )

# AJUSTES Y kB

ajustes = {}
filas_tabla_final = []
filas_1M = []

for banda in orden:

    R = procesados[banda]["R"]
    V = procesados[banda]["V"]
    C = procesados[banda]["C"]
    uR = procesados[banda]["u_R"]

    mask = R <= 100000.0

    R_fit = R[mask]
    V_fit = V[mask]
    uR_fit = uR[mask]

    C_fit = C[
        np.ix_(mask, mask)
    ]

    beta, cov_beta, C_eff = ajuste_curve_fit(
        R_fit,
        V_fit,
        C_fit,
        uR_fit
    )

    m, b = beta

    um, ub = np.sqrt(
        np.diag(cov_beta)
    )

    V_pred = (
        m * R_fit + b
    )

    residuos = (
        V_fit - V_pred
    )

    SS_res = np.sum(
        residuos**2
    )

    SS_tot = np.sum(
        (
            V_fit
            - np.mean(V_fit)
        ) ** 2
    )

    R2 = (
        1
        - SS_res / SS_tot
    )

    C_inv = np.linalg.inv(
        C_eff
    )

    chi2 = (
        residuos.T
        @ C_inv
        @ residuos
    )

    gl = len(R_fit) - 2

    chi2_red = (
        chi2 / gl
    )

    um_interno = um
    ub_interno = ub
    factor_birge = np.sqrt(
        max(1.0, chi2_red)
    )
    um = um_interno * factor_birge
    ub = ub_interno * factor_birge

    delta_f = ENBW[banda]
    u_delta_f = (
        u_rel_ENBW
        * delta_f
    )

    kB = (
        K_squarer
        * m
        / (
            4
            * T
            * delta_f
            * (G1 * G2) ** 2
        )
    )

    u_rel_kB = np.sqrt(
        (um / m) ** 2
        + (u_T / T) ** 2
        + u_rel_ENBW**2
        + (2 * u_rel_G1) ** 2
        + (2 * u_rel_G2) ** 2
        + u_rel_squarer**2
        + u_rel_average**2
    )

    u_kB = (
        abs(kB)
        * u_rel_kB
    )

    z = (
        abs(
            kB - kB_ref
        )
        / u_kB
    )

    ajustes[banda] = {
        "m": m,
        "um": um,
        "um_interno": um_interno,
        "b": b,
        "ub": ub,
        "ub_interno": ub_interno,
        "factor_birge": factor_birge,
        "R2": R2,
        "chi2_red": chi2_red,
        "residuos": residuos,
        "C_eff": C_eff,
        "kB": kB,
        "u_kB": u_kB,
        "z": z,
    }

    filas_tabla_final.append(
        {
            "Configuracion": banda,
            "ENBW (Hz)": delta_f,
            "u(ENBW) (Hz)": u_delta_f,
            "m (V/ohm)": m,
            "u(m) (V/ohm)": um,
            "b (V)": b,
            "u(b) (V)": ub,
            "kB (J/K)": kB,
            "u(kB) (J/K)": u_kB,
            "R2": R2,
            "chi2_reducido": chi2_red,
            "z": z,
        }
    )

    V_1M = V[-1]

    V_pred_1M = (
        m * 1000000.0
        + b
    )

    razon_1M = (
        V_1M
        / V_pred_1M
    )

    filas_1M.append(
        {
            "Configuracion": banda,
            "V_1M_medido (V)": V_1M,
            "V_1M_predicho (V)": V_pred_1M,
            "Medido/Predicho": razon_1M,
        }
    )

tabla_final = pd.DataFrame(
    filas_tabla_final
)

tabla_1M = pd.DataFrame(
    filas_1M
)

# TABLA

filas_formateadas = []

for _, fila in tabla_final.iterrows():

    filas_formateadas.append(
        {
            "Configuracion":
                fila["Configuracion"],

            "ENBW":
                f"{fila['ENBW (Hz)']:.0f} "
                f"± {fila['u(ENBW) (Hz)']:.0f} Hz",

            "Pendiente":
                f"({fila['m (V/ohm)']:.4e} "
                f"± {fila['u(m) (V/ohm)']:.2e}) V/ohm",

            "Intercepto":
                f"({fila['b (V)']:.4e} "
                f"± {fila['u(b) (V)']:.2e}) V",

            "kB":
                f"({fila['kB (J/K)']:.4e} "
                f"± {fila['u(kB) (J/K)']:.2e}) J/K",

            "R2":
                f"{fila['R2']:.6f}",

            "chi2_red":
                f"{fila['chi2_reducido']:.3f}",

            "z":
                f"{fila['z']:.3f}",
        }
    )

tabla_final_formateada = pd.DataFrame(
    filas_formateadas
)

# COLORES

colores = {
    "1–3.3 kHz": "#0072B2",
    "1–10 kHz": "#009E73",
    "1–33 kHz": "#CC79A7",
    "1–100 kHz": "#D55E00",
    "100 Hz–100 kHz": "#E69F00",
    "30 Hz–100 kHz": "#56B4E9",
}

marcadores = {
    "1–3.3 kHz": "o",
    "1–10 kHz": "s",
    "1–33 kHz": "^",
    "1–100 kHz": "D",
    "100 Hz–100 kHz": "P",
    "30 Hz–100 kHz": "X",
}

# Gráfica 

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

for banda in orden:

    color = colores[banda]
    marcador = marcadores[banda]

    R = procesados[banda]["R"]
    V = procesados[banda]["V"]
    uV = procesados[banda]["u_V"]
    uR = procesados[banda]["u_R"]

    mask = R <= 100000.0

    R_fit = R[mask]
    V_fit = V[mask]
    uV_fit = uV[mask]
    uR_fit = uR[mask]

    m = ajustes[banda]["m"]
    b = ajustes[banda]["b"]

    residuos = ajustes[banda]["residuos"]
    C_eff = ajustes[banda]["C_eff"]

    # Datos experimentales usados en el ajuste

    ax1.errorbar(
        R_fit,
        V_fit,
        xerr=uR_fit,
        yerr=uV_fit,
        fmt=marcador,
        color=color,
        ecolor=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=3.5,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        label=banda,
        zorder=5,
    )

    # Punto de 1 MOhm (no incluido en el ajuste)

    ax1.errorbar(
        R[-1],
        V[-1],
        xerr=uR[-1],
        yerr=uV[-1],
        fmt=marcador,
        color=color,
        ecolor=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=5,
        markeredgewidth=1.1,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=5,
    )

    # Rango de las curvas: desde 100 Ohm hasta 1 MOhm

    R_linea = np.logspace(
        np.log10(100),
        np.log10(1000000),
        700,
    )

    # Predicción teórica
    # Vcorr = V(R) - V(10 Ohm)
    delta_f = ENBW[banda]

    m_teorica = (
        4
        * kB_ref
        * T
        * delta_f
        * (G1 * G2) ** 2
        / K_squarer
    )

    V_teorica = (
        m_teorica
        * (R_linea - 10.0)
    )

    ax1.plot(
        R_linea,
        V_teorica,
        color=color,
        linestyle=":",
        linewidth=1.2,
    )

    # Ajuste experimental

    V_ajuste = (
        m * R_linea + b
    )

    ax1.plot(
        R_linea,
        V_ajuste,
        color=color,
        linestyle="-",
        linewidth=1.0,
    )

    # Residuales normalizados

    u_resid = np.sqrt(
        np.diag(C_eff)
    )

    residuos_norm = (
        residuos / u_resid
    )

    ax2.scatter(
        R_fit,
        residuos_norm,
        color=color,
        marker=marcador,
        s=18,
        zorder=5,
    )



ax1.set_xscale("log")

ax1.set_yscale(
    "symlog",
    linthresh=1e-4,
    linscale=0.2,
)

ax1.set_xlim(
    70,
    1.35e6,
)

ax1.set_ylabel(
    r"$V_{\mathrm{corr}}$ (V)"
)

ax1.set_ylim(
    -1e-4,
    2e1,
)

ax1.grid(
    color="gray",
    alpha=0.14,
    linewidth=0.40,
    which="both",
)


# FORMATO RESIDUALES NORMALIZADOS

ax2.axhline(
    0,
    color="black",
    linewidth=0.8,
)

ax2.axhline(
    1,
    color="gray",
    linestyle="--",
    linewidth=0.7,
)

ax2.axhline(
    -1,
    color="gray",
    linestyle="--",
    linewidth=0.7,
)

ax2.set_xscale("log")
ax2.set_yscale(
    "symlog",
    linthresh=1
)

residual_max = max(
    np.max(
        np.abs(
            ajustes[banda]["residuos"]
            / np.sqrt(np.diag(ajustes[banda]["C_eff"]))
        )
    )
    for banda in orden
)

ax2.set_ylim(
    -1.50 * residual_max,
    1.50 * residual_max,
)

ax2.set_xlabel(
    r"Resistencia $R$ ($\Omega$)"
)

ax2.set_ylabel(
    "Residual normalizado"
)

ax2.grid(
    color="gray",
    alpha=0.14,
    linewidth=0.40,
    which="both",
)

# LEYENDA 

leyenda_modelos = [
    Line2D(
        [0],
        [0],
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Teórico",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        linewidth=1.0,
        label="Ajuste",
    ),
]

legend_modelos = ax1.legend(
    handles=leyenda_modelos,
    fontsize=9,
    loc="lower right",
    frameon=True,
)

ax1.add_artist(legend_modelos)

# Segunda leyenda: bandas
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(
    handles,
    labels,
    fontsize=9,
    title=r"Ancho de banda $\Delta f$",
    title_fontsize=10,
    ncol=2,
    loc="upper left",
    frameon=True,
)

# GUARDAR

plt.savefig(
    CARPETA_ENTREGA / "figura_ruido_resistencia.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    CARPETA_ENTREGA / "figura_ruido_resistencia.pdf",
    bbox_inches="tight",
)

plt.show()

# EXPORTAR TABLAS

dataset_completo.to_csv(
    CARPETA_DATOS / "01_dataset_completo.csv",
    index=False
)

tabla_resumen.to_csv(
    CARPETA_DATOS / "02_promedios_incertidumbres.csv",
    index=False
)

tabla_corregidos.to_csv(
    CARPETA_DATOS / "03_datos_corregidos.csv",
    index=False
)

tabla_final.to_csv(
    CARPETA_DATOS / "04_tabla_final_kB.csv",
    index=False
)

tabla_final_formateada.to_csv(
    CARPETA_DATOS / "05_tabla_final_formateada.csv",
    index=False
)

tabla_1M.to_csv(
    CARPETA_DATOS / "06_analisis_1Mohm.csv",
    index=False
)
