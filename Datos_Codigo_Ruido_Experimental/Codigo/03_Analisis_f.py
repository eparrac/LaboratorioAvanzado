#libreria
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from matplotlib.lines import Line2D
#Guardar Graficas
CARPETA_CODIGO = Path(__file__).resolve().parent
CARPETA_ENTREGA = CARPETA_CODIGO.parent
# Datos
R = np.array([10, 10000.0, 100000.0, 1000000.0], dtype=float)
P_1_10 = np.array([9.1788, 9.242013262, 9.414034364, 9.909568465])
P_1_33 = np.array([10.236136206, 10.334328751, 12.247802222, 13.657904728])
P_1_100 = np.array([11.975922672, 12.145819379, 13.506692297, 14.270036564])
uP_1_10 = np.array([0.352965204, 1.478952383, 1.197980781, 1.199969511])
uP_1_33 = np.array([0.789106224, 0.330338821, 1.622493224, 0.553859952])
uP_1_100 = np.array([0.429642298, 1.053434362, 0.856019043, 0.318559367])
datos = {
    "1–10 kHz": (P_1_10, uP_1_10),
    "1–33 kHz": (P_1_33, uP_1_33),
    "1–100 kHz": (P_1_100, uP_1_100),
}
T = 300.0
G1 = 600.0
G2 = 300.0
ENBW = {"1–10 kHz": 9997.0, "1–33 kHz": 35543.0, "1–100 kHz": 109961.0}
kB_ref = 1.380649e-23
mask_fit = R <= 100000.0
R_fit = R[mask_fit]

# modelo de regresion
def modelo(R, m, b):
    return m * R + b

# parametros de visualizacion
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
#plotear
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

colores = {
    "1–10 kHz": "#0072B2",
    "1–33 kHz": "#009E73",
    "1–100 kHz": "#CC79A7",
}

marcadores = {
    "1–10 kHz": "o",
    "1–33 kHz": "s",
    "1–100 kHz": "^",
}

residual_max = 0.0

for banda, (P, uP) in datos.items():
    P_corr = P - P[0]
    uP_corr = np.sqrt(uP**2 + uP[0] ** 2)
    P_corr[0] = 0.0
    uP_corr[0] = uP[0]
    y = P_corr[mask_fit]
    uy = uP_corr[mask_fit]
    popt, pcov = curve_fit(modelo, R_fit, y, sigma=uy, absolute_sigma=True)
    m, b = popt
    um, ub = np.sqrt(np.diag(pcov))
    y_pred = modelo(R_fit, m, b)
    residuos = y - y_pred
    residuos_norm = residuos / uy
    R2 = 1 - np.sum(residuos**2) / np.sum((y - np.mean(y)) ** 2)
    chi2 = np.sum((residuos / uy) ** 2)
    dof = len(y) - 2
    chi2_red = chi2 / dof
    kB = m / (4 * T * ENBW[banda] * (G1 * G2) ** 2)
    ukB = abs(kB) * (um / abs(m))
    z = abs(kB - kB_ref) / ukB
    color = colores[banda]
    marcador = marcadores[banda]

    R_line = np.logspace(
        np.log10(10),
        np.log10(1000000.0),
        700,
    )

    # Datos usados en el ajuste
    ax1.errorbar(
        R_fit,
        y,
        yerr=uy,
        fmt=marcador,
        color=color,
        ecolor=color,
        markerfacecolor=color,
        markeredgecolor=color,
        markersize=4,
        capsize=3,
        elinewidth=1.0,
        capthick=1.0,
        label=banda,
        zorder=5,
    )

    # Punto de 1 MOhm mostrado, pero excluido del ajuste
    ax1.errorbar(
        R[-1],
        P_corr[-1],
        yerr=uP_corr[-1],
        fmt=marcador,
        markerfacecolor=color,
        markeredgecolor=color,
        ecolor=color,
        markersize=5,
        capsize=3,
        elinewidth=1.0,
        capthick=1.0,
        zorder=5,
    )

    # Predicción teórica de Johnson--Nyquist
    m_teorica = (
        4
        * kB_ref
        * T
        * ENBW[banda]
        * (G1 * G2) ** 2
    )
    P_teorica = m_teorica * (R_line - 10.0)

    ax1.plot(
        R_line,
        P_teorica,
        color=color,
        linestyle=":",
        linewidth=1.2,
    )

    # Ajuste experimental
    ax1.plot(
        R_line,
        modelo(R_line, m, b),
        color=color,
        linestyle="-",
        linewidth=1.0,
    )

    # Residuales normalizados
    ax2.scatter(
        R_fit,
        residuos_norm,
        marker=marcador,
        color=color,
        s=22,
        zorder=5,
    )

    residual_max = max(
        residual_max,
        float(np.max(np.abs(residuos_norm))),
    )

    print("\n" + "=" * 50)
    print(banda)
    print("=" * 50)
    print(f"m = {m:.6e} ± {um:.2e} V²/Ω")
    print(f"b = {b:.6e} ± {ub:.2e} V²")
    print(f"R² = {R2:.4f}")
    print(f"χ²ν = {chi2_red:.4f}")
    print(f"kB = {kB:.6e} ± {ukB:.2e} J/K")
    print(f"z = {z:.2f}")
ax1.set_xscale("log")
ax1.set_yscale(
    "symlog",
    linthresh=1.0,
    linscale=0.2,
)
ax1.set_xlim(7, 1.35e6)
ax1.set_ylabel(r"Potencia corregida $P'$ (V$^2$)")
ax1.grid(
    color="gray",
    alpha=0.14,
    linewidth=0.40,
    which="both",
)

margen_residual = 1.25 * residual_max
if margen_residual == 0.0:
    margen_residual = 1.0

ax2.axhline(
    0,
    color="black",
    linewidth=0.8,
)

ax2.set_ylim(
    -margen_residual,
    margen_residual,
)
ax2.yaxis.set_major_locator(
    MaxNLocator(nbins=6)
)
ax2.set_ylabel("Residual normalizado")
ax2.set_xlabel(r"Resistencia $R$ ($\Omega$)")
ax2.grid(
    color="gray",
    alpha=0.14,
    linewidth=0.40,
    which="both",
)

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

ax2.set_xticks([10, 10000.0, 100000.0, 1000000.0])
ax2.set_xticklabels(
    [
        "$10\\,\\Omega$",
        "$10\\,\\mathrm{k}\\Omega$",
        "$100\\,\\mathrm{k}\\Omega$",
        "$1\\,\\mathrm{M}\\Omega$",
    ]
)
plt.savefig(
    CARPETA_ENTREGA / "figura3_ajustes_f.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
)
plt.savefig(
    CARPETA_ENTREGA / "figura3_ajustes_f.pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)
plt.show()
