#libreria
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Carpeta
CARPETA_CODIGO = Path(__file__).resolve().parent
CARPETA_DATOS = CARPETA_CODIGO.parent / "Datos"
CARPETA_ENTREGA = CARPETA_CODIGO.parent

# Datos
series = {
    "1–10 kHz": CARPETA_DATOS / "PSD_10k_1-10kHz.csv",
    "1–33 kHz": CARPETA_DATOS / "PSD_10k_1-33kHz.csv",
    "1–100 kHz": CARPETA_DATOS / "PSD_10k_1-100kHz.csv",
}

#colores

colores = {
    "1–10 kHz":"#0072B2",
    "1–33 kHz":"#D55E00",
    "1–100 kHz":"#009E73",
}

#para que se parezca a Latex

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "Latin Modern Roman",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
    }
)

#Plotear
fig, ax = plt.subplots(figsize=(9.0, 5.8))

for etiqueta, archivo in series.items():
    tabla = pd.read_csv(archivo)

    frecuencia_kHz = tabla["frecuencia_Hz"] / 1000.0
    psd = tabla["PSD_V2_Hz"]

    mask = (
        (frecuencia_kHz >= 0.0)
        & (frecuencia_kHz <= 50.0)
        & (psd > 0.0)
    )

    ax.plot(
        frecuencia_kHz[mask],
        psd[mask],
        color=colores[etiqueta],
        linewidth=1.25,
        label=etiqueta,
    )

ax.set_yscale("log")
ax.set_xlim(0.0, 50.0)

ax.set_xlabel(r"Frecuencia $f$ (kHz)")
ax.set_ylabel(r"$S_V(f)$ (V$^2$/Hz)")

ax.legend(
    loc="upper right",
    title=r"Ancho de banda $\Delta f$",
    title_fontsize=11,
    frameon=True,
    borderpad=0.45,
    handlelength=2.2,
)

ax.grid(
    color="gray",
    alpha=0.24,
    linewidth=0.50,
    which="both",
)

ax.tick_params(
    which="major",
    direction="out",
    length=4,
    width=0.8,
)
ax.tick_params(
    which="minor",
    direction="out",
    length=2.5,
    width=0.6,
)

fig.tight_layout(pad=0.6)

plt.savefig(
    CARPETA_ENTREGA / "PSD_10k_comparacion.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
)
plt.savefig(
    CARPETA_ENTREGA / "PSD_10k_comparacion.pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.show()
