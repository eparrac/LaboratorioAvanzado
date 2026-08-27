from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

CARPETA = Path(__file__).resolve().parent
ARCHIVO = next(CARPETA.glob("*CH1.CSV"))
NPERSEG = 1024
NOVERLAP = 512
tabla = pd.read_csv(ARCHIVO, header=None)
tiempo = pd.to_numeric(tabla.iloc[:, 3], errors="coerce")
voltaje = pd.to_numeric(tabla.iloc[:, 4], errors="coerce")
validos = tiempo.notna() & voltaje.notna()
tiempo = tiempo[validos].to_numpy(float)
voltaje = voltaje[validos].to_numpy(float)
voltaje = voltaje - np.mean(voltaje)
dt = float(np.median(np.diff(tiempo)))
fs = 1.0 / dt
frecuencia_fft = np.fft.rfftfreq(len(voltaje), d=dt)
amplitud_fft = 2.0 * np.abs(np.fft.rfft(voltaje)) / len(voltaje)
frecuencia, psd = welch(
    voltaje,
    fs=fs,
    window="hann",
    nperseg=NPERSEG,
    noverlap=NOVERLAP,
    detrend=False,
    scaling="density",
    return_onesided=True,
)


def integrar(limite_hz):
    limite = min(float(limite_hz), float(frecuencia[-1]))
    dentro = (frecuencia > 0.0) & (frecuencia < limite)
    f = np.r_[0.0, frecuencia[dentro], limite]
    s = np.r_[
        np.interp(0.0, frecuencia, psd), psd[dentro], np.interp(limite, frecuencia, psd)
    ]
    return float(np.trapezoid(s, f))


p_0_40 = integrar(40000.0)
p_0_50 = integrar(50000.0)
pd.DataFrame(
    {
        "archivo": [ARCHIVO.name],
        "fs_Hz": [fs],
        "n_muestras": [len(voltaje)],
        "nperseg": [NPERSEG],
        "noverlap": [NOVERLAP],
        "ventana": ["hann"],
        "P_0_40kHz_V2": [p_0_40],
        "P_0_50kHz_V2": [p_0_50],
    }
).to_csv(CARPETA / "integrales_P.csv", index=False)
pd.DataFrame({"frecuencia_Hz": frecuencia_fft, "amplitud_FFT_V": amplitud_fft}).to_csv(
    CARPETA / "datos_FFT.csv", index=False
)
pd.DataFrame({"frecuencia_Hz": frecuencia, "PSD_V2_Hz": psd}).to_csv(
    CARPETA / "datos_PSD.csv", index=False
)
fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.semilogy(frecuencia / 1000.0, psd, color="#4c78a8", linewidth=1.1)
ax.set_xlim(0, 50)
ax.set_xlabel("Frecuencia (kHz)")
ax.set_ylabel("$S_V(f)$ [V$^2$/Hz]")
ax.grid(color="#d9d9d9", linewidth=0.7, alpha=0.8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(CARPETA / "PSD.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"fs = {fs:.6f} Hz")
print(f"P 0-40 kHz = {p_0_40:.9f} V^2")
print(f"P 0-50 kHz = {p_0_50:.9f} V^2")
