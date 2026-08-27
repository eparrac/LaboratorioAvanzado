import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


def modelo(R, m, b):
    return m * R + b


fig, (ax1, ax2, ax3) = plt.subplots(
    3,
    1,
    figsize=(7.5, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3.2, 1.2, 1.2], "hspace": 0.08},
)
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
    R_line = np.logspace(np.log10(10), np.log10(100000.0), 400)
    R_extra = np.logspace(5, 6, 200)
    error = ax1.errorbar(
        R_fit,
        y,
        yerr=uy,
        fmt="o",
        markersize=5,
        capsize=3,
        elinewidth=0.8,
        label=f"{banda}, $R^2={R2:.3f}$",
    )
    color = error[0].get_color()
    ax1.plot(R_line, modelo(R_line, m, b), "-", linewidth=1.2, color=color)
    ax1.plot(R_extra, modelo(R_extra, m, b), "--", linewidth=1.0, color=color)
    ax1.errorbar(
        R[-1],
        P_corr[-1],
        yerr=uP_corr[-1],
        fmt="o",
        markerfacecolor="none",
        markeredgecolor=color,
        ecolor=color,
        markersize=6,
        capsize=3,
    )
    ax2.errorbar(
        R_fit,
        residuos,
        yerr=uy,
        fmt="o",
        markersize=5,
        capsize=3,
        elinewidth=0.8,
        color=color,
    )
    ax3.plot(R_fit, residuos_norm, "o", markersize=5, linewidth=0.8, color=color)
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
ax1.set_ylabel("Potencia corregida $P'$ (V$^2$)")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.25, linewidth=0.5)
ax2.axhline(0, linestyle="--", linewidth=0.8)
ax2.set_ylabel("Residual (V$^2$)")
ax2.grid(alpha=0.25, linewidth=0.5)
ax3.axhline(0, linewidth=0.8)
ax3.axhline(1, linestyle="--", linewidth=0.7)
ax3.axhline(-1, linestyle="--", linewidth=0.7)
ax3.set_ylabel("Residual\nnormalizado")
ax3.set_xlabel("Resistencia $R$ ($\\Omega$)")
ax3.grid(alpha=0.25, linewidth=0.5)
ax3.set_xticks([10, 10000.0, 100000.0, 1000000.0])
ax3.set_xticklabels(
    [
        "$10\\,\\Omega$",
        "$10\\,\\mathrm{k}\\Omega$",
        "$100\\,\\mathrm{k}\\Omega$",
        "$1\\,\\mathrm{M}\\Omega$",
    ]
)
plt.show()
