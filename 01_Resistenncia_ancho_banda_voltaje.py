import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        u_total = np.sqrt(u_media**2 + ui**2)
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
    R_array = np.array([100, 1000.0, 10000.0, 100000.0, 1000000.0], dtype=float)
    Vcorr = np.array([medias[R] - Vref for R in R_array])
    n = len(R_array)
    C = np.full((n, n), uref**2)
    for i, R in enumerate(R_array):
        C[i, i] += incertidumbres[R] ** 2
    uVcorr = np.sqrt(np.diag(C))
    uR = u_rel_R * R_array
    procesados[banda] = {"R": R_array, "u_R": uR, "V": Vcorr, "u_V": uVcorr, "C": C}
tabla_resumen = pd.DataFrame(filas_resumen)
filas_corregidos = []
for banda in orden:
    R = procesados[banda]["R"]
    V = procesados[banda]["V"]
    uV = procesados[banda]["u_V"]
    uR = procesados[banda]["u_R"]
    for Ri, uiR, Vi, uiV in zip(R, uR, V, uV):
        filas_corregidos.append(
            {
                "Banda": banda,
                "R (ohm)": Ri,
                "u(R) (ohm)": uiR,
                "Vcorr (V)": Vi,
                "u(Vcorr) (V)": uiV,
                "Usado en ajuste": "Si" if Ri <= 100000.0 else "No",
            }
        )
tabla_corregidos = pd.DataFrame(filas_corregidos)


def ajuste_gls(R, V, C_y, u_R, max_iter=100):
    X = np.column_stack([R, np.ones_like(R)])
    beta = np.linalg.lstsq(X, V, rcond=None)[0]
    m = beta[0]
    for _ in range(max_iter):
        C_eff = C_y + np.diag((m * u_R) ** 2)
        C_inv = np.linalg.inv(C_eff)
        cov_beta = np.linalg.inv(X.T @ C_inv @ X)
        beta_nuevo = cov_beta @ X.T @ C_inv @ V
        m_nuevo = beta_nuevo[0]
        if np.isclose(m_nuevo, m, rtol=1e-12, atol=1e-20):
            beta = beta_nuevo
            break
        beta = beta_nuevo
        m = m_nuevo
    m = beta[0]
    C_eff = C_y + np.diag((m * u_R) ** 2)
    C_inv = np.linalg.inv(C_eff)
    cov_beta = np.linalg.inv(X.T @ C_inv @ X)
    beta = cov_beta @ X.T @ C_inv @ V
    return (beta, cov_beta, C_eff)


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
    C_fit = C[np.ix_(mask, mask)]
    beta, cov_beta, C_eff = ajuste_gls(R_fit, V_fit, C_fit, uR_fit)
    m, b = beta
    um, ub = np.sqrt(np.diag(cov_beta))
    V_pred = m * R_fit + b
    residuos = V_fit - V_pred
    SS_res = np.sum(residuos**2)
    SS_tot = np.sum((V_fit - np.mean(V_fit)) ** 2)
    R2 = 1 - SS_res / SS_tot
    C_inv = np.linalg.inv(C_eff)
    chi2 = residuos.T @ C_inv @ residuos
    gl = len(R_fit) - 2
    chi2_red = chi2 / gl
    delta_f = ENBW[banda]
    u_delta_f = u_rel_ENBW * delta_f
    kB = K_squarer * m / (4 * T * delta_f * (G1 * G2) ** 2)
    u_rel_kB = np.sqrt(
        (um / m) ** 2
        + (u_T / T) ** 2
        + u_rel_ENBW**2
        + (2 * u_rel_G1) ** 2
        + (2 * u_rel_G2) ** 2
        + u_rel_squarer**2
        + u_rel_average**2
    )
    u_kB = abs(kB) * u_rel_kB
    z = abs(kB - kB_ref) / u_kB
    ajustes[banda] = {
        "m": m,
        "um": um,
        "b": b,
        "ub": ub,
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
    V_pred_1M = m * 1000000.0 + b
    razon_1M = V_1M / V_pred_1M
    filas_1M.append(
        {
            "Configuracion": banda,
            "V_1M_medido (V)": V_1M,
            "V_1M_predicho (V)": V_pred_1M,
            "Medido/Predicho": razon_1M,
        }
    )
tabla_final = pd.DataFrame(filas_tabla_final)
tabla_1M = pd.DataFrame(filas_1M)
filas_formateadas = []
for _, fila in tabla_final.iterrows():
    filas_formateadas.append(
        {
            "Configuracion": fila["Configuracion"],
            "ENBW": f"{fila['ENBW (Hz)']:.0f} ± {fila['u(ENBW) (Hz)']:.0f} Hz",
            "Pendiente": f"({fila['m (V/ohm)']:.4e} ± {fila['u(m) (V/ohm)']:.2e}) V/ohm",
            "Intercepto": f"({fila['b (V)']:.4e} ± {fila['u(b) (V)']:.2e}) V",
            "kB": f"({fila['kB (J/K)']:.4e} ± {fila['u(kB) (J/K)']:.2e}) J/K",
            "R2": f"{fila['R2']:.6f}",
            "chi2_red": f"{fila['chi2_reducido']:.3f}",
            "z": f"{fila['z']:.3f}",
        }
    )
tabla_final_formateada = pd.DataFrame(filas_formateadas)
colores = {
    "1–3.3 kHz": "#0072B2",
    "1–10 kHz": "#009E73",
    "1–33 kHz": "#CC79A7",
    "1–100 kHz": "#D55E00",
    "100 Hz–100 kHz": "#E69F00",
    "30 Hz–100 kHz": "#56B4E9",
}
fig, (ax1, ax2, ax3) = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [3.6, 1.2, 1.2], "hspace": 0.07},
)
ax_in = ax1.inset_axes([0.1, 0.27, 0.58, 0.53])
for banda in orden:
    color = colores[banda]
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
    ax1.errorbar(
        R_fit,
        V_fit,
        xerr=uR_fit,
        yerr=uV_fit,
        fmt="o",
        color=color,
        ecolor=color,
        markersize=2.3,
        elinewidth=1.25,
        capsize=4,
        capthick=1.25,
        label=banda,
        zorder=5,
    )
    ax1.errorbar(
        R[-1],
        V[-1],
        xerr=uR[-1],
        yerr=uV[-1],
        fmt="o",
        color=color,
        ecolor=color,
        markerfacecolor="none",
        markersize=4.3,
        markeredgewidth=1.0,
        elinewidth=1.25,
        capsize=4,
        capthick=1.25,
        zorder=5,
    )
    R_linea = np.linspace(100, 100000.0, 600)
    ax1.plot(R_linea, m * R_linea + b, color=color, linewidth=0.7)
    R_extra = np.linspace(100000.0, 1000000.0, 800)
    ax1.plot(R_extra, m * R_extra + b, color=color, linestyle="--", linewidth=0.7)
    ax_in.errorbar(
        R_fit,
        V_fit,
        xerr=uR_fit,
        yerr=uV_fit,
        fmt="o",
        color=color,
        ecolor=color,
        markersize=2.2,
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
    )
    ax_in.plot(R_linea, m * R_linea + b, color=color, linewidth=0.65)
    residuos_mV = residuos * 1000
    u_resid = np.sqrt(np.diag(C_eff))
    u_resid_mV = u_resid * 1000
    ax2.errorbar(
        R_fit,
        residuos_mV,
        yerr=u_resid_mV,
        fmt="o",
        color=color,
        ecolor=color,
        markersize=2.5,
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=5,
    )
    residuos_norm = residuos / u_resid
    ax3.scatter(R_fit, residuos_norm, color=color, s=11, zorder=5)
ax1.set_xscale("log")
ax1.set_ylabel("$V_{\\mathrm{corr}}$ (V)")
ax1.legend(fontsize=7.2, ncol=2, loc="upper left", frameon=True)
ax1.grid(alpha=0.18, linewidth=0.45)
ax_in.set_xlim(-3000, 105000.0)
ax_in.set_ylim(-0.02, 0.63)
ax_in.set_xlabel("$R$ ($\\Omega$)", fontsize=8)
ax_in.set_ylabel("$V_{\\mathrm{corr}}$ (V)", fontsize=8)
ax_in.set_title("Región del ajuste", fontsize=9)
ax_in.tick_params(labelsize=7)
ax_in.grid(alpha=0.18, linewidth=0.4)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xscale("log")
ax2.set_ylabel("Residual (mV)")
ax2.grid(alpha=0.18, linewidth=0.45)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.axhline(1, color="gray", linestyle="--", linewidth=0.6)
ax3.axhline(-1, color="gray", linestyle="--", linewidth=0.6)
ax3.set_xscale("log")
ax3.set_xlabel("Resistencia $R$ ($\\Omega$)")
ax3.set_ylabel("Residual normalizado")
ax3.set_ylim(-42, 62)
ax3.set_yticks(np.arange(-40, 61, 20))
ax3.grid(alpha=0.18, linewidth=0.45)
plt.savefig("figura_ruido_resistencia.png", dpi=300, bbox_inches="tight")
plt.savefig("figura_ruido_resistencia.pdf", bbox_inches="tight")
plt.show()
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
print("\n\n================ DATASET COMPLETO ================\n")
print(dataset_completo.to_string(index=False))
print("\n\n========== PROMEDIOS E INCERTIDUMBRES ==========\n")
print(tabla_resumen.to_string(index=False))
print("\n\n============== DATOS CORREGIDOS ================\n")
print(tabla_corregidos.to_string(index=False))
print("\n\n============== TABLA FINAL ======================\n")
print(tabla_final_formateada.to_string(index=False))
print("\nValor de referencia:")
print(f"kB = {kB_ref:.6e} J/K")
print("\n\n========== COMPORTAMIENTO DE 1 MOHM ============\n")
print(tabla_1M.to_string(index=False))
dataset_completo.to_csv("01_dataset_completo.csv", index=False)
tabla_resumen.to_csv("02_promedios_incertidumbres.csv", index=False)
tabla_corregidos.to_csv("03_datos_corregidos.csv", index=False)
tabla_final.to_csv("04_tabla_final_kB.csv", index=False)
tabla_final_formateada.to_csv("05_tabla_final_formateada.csv", index=False)
tabla_1M.to_csv("06_analisis_1Mohm.csv", index=False)
