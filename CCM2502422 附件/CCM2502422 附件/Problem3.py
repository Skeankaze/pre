# =========================
# 一、导入与字体设置
# =========================
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager as fm


try:
    fm.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    rcParams["font.sans-serif"] = ["Microsoft YaHei"]
except Exception:
    pass
rcParams["axes.unicode_minus"] = False

from scipy.optimize import nnls
from scipy.signal import savgol_filter

# =========================
# 二、参数区
# =========================
# 太阳光谱与 LED 光谱的 Excel 路径
SUN_FILE = r"C:\Users\23118\PycharmProjects\pythonProject\huashu\Problem 3 SUN_SPD.xlsx"
LED_FILE = r"C:\Users\23118\PycharmProjects\pythonProject\huashu\Problem 2_LED_SPD.xlsx"


METRICS_CSV = Path("problem4_results") / "metrics_long.csv"

# 列名/通道等配置
WAVE_COL = "wavelength"
LED_CHANNELS = ["B", "G", "R", "WW", "CW"]
REP_TIMES = [("早晨", "时间7:30"),
             ("正午", "时间12:30"),
             ("傍晚", "时间17:30")]


SG_WINDOW, SG_POLY = 7, 2

NORMALIZE_VIZ = True


LED_MANUAL_MAP = {
    "wavelength": "波长(mW/m2/nm)",
    "B": "Blue",
    "G": "Green",
    "R": "Red",
    "WW": "Warm White",
    "CW": "Cold White",
}

# 输出文件名
FIG_COMPARE = "compare_3times.png"
FIG_TIMELINE = "weights_timeline.png"
CSV_STRATEGY = "control_strategy.csv"

# =========================
# 三、色度学与数据工具函数
# =========================
# 内置 CIE 1931 2° CMF（5 nm 采样，380–780 nm），用于快速 XYZ 积分
wl_5nm = np.arange(380, 781, 5, dtype=float)
xbar_5nm = np.array([
    0.0014,0.0022,0.0042,0.0076,0.0143,0.0232,0.0435,0.0776,0.1344,0.2148,
    0.2839,0.3285,0.3483,0.3481,0.3362,0.3187,0.2908,0.2511,0.1954,0.1421,
    0.0956,0.05795,0.03201,0.0147,0.0049,0.0024,0.0093,0.0291,0.06327,0.1096,
    0.1655,0.2257,0.2904,0.3597,0.43345,0.51205,0.5945,0.6784,0.7621,0.8425,
    0.9163,0.9786,1.0263,1.0567,1.0622,1.0456,1.0026,0.9384,0.85445,0.7514,
    0.6424,0.5419,0.4479,0.3608,0.2835,0.2187,0.1649,0.1212,0.0874,0.0636,
    0.04677,0.0329,0.0227,0.01584,0.01136,0.00811,0.00579,0.00411,0.0029,0.00205,
    0.00144,0.001,0.00069,0.00048,0.00034,0.00024,0.00017,0.00012,0.000085,0.00006,0.000042
])
ybar_5nm = np.array([
    0.0000,0.0001,0.00012,0.00013,0.00014,0.00016,0.0002,0.0003,0.0004,0.00064,
    0.0012,0.00218,0.004,0.0073,0.0116,0.01684,0.023,0.0298,0.038,0.048,
    0.06,0.0739,0.09098,0.1126,0.13902,0.1693,0.20802,0.2586,0.323,0.4073,
    0.503,0.6082,0.71,0.7932,0.862,0.91485,0.954,0.9803,0.99495,1.0,
    0.995,0.9786,0.952,0.9154,0.87,0.8163,0.757,0.6949,0.631,0.5668,
    0.503,0.4412,0.381,0.321,0.265,0.217,0.175,0.1382,0.107,0.0816,
    0.061,0.04458,0.032,0.0232,0.017,0.01192,0.00821,0.00572,0.0041,0.00293,
    0.00209,0.00148,0.00105,0.00074,0.00052,0.00036,0.00025,0.00017,0.00012,0.000085,0.00006
])
zbar_5nm = np.array([
    0.0065,0.0105,0.0201,0.0362,0.0679,0.1102,0.2074,0.3713,0.6456,1.03905,
    1.3856,1.62296,1.74706,1.7826,1.77211,1.7441,1.6692,1.5281,1.28764,1.0419,
    0.81295,0.6162,0.46518,0.3533,0.272,0.2123,0.1582,0.1117,0.07825,0.05725,
    0.04216,0.02984,0.0203,0.0134,0.00875,0.00575,0.0039,0.00275,0.0021,0.0018,
    0.00165,0.0014,0.0011,0.001,0.0008,0.0006,0.00034,0.00024,0.00019,0.0001,
    0.00005,0.00003,0.00002,0.00001,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
])

def _interp_mat(dst_wl: np.ndarray, src_wl: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(dst_wl, src_wl, mat[:, j], left=0.0, right=0.0)
                      for j in range(mat.shape[1])]).T

def _area_norm(v: np.ndarray, wl: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        a = max(np.trapz(v, wl), 1e-12)
        return v / a
    out = v.copy()
    for j in range(v.shape[1]):
        a = max(np.trapz(v[:, j], wl), 1e-12)
        out[:, j] = v[:, j] / a
    return out

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _ensure_file(path: str, label: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到{label}文件：{path}")
    if p.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"{label}应为 Excel 文件（.xlsx/.xls），当前：{p.suffix}")

def spd_to_xyz(wl: np.ndarray, spd: np.ndarray):
    cmf = np.vstack([
        np.interp(wl, wl_5nm, xbar_5nm, left=0.0, right=0.0),
        np.interp(wl, wl_5nm, ybar_5nm, left=0.0, right=0.0),
        np.interp(wl, wl_5nm, zbar_5nm, left=0.0, right=0.0),
    ])
    X = float(np.trapz(spd * cmf[0], wl))
    Y = float(np.trapz(spd * cmf[1], wl))
    Z = float(np.trapz(spd * cmf[2], wl))
    return X, Y, Z

def xy_from_xyz(X: float, Y: float, Z: float):
    denom = max(X + Y + Z, 1e-12)
    return X / denom, Y / denom

def mccamy_cct_from_xy(x: float, y: float) -> float:
    xe, ye = 0.3320, 0.1858
    n = (x - xe) / (y - ye + 1e-12)
    CCT = (-449.0 * n**3) + (3525.0 * n**2) - (6823.3 * n) + 5520.33
    return float(max(CCT, 0.0))

# =========================
# 四、数据读取（太阳 & LED）
# =========================
def load_sun(path: str, wave_col: str):
    """读取太阳光谱：第一列为波长，其余列为时刻（如“时间7:30”）。"""
    _ensure_file(path, "太阳光谱")
    df = pd.read_excel(path, engine="openpyxl")
    df = df.rename(columns={df.columns[0]: wave_col})
    wl = df[wave_col].to_numpy(float)
    time_cols = [c for c in df.columns if c != wave_col]
    if not time_cols:
        raise ValueError("太阳光谱表中未找到时间列，请确认第一列为波长，其余列为时刻。")
    return wl, df.set_index(wave_col)[time_cols]

def load_led(path: str, wave_col: str, ch_cols: List[str]):
    _ensure_file(path, "LED光谱")
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    required_keys = ["wavelength"] + ch_cols
    missing_keys = [k for k in required_keys if k not in LED_MANUAL_MAP]
    if missing_keys:
        raise ValueError(f"LED_MANUAL_MAP 缺少键：{missing_keys}")

    missing_cols = [LED_MANUAL_MAP[k] for k in required_keys if LED_MANUAL_MAP[k] not in df.columns]
    if missing_cols:
        raise ValueError(f"LED 文件找不到手动映射所指列：{missing_cols}\n现有列：{list(df.columns)}")

    wl = df[LED_MANUAL_MAP["wavelength"]].to_numpy(float)
    led_df = df.set_index(LED_MANUAL_MAP["wavelength"])[
        [LED_MANUAL_MAP[c] for c in ch_cols]
    ].copy()
    led_df.columns = ch_cols
    return wl, led_df

# =========================
# 五、问题三：控制策略主流程
# =========================
def run_problem3_control():
    """求解五通道 NNLS 权重，生成控制策略与主图。"""
    print("[P3] 读取数据…")
    sun_wl, sun_df = load_sun(SUN_FILE, WAVE_COL)
    led_wl, led_df = load_led(LED_FILE, WAVE_COL, LED_CHANNELS)

    wl = sun_wl.copy()
    M = _interp_mat(wl, led_wl, led_df.to_numpy(float))
    S = sun_df.to_numpy(float)
    time_labels = list(sun_df.columns)
    n_times = S.shape[1]
    print(f"[P3] 波长点数={len(wl)}, 时间点数={n_times}, LED通道数={M.shape[1]}")

    W = np.zeros((n_times, M.shape[1]))
    Y = np.zeros_like(S)
    for j in range(n_times):
        w, _ = nnls(M, S[:, j])
        W[j, :] = w
        Y[:, j] = M @ w

    if n_times >= SG_WINDOW and SG_WINDOW % 2 == 1:
        W_s = np.zeros_like(W)
        for k in range(W.shape[1]):
            W_s[:, k] = savgol_filter(W[:, k], SG_WINDOW, SG_POLY, mode="interp")
            W_s[:, k] = np.clip(W_s[:, k], 0, None)
    else:
        W_s = W.copy()
        print("[P3][Warn] 时间点不足或窗口非奇数，未进行 Savitzky–Golay 平滑。")

    # 用平滑权重重合成（仅用于评估展示）
    Y_s = (M @ W_s.T).T
    Y_s = Y_s.T  # 统一成 (n_wl, n_times)

    # —— 导出控制策略（原始值 + 归一化百分比） —— #
    percent = W_s.copy()
    for k in range(percent.shape[1]):
        mx = max(percent[:, k].max(), 1e-12)
        percent[:, k] = percent[:, k] / mx * 100.0
    out_raw = pd.DataFrame(W_s, columns=[f"{c}_raw" for c in LED_CHANNELS], index=time_labels)
    out_pct = pd.DataFrame(percent, columns=[f"{c}_pct" for c in LED_CHANNELS], index=time_labels)
    control = pd.concat([out_raw, out_pct], axis=1)
    control.index.name = "time"
    control.to_csv(CSV_STRATEGY, encoding="utf-8-sig")
    print(f"[P3][Saved] 控制策略表：{CSV_STRATEGY}")

    # —— 权重轨迹图 —— #
    plt.figure(figsize=(12, 4.2))
    for k, c in enumerate(LED_CHANNELS):
        plt.plot(range(n_times), W_s[:, k], label=c, linewidth=2)
    plt.xlabel("时间索引（按文件列顺序）")
    plt.ylabel("权重（相对强度）")
    plt.title("五通道权重随时间的轨迹（平滑后）")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=len(LED_CHANNELS))
    plt.tight_layout()
    plt.savefig(FIG_TIMELINE, dpi=220)
    plt.close()
    print(f"[P3][Saved] 权重时间轨迹图：{FIG_TIMELINE}")

    # —— 三代表时刻光谱对比图 —— #
    def find_idx(label: str) -> int:
        if label in time_labels:
            return time_labels.index(label)
        # 容错：去前导 0
        for i, t in enumerate(time_labels):
            if t.strip().lstrip("0") == label.strip().lstrip("0"):
                return i
        raise ValueError(f"找不到时刻列：{label}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    for ax, (zh, tlabel) in zip(axes, REP_TIMES):
        j = find_idx(tlabel)
        s = S[:, j].copy()
        y = Y_s[:, j].copy()
        s_v, y_v = (_area_norm(s, wl), _area_norm(y, wl)) if NORMALIZE_VIZ else (s, y)

        ax.plot(wl, s_v, label="太阳光谱", linewidth=2)
        ax.plot(wl, y_v, "--", label="LED合成光谱", linewidth=2)
        ax.set_title(f"{zh}（{tlabel}）")
        ax.set_xlabel("波长（nm）")
        ax.set_ylabel("光谱功率（归一）" if NORMALIZE_VIZ else "光谱功率")
        ax.grid(True, alpha=0.3)

        # 计算并展示 CCT 与 RMSE
        Xs, Ys, Zs = spd_to_xyz(wl, s_v); xs, ys = xy_from_xyz(Xs, Ys, Zs)
        Xy, Yy, Zy = spd_to_xyz(wl, y_v); xy, yy = xy_from_xyz(Xy, Yy, Zy)
        cct_s = mccamy_cct_from_xy(xs, ys)
        cct_y = mccamy_cct_from_xy(xy, yy)
        rmse_raw = _rmse(S[:, j], Y_s[:, j])
        rmse_norm = _rmse(s_v, y_v)

        info = (f"CCT(合成): {cct_y:.0f} K\n"
                f"CCT(目标): {cct_s:.0f} K\n"
                f"RMSE(raw):  {rmse_raw:.6f}\n"
                f"RMSE(norm): {rmse_norm:.6f}")
        ax.text(0.03, 0.97, info, transform=ax.transAxes,
                va="top", ha="left", fontsize=8, linespacing=0.9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.8, pad=0.3))
    axes[1].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(FIG_COMPARE, dpi=220)
    plt.close()
    print(f"[P3][Saved] 三联光谱对比图：{FIG_COMPARE}")

# =========================
# 六、箱线图
# =========================
def plot_boxgrid(metrics_csv: Path):
    """
    读取问题四导出的 metrics_long.csv，生成 6 指标分面箱线图，
    并叠加原始观测点（黑色实心），便于论文展示个体差异。
    """
    import seaborn as sns

    if not metrics_csv.exists():
        raise FileNotFoundError(f"未找到 {metrics_csv}，请先运行问题四脚本生成，"
                                f"或把文件放到该路径。来源：problem4_analysis_numpy.py。")

    df = pd.read_csv(metrics_csv)

    # 长表：列包含 subject、condition、六个指标
    metrics = ["SOL", "TST", "SE", "N3pct", "REMpct", "Awakenings"]
    df_long = df.melt(id_vars=["subject", "condition"],
                      value_vars=metrics, var_name="Metric", value_name="Value")

    cond_order = ["A", "B", "C"]
    df_long["condition"] = pd.Categorical(df_long["condition"], categories=cond_order, ordered=True)

    pad = pd.DataFrame([(m, c, "__pad__", np.nan) for m in metrics for c in cond_order],
                       columns=["Metric", "condition", "subject", "Value"])
    pad["condition"] = pd.Categorical(pad["condition"], categories=cond_order, ordered=True)
    df_plot = pd.concat([df_long, pad], ignore_index=True)

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_plot, x="condition", y="Value",
        col="Metric", col_order=metrics,
        kind="box", order=cond_order, col_wrap=3,
        height=4, sharey=False, sharex=False, showfliers=True
    )

    # 叠加真实观测点
    for ax, metric in zip(g.axes.flatten(), metrics):
        plot_data = df_long[df_long["Metric"] == metric]
        sns.stripplot(data=plot_data, x="condition", y="Value",
                      order=cond_order, jitter=True, size=3, color="black", ax=ax)

    for ax in g.axes.flatten():
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels(cond_order)
        ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
        ax.spines['bottom'].set_visible(True)

    g.set_xlabels("Condition"); g.set_ylabels("Value")
    plt.tight_layout()
    plt.savefig("boxgrid_blackpoints_solid.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("[FIG][Saved] 箱线图：boxgrid_blackpoints_solid.png")

# =========================
# 七、均值±标准差柱状图
# =========================
def plot_bar_mean_sd(metrics_csv: Path):
    if not metrics_csv.exists():
        raise FileNotFoundError(f"未找到 {metrics_csv}，无法绘制柱状图。")

    df = pd.read_csv(metrics_csv)
    cond_order = ["A", "B", "C"]

    # 指标 -> (标题, y轴标签, 输出文件名)
    configs = {
        "Awakenings": ("Awakenings - Mean ± SD by Condition", "Awakenings (mean ± SD)", "AWX.png"),
        "N3pct":      ("N3pct - Mean ± SD by Condition",       "N3pct (mean ± SD)",      "N3X.png"),
        "REMpct":     ("REMpct - Mean ± SD by Condition",      "REMpct (mean ± SD)",     "REMX.png"),
        "SE":         ("SE - Mean ± SD by Condition",          "SE (mean ± SD)",         "SEX.png"),
        "SOL":        ("SOL - Mean ± SD by Condition",         "SOL (mean ± SD)",        "SOLX.png"),
        "TST":        ("TST - Mean ± SD by Condition",         "TST (mean ± SD)",        "TSTX.png"),
    }

    for metric, (title, ylabel, fname) in configs.items():
        # 计算均值与标准差
        grp = df.groupby("condition")[metric].agg(['mean', 'std']).reindex(cond_order)
        means = grp['mean'].values
        stds  = grp['std'].values

        # 绘图
        plt.figure(figsize=(12, 7))
        x = np.arange(len(cond_order))
        bars = plt.bar(x, means, yerr=stds, capsize=8, width=0.6,
                       color=["#87CEEB", "#90EE90", "#FF7F7F"], edgecolor="none")
        # 误差棒样式
        _, caps, bars_err = plt.errorbar(x, means, yerr=stds, fmt='none', ecolor='black', elinewidth=2, capsize=8)
        # 轴与标题
        plt.xticks(x, cond_order)
        plt.xlabel("Condition"); plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # 数值标注（均值）
        for xi, m, s in zip(x, means, stds):
            plt.text(xi, m + s + 0.5, f"{m:.2f}", ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        plt.savefig(fname, dpi=260)
        plt.close()
        print(f"[FIG][Saved] 均值±SD柱状图：{fname}")

# =========================
# 八、总入口
# =========================
def main():
    # 1) 问题三：控制策略
    run_problem3_control()

    # 2) 箱线图 & 六张柱状图
    csv_path = Path(METRICS_CSV)
    if not csv_path.exists() and Path("metrics_long.csv").exists():
        csv_path = Path("metrics_long.csv")  # 退回当前目录

    if csv_path.exists():
        plot_boxgrid(csv_path)
        plot_bar_mean_sd(csv_path)
    else:
        print(f"[FIG][Warn] 未找到 {METRICS_CSV} / ./metrics_long.csv，跳过箱线图与柱状图生成。")

if __name__ == "__main__":
    main()
