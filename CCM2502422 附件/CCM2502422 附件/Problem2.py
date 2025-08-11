# ========================== 导包与基础配置 ==========================
import os
import warnings
from typing import Tuple, Dict, List


warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="colour")
try:
    from colour.utilities import ColourUsageWarning
    warnings.filterwarnings("ignore", category=ColourUsageWarning)
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import nnls, minimize, differential_evolution

import colour
from colour.quality import tm3018

# 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ========================== 全局配置 ==========================
LED_XLSX   = "Problem 2_LED_SPD.xlsx"
FILE_CIE   = "Problem 1.xlsx"
SHEET_CIE  = "CIE1931"
FILE_MEL   = "cs.xlsx"
FILE_D65   = "CIE_D65_380_780.csv"

# True：仅 SLSQP；False：DE + SLSQP
FAST_MODE  = True
# 是否打印通道体检与采样检查
PRINT_CHECK = True

# 题面指标约束
SPECS = {
    "day": {   # 昼间：高色温，高显色
        "CCT_min": 6000.0, "CCT_max": 7000.0,
        "Duv_abs_max": 0.0060,
        "Rg_min": 95.0,    "Rg_max": 105.0,
        "Rf_min": 90.0
    },
    "night": { # 夜间：低色温，限制 melanopic DER
        "CCT_min": 2700.0, "CCT_max": 3300.0,
        "Duv_abs_max": 0.0060,
        "Rg_min": 95.0,    "Rg_max": 105.0,
        "Rf_min": 80.0,
        "melDER_max": 0.50
    }
}


# ========================== I/O 与基础工具 ==========================
def read_led_spd_from_excel(xlsx_path: str, sheet_name=0) -> Tuple[np.ndarray, np.ndarray]:
    """读取 5 通道 LED 光谱，返回：波长向量 wl、通道矩阵 channels（列顺序：B,G,R,WW,CW）"""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=0)
    if df.shape[1] < 6:
        raise ValueError("Excel 需至少 6 列（波长 + 5 通道：B,G,R,WW,CW）。")
    wl = df.iloc[:, 0].to_numpy(float)
    ch = df.iloc[:, 1:6].to_numpy(float)
    ch[ch < 0] = 0.0
    return wl, ch


def sanitize_wl_channels(wl: np.ndarray, channels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """清洗波长与通道数据（去无效行、对齐波长、保证单调）"""
    df = pd.DataFrame(np.column_stack([wl, channels]), columns=["wl", "B", "G", "R", "WW", "CW"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    df = df.groupby("wl", as_index=False).mean().sort_values("wl")
    wl_clean = df["wl"].to_numpy(float)
    ch_clean = df[["B", "G", "R", "WW", "CW"]].to_numpy(float)
    if wl_clean.size < 2:
        raise ValueError("清洗后有效波长点 < 2，请检查输入数据。")
    return wl_clean, ch_clean


def is_1nm_grid_380_780(wl: np.ndarray) -> bool:
    """判断是否已是 380–780 nm，1 nm 等间隔网格"""
    if wl.min() > 380 or wl.max() < 780:
        return False
    step = np.diff(wl)
    return np.allclose(step, 1.0) and abs(wl[0] - 380) <= 1e-6 and abs(wl[-1] - 780) <= 1e-6


def resample_to_1nm(wl: np.ndarray, spd: np.ndarray, lo=380, hi=780) -> Tuple[np.ndarray, np.ndarray]:
    """重采样到 1 nm 网格（闭区间），线性插值 + 越界置零"""
    mask = np.isfinite(wl) & np.isfinite(spd)
    wl, spd = wl[mask], spd[mask]
    if wl.size == 0:
        raise ValueError("resample_to_1nm: 输入为空。")
    order = np.argsort(wl)
    wl, spd = wl[order], spd[order]
    wl_u, idx = np.unique(wl, return_index=True)
    wl, spd = wl_u, spd[idx]

    lo_eff = max(lo, int(np.ceil(wl.min())))
    hi_eff = min(hi, int(np.floor(wl.max())))
    if hi_eff < lo_eff:
        raise ValueError(f"resample_to_1nm: 范围 [{wl.min():.1f},{wl.max():.1f}] 与 [{lo},{hi}] 无交集。")
    wl_grid = np.arange(lo_eff, hi_eff + 1, 1.0)
    spd_grid = np.interp(wl_grid, wl, spd, left=0.0, right=0.0)
    if not np.any(spd_grid > 0):
        raise ValueError("resample_to_1nm: 重采样后全为 0。")
    return wl_grid, spd_grid


def maybe_resample_1nm(wl: np.ndarray, spd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """若非 1 nm 网格，则重采样到 380–780 nm"""
    return (wl, spd) if is_1nm_grid_380_780(wl) else resample_to_1nm(wl, spd, 380, 780)


def normalize_spd(spd: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """按面积归一化 SPD（积分为 1）"""
    area = np.trapz(spd, wl)
    return spd if area <= 1e-20 else spd / area


def combine_spd(channels: np.ndarray, w: np.ndarray) -> np.ndarray:
    """通道线性混光"""
    spd = channels @ w
    spd[spd < 0] = 0.0
    return spd


def project_to_simplex(w: np.ndarray, s: float = 1.0) -> np.ndarray:
    """投影到概率单纯形（w>=0 且 sum(w)=s）"""
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(w) + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    return np.maximum(w - theta, 0.0)


# ========================== 参考数据与色度计算 ==========================
class Refs:
    """封装 CIE 1931 / melanopic / D65 数据与插值函数"""
    def __init__(self, cie_path: str, cie_sheet: str, mel_path: str, d65_csv: str):
        # CIE 1931
        cie_raw = pd.read_excel(cie_path, sheet_name=cie_sheet, skiprows=1)
        cie = cie_raw.iloc[:, :4].copy()
        cie.columns = ["wl", "xbar", "ybar", "zbar"]
        for c in ["wl", "xbar", "ybar", "zbar"]:
            cie[c] = pd.to_numeric(cie[c], errors="coerce")
        cie = cie.dropna(subset=["wl"]).fillna(0.0).sort_values("wl")
        self._λ_cie = cie["wl"].to_numpy(float)
        self._xbar  = cie["xbar"].to_numpy(float)
        self._ybar  = cie["ybar"].to_numpy(float)
        self._zbar  = cie["zbar"].to_numpy(float)
        self._ix = interp1d(self._λ_cie, self._xbar, bounds_error=False, fill_value=0.0)
        self._iy = interp1d(self._λ_cie, self._ybar, bounds_error=False, fill_value=0.0)
        self._iz = interp1d(self._λ_cie, self._zbar, bounds_error=False, fill_value=0.0)

        # melanopic
        mel_df = pd.read_excel(mel_path).iloc[:, :2].copy()
        mel_df.columns = ["wl", "mel"]
        mel_df["wl"] = pd.to_numeric(mel_df["wl"], errors="coerce")
        mel_df["mel"] = (mel_df["mel"].astype(str)
                         .str.replace(",", "")
                         .str.replace(" ", ""))
        mel_df["mel"] = pd.to_numeric(mel_df["mel"], errors="coerce").fillna(0.0)
        mel_df = mel_df.dropna(subset=["wl"]).sort_values("wl")
        self._λ_mel = mel_df["wl"].to_numpy(float)
        self._mel   = mel_df["mel"].to_numpy(float)
        self._im = interp1d(self._λ_mel, self._mel, bounds_error=False, fill_value=0.0)

        # D65
        d65 = pd.read_csv(d65_csv).iloc[:, :2].copy()
        d65.columns = ["wl", "spd"]
        d65["wl"] = pd.to_numeric(d65["wl"], errors="coerce")
        d65["spd"] = pd.to_numeric(d65["spd"], errors="coerce").fillna(0.0)
        d65 = d65.dropna(subset=["wl"]).sort_values("wl")
        self._λ_d65 = d65["wl"].to_numpy(float)
        self._spd_d65 = d65["spd"].to_numpy(float)
        self._ybar_d65 = self._iy(self._λ_d65)
        self._mel_d65  = self._im(self._λ_d65)

    # 插值访问器
    def xbar(self, wl): return self._ix(wl)
    def ybar(self, wl): return self._iy(wl)
    def zbar(self, wl): return self._iz(wl)
    def melanopic_sens(self, wl): return self._im(wl)

    @property
    def d65_wl(self): return self._λ_d65
    @property
    def d65_spd(self): return self._spd_d65
    @property
    def d65_ybar(self): return self._ybar_d65
    @property
    def d65_mel(self): return self._mel_d65


def compute_XYZ(wl: np.ndarray, spd: np.ndarray, refs: Refs) -> Tuple[float, float, float]:
    """积分计算三刺激值 XYZ（并按 Y=100 归一）"""
    xbar, ybar, zbar = refs.xbar(wl), refs.ybar(wl), refs.zbar(wl)
    X = float(np.sum(spd * xbar))
    Y = float(np.sum(spd * ybar))
    Z = float(np.sum(spd * zbar))
    k = 100.0 / max(Y, 1e-20)
    return k * X, k * Y, k * Z


def XYZ_to_xy(X: float, Y: float, Z: float) -> Tuple[float, float]:
    """XYZ → CIE1931 xy"""
    den = X + Y + Z
    return (np.nan, np.nan) if den <= 1e-20 else (X / den, Y / den)


def xy_to_1960uv(x: float, y: float) -> Tuple[float, float]:
    """CIE1931 xy → 1960 uv"""
    den = -2.0 * x + 12.0 * y + 3.0
    return (np.nan, np.nan) if abs(den) <= 1e-20 else (4.0 * x / den, 6.0 * y / den)


def compute_CCT_Duv_from_spd(wl: np.ndarray, spd: np.ndarray, refs: Refs) -> Tuple[float, float, float, float]:
    """基于 1960 uv 与黑体轨迹近似求 CCT 与 Duv（插值 + 二次细化）"""
    wl1, spd1 = maybe_resample_1nm(wl, spd)
    X, Y, Z = compute_XYZ(wl1, spd1, refs)
    x, y = XYZ_to_xy(X, Y, Z)
    u, v = xy_to_1960uv(x, y)
    if not np.isfinite(u) or not np.isfinite(v):
        return np.nan, np.nan, float(u), float(v)

    def uv_p(t):
        up = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t**2) / (1 + 8.42420235e-4 * t + 7.08145163e-7 * t**2)
        vp = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t**2) / (1 - 2.89741816e-5 * t + 1.61456053e-7 * t**2)
        return up, vp

    cct_grid = np.arange(1000.0, 25000.0 + 1e-9, 20.0)
    up, vp = uv_p(cct_grid)
    dist2 = (u - up) ** 2 + (v - vp) ** 2
    idx = int(np.argmin(dist2))
    cct_best = float(cct_grid[idx])

    i0 = max(1, idx - 1)
    i1 = min(len(cct_grid) - 2, idx + 1)
    xs = cct_grid[i0 - 1:i1 + 2] if (i0 - 1 >= 0 and i1 + 1 < len(cct_grid)) else cct_grid[i0:i1 + 1]
    if xs.size >= 3:
        us = (0.860117757 + 1.54118254e-4 * xs + 1.28641212e-7 * xs**2) / (1 + 8.42420235e-4 * xs + 7.08145163e-7 * xs**2)
        vs = (0.317398726 + 4.22806245e-5 * xs + 4.20481691e-8 * xs**2) / (1 - 2.89741816e-5 * xs + 1.61456053e-7 * xs**2)
        ys = (u - us) ** 2 + (v - vs) ** 2
        A = np.vstack([xs**2, xs, np.ones_like(xs)]).T
        try:
            a, b, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
            if a > 0:
                cct_refined = -b / (2 * a)
                if 1000.0 <= cct_refined <= 25000.0:
                    cct_best = float(cct_refined)
        except Exception:
            pass

    upb, vpb = uv_p(cct_best)
    Duv = float(np.hypot(u - upb, v - vpb))
    Duv *= np.sign(v - vpb) or 1.0
    return cct_best, Duv, float(u), float(v)


def compute_TM30_Rf_Rg(wl: np.ndarray, spd: np.ndarray) -> Tuple[float, float]:
    """TM-30（Rf, Rg）"""
    wl1, spd1 = maybe_resample_1nm(wl, spd)
    sd = colour.SpectralDistribution(dict(zip(wl1, spd1)))
    spec = tm3018.colour_fidelity_index_ANSIIESTM3018(sd, additional_data=True)
    return float(spec.R_f), float(spec.R_g)


def compute_melanopic_DER(wl: np.ndarray, spd: np.ndarray, refs: Refs) -> float:
    """melanopic DER（相对 D65 的 melanopic/luminous 比值）"""
    wl1, spd1 = maybe_resample_1nm(wl, spd)
    mel = refs.melanopic_sens(wl1)
    ybar = refs.ybar(wl1)
    E_mel_t = np.trapz(spd1 * mel, wl1)
    E_v_t   = np.trapz(spd1 * ybar, wl1)
    E_mel_d = np.trapz(refs.d65_spd * refs.d65_mel, refs.d65_wl)
    E_v_d   = np.trapz(refs.d65_spd * refs.d65_ybar, refs.d65_wl)
    return np.inf if abs(E_v_t) <= 1e-20 else float((E_mel_t / E_v_t) / (E_mel_d / E_v_d))


def evaluate_metrics(wl: np.ndarray, spd: np.ndarray, refs: Refs) -> Dict[str, float]:
    """完整指标集合（昼/夜抛光使用）"""
    CCT, Duv, u, v = compute_CCT_Duv_from_spd(wl, spd, refs)
    Rf, Rg = compute_TM30_Rf_Rg(wl, spd)
    melDER = compute_melanopic_DER(wl, spd, refs)
    return dict(CCT=CCT, Duv=Duv, u=u, v=v, Rf=Rf, Rg=Rg, melDER=melDER)


def evaluate_metrics_night_fast(wl: np.ndarray, spd: np.ndarray, refs: Refs) -> Dict[str, float]:
    """夜间快速阶段指标（不算 TM-30，加速）"""
    CCT, Duv, u, v = compute_CCT_Duv_from_spd(wl, spd, refs)
    melDER = compute_melanopic_DER(wl, spd, refs)
    return dict(CCT=CCT, Duv=Duv, u=u, v=v, melDER=melDER)


# ========================== 约束罚则 ==========================
def hard_penalty_day(m: Dict[str, float]) -> float:
    """昼间硬约束罚则：CCT、|Duv|、Rg、Rf 全部合规"""
    s = SPECS["day"]; pen = 0.0
    if (not np.isfinite(m["CCT"])) or not (s["CCT_min"] <= m["CCT"] <= s["CCT_max"]):
        pen += 1e3 + 10.0 * min(abs(m["CCT"] - s["CCT_min"]), abs(m["CCT"] - s["CCT_max"]))
    if (not np.isfinite(m["Duv"])) or abs(m["Duv"]) > s["Duv_abs_max"]:
        pen += 2e3 + 2e5 * max(0.0, abs(m["Duv"]) - s["Duv_abs_max"])
    if (not np.isfinite(m["Rg"])) or not (s["Rg_min"] <= m["Rg"] <= s["Rg_max"]):
        pen += 1e3 + 50.0 * (max(0.0, s["Rg_min"] - m["Rg"]) + max(0.0, m["Rg"] - s["Rg_max"]))
    if (not np.isfinite(m["Rf"])) or m["Rf"] < s["Rf_min"]:
        pen += 1e3 + 50.0 * max(0.0, s["Rf_min"] - m["Rf"])
    return pen


def hard_penalty_night_full(m: Dict[str, float]) -> float:
    """夜间硬约束罚则：CCT、|Duv|、Rg、Rf、melDER"""
    s = SPECS["night"]; pen = 0.0
    if (not np.isfinite(m["CCT"])) or not (s["CCT_min"] <= m["CCT"] <= s["CCT_max"]):
        pen += 1e3 + 10.0 * min(abs(m["CCT"] - s["CCT_min"]), abs(m["CCT"] - s["CCT_max"]))
    if (not np.isfinite(m["Duv"])) or abs(m["Duv"]) > s["Duv_abs_max"]:
        pen += 2e3 + 2e5 * max(0.0, abs(m["Duv"]) - s["Duv_abs_max"])
    if (not np.isfinite(m["Rg"])) or not (s["Rg_min"] <= m["Rg"] <= s["Rg_max"]):
        pen += 1e3 + 50.0 * (max(0.0, s["Rg_min"] - m["Rg"]) + max(0.0, m["Rg"] - s["Rg_max"]))
    if (not np.isfinite(m["Rf"])) or m["Rf"] < s["Rf_min"]:
        pen += 1e3 + 50.0 * max(0.0, s["Rf_min"] - m["Rf"])
    if (not np.isfinite(m["melDER"])) or m["melDER"] > s["melDER_max"]:
        pen += 2e3 + 1e4 * max(0.0, m["melDER"] - s["melDER_max"])
    return pen


# ========================== 目标函数 ==========================
def loss_day(w: np.ndarray, wl: np.ndarray, channels: np.ndarray, refs: Refs) -> float:
    """昼间目标：先硬约束，再尽量最大化 Rf、轻度压 |Duv|"""
    if not np.all(np.isfinite(w)): return 1e12
    w = project_to_simplex(w)
    spd = normalize_spd(combine_spd(channels, w), wl)
    m = evaluate_metrics(wl, spd, refs)
    pen = hard_penalty_day(m)
    if pen >= 1e3:
        return pen
    return -m["Rf"] + 50.0 * abs(m["Duv"])


def loss_night_fast(w: np.ndarray, wl: np.ndarray, channels: np.ndarray, refs: Refs) -> float:
    """夜间快速目标：先把 CCT/Duv/melDER 拉入框，然后最小化 melDER + 轻度压 |Duv|"""
    if not np.all(np.isfinite(w)): return 1e12
    w = project_to_simplex(w)
    spd = normalize_spd(combine_spd(channels, w), wl)
    m = evaluate_metrics_night_fast(wl, spd, refs)
    s = SPECS["night"]
    pen = 0.0
    if (not np.isfinite(m["CCT"])) or not (s["CCT_min"] <= m["CCT"] <= s["CCT_max"]):
        pen += 1e3 + 10.0 * min(abs(m["CCT"] - s["CCT_min"]), abs(m["CCT"] - s["CCT_max"]))
    if (not np.isfinite(m["Duv"])) or abs(m["Duv"]) > s["Duv_abs_max"]:
        pen += 2e3 + 2e5 * max(0.0, abs(m["Duv"]) - s["Duv_abs_max"])
    return pen + m["melDER"] + 20.0 * abs(m["Duv"])


# ========================== 启发式初始化与优化器 ==========================
def make_reference_spd(wl: np.ndarray, target_cct: float) -> np.ndarray:
    """构造简单参考 SPD（高 CCT 偏蓝、低 CCT 偏红），用于 NNLS 初值"""
    center, width = (460.0, 70.0) if target_cct >= 5000 else (600.0, 90.0)
    ref = np.exp(-0.5 * ((wl - center) / width) ** 2)
    return normalize_spd(ref, wl)


def nnls_seed(channels: np.ndarray, ref_spd: np.ndarray) -> np.ndarray:
    """以 NNLS 结果为初值，并投影到简单形"""
    w0, _ = nnls(channels, ref_spd)
    return project_to_simplex(w0)


def run_local_only(loss_fn, wl, channels, refs, seeds: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """仅 SLSQP 局部优化（快）"""
    dim = channels.shape[1]
    bounds = [(0.0, 1.0)] * dim
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    best_w, best_f = None, np.inf
    for s in seeds:
        s = project_to_simplex(s)
        res = minimize(lambda w: loss_fn(w, wl, channels, refs),
                       x0=s, method="SLSQP", bounds=bounds, constraints=cons,
                       options=dict(maxiter=300, ftol=1e-8, disp=False))
        if res.fun < best_f:
            best_w, best_f = project_to_simplex(res.x), float(res.fun)
    return best_w, best_f


def run_global_local_optim(loss_fn, wl, channels, refs, seeds: List[np.ndarray], maxiter_de=120) -> Tuple[np.ndarray, float]:
    """DE 全局 + SLSQP 抛光（稳）"""
    dim = channels.shape[1]
    bounds = [(0.0, 1.0)] * dim
    init_pop = [project_to_simplex(s) for s in seeds]
    while len(init_pop) < 10:
        init_pop.append(project_to_simplex(np.random.rand(dim)))
    init_pop = np.array(init_pop)

    def de_loss(w): return loss_fn(w, wl, channels, refs)

    result_de = differential_evolution(
        de_loss, bounds=bounds, maxiter=maxiter_de, popsize=10,
        init=init_pop, polish=False, tol=1e-6, mutation=(0.5, 1.0),
        recombination=0.7, workers=-1
    )
    w_de = project_to_simplex(result_de.x)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res_local = minimize(lambda w: loss_fn(w, wl, channels, refs),
                         x0=w_de, method="SLSQP", bounds=bounds, constraints=cons,
                         options=dict(maxiter=500, ftol=1e-9, disp=False))
    return project_to_simplex(res_local.x), float(res_local.fun)


# ========================== 体检与输出 ==========================
def channel_health_report(wl: np.ndarray, channels: np.ndarray, name=("B", "G", "R", "WW", "CW")) -> None:
    """简单通道体检：>0 点数与 420–680 nm 中段最大连续零长度"""
    print("\n[CHANNEL HEALTH]")
    for i, n in enumerate(name):
        nz = channels[:, i] > 0
        cover = int(nz.sum())
        mask_mid = (wl >= 420) & (wl <= 680)
        mid = nz[mask_mid]
        zeros = (~mid).astype(int)
        max_gap = cur = 0
        for z in zeros:
            cur = cur + 1 if z == 1 else 0
            max_gap = max(max_gap, cur)
        print(f"{n}: >0点数={cover}, 中段最大连续零长度={max_gap} (1nm 步长)")


def plot_channels_and_mix(wl: np.ndarray, channels: np.ndarray, w: np.ndarray, title: str, fname: str) -> None:
    """绘制通道与合成光谱并保存"""
    spd_mix = combine_spd(channels, w)
    plt.figure(figsize=(9, 4))
    labels = ["B", "G", "R", "WW", "CW"]
    for i in range(channels.shape[1]):
        plt.plot(wl, channels[:, i], linewidth=1.0, alpha=0.85, label=labels[i])
    plt.plot(wl, spd_mix, linewidth=2.0, linestyle="--", label="Mix")
    plt.xlabel("波长 (nm)")
    plt.ylabel("相对光谱功率 (a.u.)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(ncol=6, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(fname, dpi=220)
    plt.close()


def print_solution(tag: str, wl: np.ndarray, channels: np.ndarray, w: np.ndarray, refs: Refs) -> None:
    """打印权重与关键指标"""
    spd = normalize_spd(combine_spd(channels, w), wl)
    m = evaluate_metrics(wl, spd, refs)
    print(f"\n==== {tag} ====")
    print("weights (B, G, R, WW, CW):", np.round(w, 4))
    for k in ["CCT", "Duv", "Rf", "Rg", "melDER"]:
        v = m.get(k, np.nan)
        print(f"{k}: {v:.4f}" if np.isfinite(v) else f"{k}: NaN")


# ========================== 主流程 ==========================
def main() -> None:
    # ---- 读取与清洗 ----
    wl, channels = read_led_spd_from_excel(LED_XLSX)
    wl, channels = sanitize_wl_channels(wl, channels)

    # ---- 重采样到 1 nm 网格（逐通道）----
    wl_grid, _ = resample_to_1nm(wl, channels[:, 0], 380, 780)
    channels = np.column_stack([
        np.interp(wl_grid, wl, channels[:, i], left=0.0, right=0.0)
        for i in range(channels.shape[1])
    ])
    wl = wl_grid

    # ---- 通道面积归一化（便于混光）----
    for i in range(channels.shape[1]):
        channels[:, i] = normalize_spd(channels[:, i], wl)

    for p in [FILE_CIE, FILE_MEL, FILE_D65]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少参考文件：{p}")
    refs = Refs(FILE_CIE, SHEET_CIE, FILE_MEL, FILE_D65)

    # ---- 检查信息 ----
    if PRINT_CHECK:
        print(f"[CHECK] wl = [{wl.min():.1f}, {wl.max():.1f}], N = {wl.size}")
        print(f"[CHECK] channels shape = {channels.shape}")
        area = [np.trapz(channels[:, i], wl) for i in range(5)]
        print(f"[CHECK] per-channel area = {np.round(area, 6)}")
        channel_health_report(wl, channels)
        w_demo = np.ones(5) / 5
        m_demo = evaluate_metrics(wl, normalize_spd(combine_spd(channels, w_demo), wl), refs)
        print("[CHECK] metrics sample:", {k: float(m_demo[k]) for k in ["CCT", "Duv", "Rf", "Rg", "melDER"]})

    # ================= 昼间优化（严格约束） =================
    ref_day = make_reference_spd(wl, 6500.0)
    seeds_day = [
        nnls_seed(channels, ref_day),
        project_to_simplex(np.array([0.10, 0.15, 0.10, 0.10, 0.55])),
        project_to_simplex(np.ones(5))
    ]
    if FAST_MODE:
        w_day, _ = run_local_only(loss_day, wl, channels, refs, seeds_day)
    else:
        w_day, _ = run_global_local_optim(loss_day, wl, channels, refs, seeds_day, maxiter_de=120)

    print_solution("Daytime (maximize Rf with hard constraints)", wl, channels, w_day, refs)
    plot_channels_and_mix(wl, channels, w_day, "Daytime Mix", "mix_daytime.png")

    # ================= 夜间优化（快速 + 抛光） =================
    ref_night = make_reference_spd(wl, 3000.0)
    seeds_night = [
        nnls_seed(channels, ref_night),
        project_to_simplex(np.array([0.25, 0.20, 0.03, 0.45, 0.07])),
        project_to_simplex(np.array([0.30, 0.25, 0.02, 0.40, 0.03]))
    ]
    if FAST_MODE:
        w_night, _ = run_local_only(loss_night_fast, wl, channels, refs, seeds_night)
    else:
        w_night, _ = run_global_local_optim(loss_night_fast, wl, channels, refs, seeds_night, maxiter_de=120)

    # 抛光阶段（完整指标 + 硬约束）
    def night_polish_loss(w, wl_, ch_, refs_):
        w = project_to_simplex(w)
        spd = normalize_spd(combine_spd(ch_, w), wl_)
        m = evaluate_metrics(wl_, spd, refs_)
        pen = hard_penalty_night_full(m)
        if pen >= 1e3:
            return pen
        return m["melDER"] + 20.0 * abs(m["Duv"])

    dim = channels.shape[1]
    bounds = [(0.0, 1.0)] * dim
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    res = minimize(lambda w: night_polish_loss(w, wl, channels, refs),
                   x0=w_night, method="SLSQP", bounds=bounds, constraints=cons,
                   options=dict(maxiter=150, ftol=1e-8, disp=False))
    w_night = project_to_simplex(res.x)

    print_solution("Nighttime (hard-constrained, polished)", wl, channels, w_night, refs)
    plot_channels_and_mix(wl, channels, w_night, "Nighttime Mix", "mix_nighttime.png")

    print("\n[Saved] 图像文件：mix_daytime.png, mix_nighttime.png")
    print("[Done] 优化完成。")


# ========================== 入口 ==========================
if __name__ == "__main__":
    np.random.seed(42)
    main()
