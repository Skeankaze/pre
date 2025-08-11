# 导包
import argparse
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ===== 可选：TM-30 依赖（第二问用到）。若未安装，会在运行时给出友好提示 =====
_HAS_COLOUR = True
try:
    import colour
    from colour.quality import tm3018
except Exception:
    _HAS_COLOUR = False


#读取和插值
def _to_float(x):
    if pd.isna(x):
        return None
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(x))
    return float(m.group()) if m else None


def _interp(x_src, y_src, kind="linear", fill=0.0):
    return interp1d(
        x_src, y_src, kind=kind,
        bounds_error=False,
        fill_value=fill if fill is not None else "extrapolate"
    )


# ======================================================================
# 第一问：CCT + Duv
# ======================================================================
def load_spd_from_excel(excel_path, sheet_spd="Problem 1"):
    df = pd.read_excel(excel_path, sheet_name=sheet_spd, skiprows=1)
    wl = df.iloc[:, 0].map(_to_float).dropna().astype(float).to_numpy()
    spd = df.iloc[:, 1].map(_to_float).dropna().astype(float).to_numpy()
    L = min(len(wl), len(spd))
    wl, spd = wl[:L], spd[:L]
    return wl, spd


def load_cie1931_from_excel(excel_path, sheet_cie="CIE1931"):
    cie = pd.read_excel(excel_path, sheet_name=sheet_cie, skiprows=1)
    λ = cie.iloc[:, 0].map(_to_float).dropna().astype(float).to_numpy()
    x_bar = cie.iloc[:, 1].map(_to_float).dropna().astype(float).to_numpy()
    y_bar = cie.iloc[:, 2].map(_to_float).dropna().astype(float).to_numpy()
    z_bar = cie.iloc[:, 3].map(_to_float).dropna().astype(float).to_numpy()
    L = min(len(λ), len(x_bar), len(y_bar), len(z_bar))
    return λ[:L], x_bar[:L], y_bar[:L], z_bar[:L]


def compute_XYZ_from_spd(λ_test, spd_test, λ_cie, x_bar, y_bar, z_bar):
    fx = _interp(λ_cie, x_bar, fill=0.0)
    fy = _interp(λ_cie, y_bar, fill=0.0)
    fz = _interp(λ_cie, z_bar, fill=0.0)

    X_raw = np.sum(spd_test * fx(λ_test))
    Y_raw = np.sum(spd_test * fy(λ_test))
    Z_raw = np.sum(spd_test * fz(λ_test))

    # 归一化系数：k = 100 / Σ S(λ)·ȳ(λ)
    if Y_raw == 0:
        return np.nan, np.nan, np.nan
    k = 100.0 / Y_raw
    return k * X_raw, k * Y_raw, k * Z_raw


def xyz_to_xy(X, Y, Z):
    """XYZ -> xy 色品坐标。"""
    den = X + Y + Z
    if den == 0 or not np.isfinite(den):
        return np.nan, np.nan
    return X / den, Y / den


def mccamy_cct(x, y):
    """
    McCamy 经验公式,适用于常见相关色温区间的快速估算。
    """
    if not np.isfinite(x) or not np.isfinite(y) or (y - 0.1858) == 0:
        return np.nan
    n = (x - 0.3320) / (y - 0.1858)
    return -437.0 * n**3 + 3601.0 * n**2 - 6861.0 * n + 5514.31


def xy_to_uv1960(x, y):
    """xy -> CIE 1960 UCS (u, v)。"""
    den = (-2.0 * x + 12.0 * y + 3.0)
    if den == 0 or not np.isfinite(den):
        return np.nan, np.nan
    u = 4.0 * x / den
    v = 6.0 * y / den
    return u, v


def planck_uv_from_cct(cct):
    """
    由 CCT 计算普朗克轨迹上的 (u_p, v_p)。
    """
    if not np.isfinite(cct) or cct < 1000:
        cct = 1000.0
    u_p = (0.860117757 + 1.54118254e-4 * cct + 1.28641212e-7 * cct**2) / \
          (1 + 8.42420235e-4 * cct + 7.08145163e-7 * cct**2)
    v_p = (0.317398726 + 4.22806245e-5 * cct + 4.20481691e-8 * cct**2) / \
          (1 - 2.89741816e-5 * cct + 1.61456053e-7 * cct**2)
    return u_p, v_p


def compute_duv_from_xy_cct(x, y, cct):
    """
    把 (x,y) 转到 (u,v)，与普朗克点 (u_p, v_p) 的带符号距离作为 Duv。
    """
    u, v = xy_to_uv1960(x, y)
    if not np.isfinite(u) or not np.isfinite(v):
        return np.nan
    u_p, v_p = planck_uv_from_cct(cct)
    du = u - u_p
    dv = v - v_p
    duv = np.sqrt(du * du + dv * dv)
    # 方向：v - v_p 的符号（v 高于普朗克轨迹为正 -> 偏绿；低于为负 -> 偏品红）
    return float(np.sign(v - v_p) * duv)


def q1_cct_duv(excel_path, sheet_spd="Problem 1", sheet_cie="CIE1931"):
    """
    第一问：返回 CCT (K), Duv, 以及中间量（x,y,X,Y,Z）。
    """
    λ_test, spd_test = load_spd_from_excel(excel_path, sheet_spd)
    λ_cie, x_bar, y_bar, z_bar = load_cie1931_from_excel(excel_path, sheet_cie)

    X, Y, Z = compute_XYZ_from_spd(λ_test, spd_test, λ_cie, x_bar, y_bar, z_bar)
    x, y = xyz_to_xy(X, Y, Z)
    cct = mccamy_cct(x, y)
    duv = compute_duv_from_xy_cct(x, y, cct)
    return {
        "X": X, "Y": Y, "Z": Z,
        "x": x, "y": y,
        "CCT": cct, "Duv": duv
    }


# ======================================================================
# 第二问：TM-30-18 的 Rf / Rg
# ======================================================================
def q2_rf_rg(excel_path, sheet_spd="Problem 1"):
    """
    第二问：计算 TM-30 Rf / Rg。
    依赖 colour-science（内部的 tm3018 实现）。
    """
    if not _HAS_COLOUR:
        raise RuntimeError("未检测到 colour-science 库，请先安装：pip install colour-science")

    # 读取与清洗
    df = pd.read_excel(excel_path, header=None)
    wl = df.iloc[:, 0].map(_to_float).dropna().astype(float).to_numpy()
    val = df.iloc[:, 1].map(_to_float).dropna().astype(float).to_numpy()
    L = min(len(wl), len(val))
    wl, val = wl[:L], val[:L]

    # 构造光谱分布对象
    sd_test = colour.SpectralDistribution(dict(zip(wl, val)))

    # TM-30 计算
    spec = tm3018.colour_fidelity_index_ANSIIESTM3018(sd_test, additional_data=True)
    Rf = float(spec.R_f)
    Rg = float(spec.R_g)
    return {"Rf": Rf, "Rg": Rg}


# ======================================================================
# 第三问：melanopic DER
# ======================================================================
def q3_melanopic_der(excel_path, cs_path, d65_csv,
                     sheet_spd="Problem 1", sheet_cie="CIE1931"):
    """
    第三问：计算 melanopic DER（相对 D65）。
    先把 photopic ȳ 与 melanopic 灵敏度插值到各自波长，再做离散积分。
    """
    # 1) 测试光源 SPD
    spd_test = pd.read_excel(excel_path, sheet_name=sheet_spd, skiprows=1)
    λ_test = spd_test.iloc[:, 0].map(_to_float).dropna().astype(float).to_numpy()
    spd_val = spd_test.iloc[:, 1].map(_to_float).dropna().astype(float).to_numpy()
    L = min(len(λ_test), len(spd_val))
    λ_test, spd_val = λ_test[:L], spd_val[:L]

    # 2) photopic ȳ
    cie = pd.read_excel(excel_path, sheet_name=sheet_cie, skiprows=1)
    λ_cie = cie.iloc[:, 0].map(_to_float).dropna().astype(float).to_numpy()
    y_bar = cie.iloc[:, 2].map(_to_float).dropna().astype(float).to_numpy()
    fy = _interp(λ_cie, y_bar, fill=None)  # 允许外推（与 q3.py 的 'extrapolate' 等价）
    ybar_test = fy(λ_test)

    # 3) melanopic 灵敏度
    mel = pd.read_excel(cs_path)
    λ_mel = mel.iloc[:, 0].map(_to_float).dropna().astype(float).to_numpy()
    mel_val = mel.iloc[:, 1].astype(str).str.replace(",", "").str.replace(" ", "")
    mel_val = mel_val.map(_to_float).dropna().astype(float).to_numpy()
    fmel = _interp(λ_mel, mel_val, fill=0.0)
    melanopic_test = fmel(λ_test)

    # 4) D65 SPD
    d65 = pd.read_csv(d65_csv)
    λ_d65 = d65["Wavelength"].map(_to_float).dropna().astype(float).to_numpy()
    spd_d65 = d65["RelativeSPD"].map(_to_float).dropna().astype(float).to_numpy()
    ybar_d65 = fy(λ_d65)
    melanopic_d65 = fmel(λ_d65)

    # 5) 离散积分（步长按 1nm 近似；若波长不规则，因两边都同处理，该比值仍具可比性）
    Δλ = 1.0
    E_mel_test = float(np.sum(spd_val * melanopic_test) * Δλ)
    E_v_test   = float(np.sum(spd_val * ybar_test) * Δλ)
    E_mel_d65  = float(np.sum(spd_d65 * melanopic_d65) * Δλ)
    E_v_d65    = float(np.sum(spd_d65 * ybar_d65) * Δλ)

    mel_DER = (E_mel_test / E_v_test) / (E_mel_d65 / E_v_d65)
    return {"mel_DER": float(mel_DER)}


# ======================================================================
# 统一入口：一次性跑完 5 个指标
# ======================================================================
def run_all(excel_path="Problem 1.xlsx",
            cs_path="cs.xlsx",
            d65_csv="CIE_D65_380_780.csv",
            sheet_spd="Problem 1",
            sheet_cie="CIE1931"):

    # 第一问
    q1 = q1_cct_duv(excel_path, sheet_spd=sheet_spd, sheet_cie=sheet_cie)
    # 第二问
    try:
        q2 = q2_rf_rg(excel_path, sheet_spd=sheet_spd)
    except Exception as e:
        # 若暂未装 colour-science，可以先得到其余 3 项
        q2 = {"Rf": np.nan, "Rg": np.nan}
        print("【提示】TM-30 计算失败（可能未安装 colour-science）：", e)
    # 第三问
    q3 = q3_melanopic_der(excel_path, cs_path, d65_csv, sheet_spd=sheet_spd, sheet_cie=sheet_cie)

    # 合并为一行结果
    row = {
        **{k: q1[k] for k in ["X", "Y", "Z", "x", "y"]},
        "CCT": q1["CCT"],
        "Duv": q1["Duv"],
        "Rf": q2["Rf"],
        "Rg": q2["Rg"],
        "mel_DER": q3["mel_DER"],
    }
    df = pd.DataFrame([row])
    return df


def parse_args():
    p = argparse.ArgumentParser(description="整合版：CCT/Duv + Rf/Rg + melanopic DER 一次性输出")
    p.add_argument("--excel", default="Problem 1.xlsx", help="含 SPD 与 CIE1931 的 Excel 文件路径")
    p.add_argument("--sheet_spd", default="Problem 1", help="Excel 中 SPD 的工作表名")
    p.add_argument("--sheet_cie", default="CIE1931", help="Excel 中 CIE1931 的工作表名")
    p.add_argument("--cs", default="cs.xlsx", help="melanopic 灵敏度表路径 (xlsx)")
    p.add_argument("--d65", default="CIE_D65_380_780.csv", help="D65 SPD (csv)")
    p.add_argument("--save_csv", action="store_true", help="是否另存结果至 outputs/metrics_table.csv")
    return p.parse_args()


def main():
    args = parse_args()
    out = run_all(
        excel_path=args.excel,
        cs_path=args.cs,
        d65_csv=args.d65,
        sheet_spd=args.sheet_spd,
        sheet_cie=args.sheet_cie,
    )

    # 终端打印
    row = out.iloc[0]
    print("\n=== 统一结果 ===")
    print(f"X={row['X']:.4f}, Y={row['Y']:.4f}, Z={row['Z']:.4f}")
    print(f"x={row['x']:.5f}, y={row['y']:.5f}")
    print(f"CCT={row['CCT']:.2f} K")
    print(f"Duv={row['Duv']:.5f}")
    print(f"Rf={row['Rf'] if np.isfinite(row['Rf']) else 'NaN'}")
    print(f"Rg={row['Rg'] if np.isfinite(row['Rg']) else 'NaN'}")
    print(f"mel-DER={row['mel_DER']:.6f}")



if __name__ == "__main__":
    main()
