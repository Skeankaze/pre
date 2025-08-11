# 导包
import os
import sys
import pandas as pd

# -------------------------
# 参数区
# -------------------------
CSV_PATH = os.path.join("problem4_results", "metrics_long.csv")

# -------------------------
# 工具函数
# -------------------------
def safe_get(row: pd.Series, *keys, default=None):
    """
    安全取列：从若干候选列名中依次尝试取值，若均不存在或为缺失则返回 default。
    用于兼容 pingouin 不同版本的列名差异（如 p-corr / p-adjust / p-unc 等）。
    """
    for k in keys:
        if (k in row) and (pd.notna(row[k])):
            return row[k]
    return default


def print_mean_sd_by_condition(sub: pd.DataFrame, metric: str):
    order = ["A", "B", "C"]
    sub = sub.copy()
    sub["condition"] = pd.Categorical(sub["condition"], categories=order, ordered=True)

    g = sub.groupby("condition")["Value"].agg(["mean", "std"]).reindex(order)

    print(f"\n=== {metric} ===")
    for cond in order:
        if cond in g.index and pd.notna(g.loc[cond, "mean"]):
            m = g.loc[cond, "mean"]
            s = g.loc[cond, "std"]
            print(f"{metric} - 条件 {cond}: {m:.2f}±{s:.2f}")
        else:
            print(f"{metric} - 条件 {cond}: 无数据")


def run_rm_anova_and_posthoc(sub: pd.DataFrame):
    import pingouin as pg

    # --------------- 重复测量方差分析 ---------------
    aov = pg.rm_anova(dv="Value", within="condition", subject="subject",
                      data=sub, detailed=True)

    # 只取 “condition” 主效应行
    aov_cond = aov.loc[aov["Source"] == "condition"].iloc[0]
    p_val = safe_get(aov_cond, "p-GG-corr", "p-unc", default=float("nan"))
    print(f"RM-ANOVA p值（GG校正/或未校正）: {p_val:.4f}")

    # --------------- 事后两两比较（Bonferroni） ---------------
    post = pg.pairwise_tests(
        dv="Value",
        within="condition",
        subject="subject",
        data=sub,
        padjust="bonf"
    )

    print("事后配对（Bonferroni校正）：")
    for (a, b) in [("A", "B"), ("A", "C"), ("B", "C")]:
        q = post[(post["A"] == a) & (post["B"] == b)]
        if q.empty:
            print(f"{a} vs {b}: 数据不足，跳过")
            continue
        row = q.iloc[0]

        p_corr = safe_get(row, "p-corr", "p-adjust", "p-unc", default=float("nan"))

        eff = safe_get(row, "hedges", "cohen-d", "effsize", default=float("nan"))

        tail = safe_get(row, "tail", "alternative", default="two-sided")

        sig = "显著" if (pd.notna(p_corr) and p_corr < 0.05) else "不显著"
        eff_str = "NA" if pd.isna(eff) else f"{float(eff):.3f}"
        p_str = "NA" if pd.isna(p_corr) else f"{float(p_corr):.4f}"
        print(f"{a} vs {b}: p_adj = {p_str}（{sig}），效应量 {eff_str}（{tail}）")


# -------------------------
# 主流程
# -------------------------
def main():
    # —— 1) 读取 CSV 并宽转长 —— #
    if not os.path.exists(CSV_PATH):
        print(f"[错误] 找不到数据文件：{CSV_PATH}")
        print("请确认路径是否正确，或修改脚本顶部的 CSV_PATH。")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"[错误] 读取 CSV 失败：{e}")
        sys.exit(1)

    # 将宽表转成长表，保留 6 个要求的指标
    metrics = ["SOL", "TST", "SE", "N3pct", "REMpct", "Awakenings"]
    df_long = df.melt(
        id_vars=["subject", "condition"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Value"
    )

    # —— 2) 逐指标计算并输出 —— #
    for metric in df_long["Metric"].unique():
        sub = df_long[df_long["Metric"] == metric].copy()
        print_mean_sd_by_condition(sub, metric)
        try:
            run_rm_anova_and_posthoc(sub)
        except ModuleNotFoundError:
            # pingouin 未安装时给出清晰提示
            print("【提示】未安装 pingouin，无法执行 RM-ANOVA 与事后检验。")
            print("请先运行：pip install pingouin")
            break


if __name__ == "__main__":
    main()
