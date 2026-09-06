# -*- coding: utf-8 -*-
"""
変圧器オイル温度(OT)予測 PoC — 分析・前処理スクリプト

1. 1年分(2016-07-01〜2017-06-30)の抽出と時系列特徴量の付与
2. 11月〜1月のOT急落の要因検証(設備停止 vs 外気/センサ要因)
3. ラグ特徴量の有効性確認(クロス相関分析)
4. 図の保存と考察の要約出力

グラフのラベルは文字化け回避のため英語、考察はコンソールに日本語で出力する。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET = "OT"
PERIOD_START, PERIOD_END = "2016-07-01", "2017-07-01"  # end は排他 → 2017-06-30まで
WINTER_START, WINTER_END = "2016-11-01", "2017-02-01"
MAX_LAG = 24
FIG_DIR, OUT_DIR = "figures", "outputs"

COLORS = dict(zip(FEATURES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]))


# ----------------------------------------------------------------------------
# 1. 読み込み・前処理
# ----------------------------------------------------------------------------
def load_and_prepare(csv_path):
    """CSVを読み込み、1年分を抽出して時系列特徴量を付与する。"""
    df = pd.read_csv(csv_path, sep=",", header=0)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df_year = df[(df["date"] >= PERIOD_START) & (df["date"] < PERIOD_END)].copy()

    df_year["month"] = df_year["date"].dt.month
    df_year["day"] = df_year["date"].dt.day
    df_year["hour"] = df_year["date"].dt.hour
    df_year["dayofweek"] = df_year["date"].dt.dayofweek  # 0=月曜

    # 欠損時刻(記録落ち)の確認 — 1時間刻みで連続しているか
    expected = pd.date_range(df_year["date"].min(), df_year["date"].max(), freq="h")
    missing = len(expected) - len(df_year)

    info = {
        "rows": len(df_year),
        "missing_timestamps": missing,
        "nan_count": int(df_year[FEATURES + [TARGET]].isna().sum().sum()),
        "ot_min": df_year[TARGET].min(),
        "ot_max": df_year[TARGET].max(),
        "ot_mean": df_year[TARGET].mean(),
    }
    return df_year.reset_index(drop=True), info


# ----------------------------------------------------------------------------
# 2. 冬季OT急落の検証
# ----------------------------------------------------------------------------
def detect_drop_events(df_year, low_q=0.01, min_duration_h=2, merge_gap_h=6):
    """年間下位1%水準を下回るOT低下を「急落イベント」として抽出する。"""
    thr = df_year[TARGET].quantile(low_q)
    win = df_year[(df_year["date"] >= WINTER_START) & (df_year["date"] < WINTER_END)].copy()

    flag = win[TARGET] < thr
    # 連続区間へグルーピング(短い中断はまたいで結合する)
    idx = np.where(flag.values)[0]
    events = []
    if len(idx):
        start = prev = idx[0]
        for i in idx[1:]:
            if i - prev > merge_gap_h:
                events.append((start, prev))
                start = i
            prev = i
        events.append((start, prev))

    records = []
    for s, e in events:
        seg = win.iloc[s : e + 1]
        if len(seg) < min_duration_h:
            continue
        records.append(
            {
                "start": seg["date"].iloc[0],
                "end": seg["date"].iloc[-1],
                "duration_h": len(seg),
                "ot_min": seg[TARGET].min(),
                "ot_mean": seg[TARGET].mean(),
                "_seg": seg,
            }
        )
    return thr, records


def judge_cause(df_year, event, baseline_h=72):
    """イベント中の負荷が平常時ベースラインに対しどれだけ落ちたかで要因を判定する。

    負荷は負値も取るため、絶対値の平均比(retention)で評価する。
    """
    s, e = event["start"], event["end"]
    base = df_year[
        ((df_year["date"] >= s - pd.Timedelta(hours=baseline_h)) & (df_year["date"] < s))
        | ((df_year["date"] > e) & (df_year["date"] <= e + pd.Timedelta(hours=baseline_h)))
    ]
    seg = event["_seg"]

    ret = {}
    for c in FEATURES:
        b = np.abs(base[c]).mean()
        v = np.abs(seg[c]).mean()
        ret[c] = v / b if b > 1e-9 else np.nan

    vals = np.array([ret[c] for c in FEATURES], dtype=float)
    mean_ret = np.nanmean(vals)
    if mean_ret < 0.20:
        verdict = "設備停止(負荷ほぼ消失)"
    elif mean_ret < 0.70:
        verdict = "部分停止・負荷切替の可能性"
    else:
        verdict = "負荷は維持(外気温/放熱/センサ要因)"

    ot_base = base[TARGET].mean()
    return ret, mean_ret, verdict, ot_base


def plot_drop_event(df_year, event, name, rank, pad_h=48):
    """急落イベント前後をOT(左軸)と負荷(右軸)の2軸でプロットする。"""
    s, e = event["start"], event["end"]
    w = df_year[
        (df_year["date"] >= s - pd.Timedelta(hours=pad_h))
        & (df_year["date"] <= e + pd.Timedelta(hours=pad_h))
    ]

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(w["date"], w[TARGET], color="red", linewidth=2, label="OT (Target)")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.axvspan(s, e, color="red", alpha=0.08)
    ax1.set_ylabel("Oil Temperature (OT) [degC]", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    for c in FEATURES:
        ax2.plot(w["date"], w[c], color=COLORS[c], alpha=0.75, linestyle="--",
                 linewidth=1.1, label=c)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax2.set_ylabel("Power Load", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", ncol=4, fontsize=9)
    ax1.set_title(
        "{}: OT Drop Event #{}  ({:%Y-%m-%d %H:%M} - {:%Y-%m-%d %H:%M}, min OT={:.2f})".format(
            name, rank, s, e, event["ot_min"]
        )
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "{}_drop_event_{}.png".format(name, rank))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_winter_overview(df_year, thr, name):
    """冬季(11-1月)全体のOTと負荷を俯瞰する。"""
    w = df_year[(df_year["date"] >= WINTER_START) & (df_year["date"] < WINTER_END)]

    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(w["date"], w[TARGET], color="red", linewidth=1.4, label="OT (Target)")
    ax1.axhline(thr, color="darkred", linestyle=":", linewidth=1.2,
                label="drop threshold ({:.2f})".format(thr))
    ax1.set_ylabel("Oil Temperature (OT) [degC]", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    for c in FEATURES:
        ax2.plot(w["date"], w[c], color=COLORS[c], alpha=0.55, linewidth=0.8, label=c)
    ax2.set_ylabel("Power Load", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", ncol=4, fontsize=9)
    ax1.set_title("{}: Winter Overview (Nov 2016 - Jan 2017) - OT vs Power Loads".format(name))
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "{}_winter_overview.png".format(name))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# 3. クロス相関(ラグ)分析
# ----------------------------------------------------------------------------
def lag_correlation(df_year):
    """lag=0..24 の corr(OT[t], X[t-lag]) を算出する。"""
    rows = {}
    for c in FEATURES:
        rows[c] = [df_year[TARGET].corr(df_year[c].shift(lag)) for lag in range(MAX_LAG + 1)]
    return pd.DataFrame(rows, index=pd.Index(range(MAX_LAG + 1), name="lag_h"))


def rolling_load_correlation(df_year, windows=(1, 3, 6, 12, 24, 48, 72, 168)):
    """過去k時間の平均負荷(累積熱入力の代理)とOTの相関。"""
    rows = {}
    for c in FEATURES:
        rows[c] = [df_year[TARGET].corr(df_year[c].rolling(k).mean()) for k in windows]
    return pd.DataFrame(rows, index=pd.Index(windows, name="window_h"))


def plot_lag_curves(lag_df, roll_df, name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    for c in FEATURES:
        ax.plot(lag_df.index, lag_df[c], marker="o", markersize=3, color=COLORS[c], label=c)
        best = lag_df[c].abs().idxmax()
        ax.scatter([best], [lag_df[c].loc[best]], color=COLORS[c], s=90, zorder=5,
                   edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lag [hours]  (corr(OT[t], X[t-lag]))")
    ax.set_ylabel("Pearson correlation with OT")
    ax.set_title("{}: Cross-correlation OT vs Loads (lag 0-{}h)".format(name, MAX_LAG))
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for c in FEATURES:
        ax.plot(roll_df.index, roll_df[c], marker="s", markersize=4, color=COLORS[c], label=c)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks(list(roll_df.index))
    ax.set_xticklabels([str(k) for k in roll_df.index])
    ax.set_xlabel("Rolling mean window [hours] (accumulated heat input proxy)")
    ax.set_ylabel("Pearson correlation with OT")
    ax.set_title("{}: OT vs Accumulated (rolling mean) Load".format(name))
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "{}_lag_correlation.png".format(name))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_year_overview(df_year, name):
    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(df_year["date"], df_year[TARGET], color="gray", alpha=0.35, linewidth=0.7,
             label="OT (1h)")
    ax1.plot(df_year["date"], df_year[TARGET].rolling(168).mean(), color="red", linewidth=2,
             label="OT 7d rolling mean")
    ax1.set_ylabel("Oil Temperature (OT) [degC]", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    for c in FEATURES:
        ax2.plot(df_year["date"], df_year[c].rolling(168).mean(), color=COLORS[c],
                 alpha=0.8, linestyle="--", linewidth=1.2, label="{} (7d)".format(c))
    ax2.set_ylabel("Power Load (7d rolling mean)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", ncol=4, fontsize=8)
    ax1.set_title("{}: 1-Year Overview (2016-07-01 - 2017-06-30)".format(name))
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "{}_year_overview.png".format(name))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------------
def analyze(name, csv_path):
    print("\n" + "=" * 78)
    print("■ {}  ({})".format(name, csv_path))
    print("=" * 78)

    df_year, info = load_and_prepare(csv_path)
    print("[1] 前処理 — 対象期間 {} 〜 2017-06-30".format(PERIOD_START))
    print("    行数: {} / 欠測時刻: {} / NaN: {}".format(
        info["rows"], info["missing_timestamps"], info["nan_count"]))
    print("    OT レンジ: {:.2f} 〜 {:.2f} ℃ (平均 {:.2f} ℃)".format(
        info["ot_min"], info["ot_max"], info["ot_mean"]))
    print("    追加特徴量: month, day, hour, dayofweek")

    hourly = df_year.groupby("hour")[TARGET].mean()
    print("    時刻別平均OT 最大/最小: {}時 {:.2f}℃ / {}時 {:.2f}℃ (日内振幅 {:.2f}℃)".format(
        hourly.idxmax(), hourly.max(), hourly.idxmin(), hourly.min(),
        hourly.max() - hourly.min()))
    monthly = df_year.groupby("month")[TARGET].mean()
    print("    月別平均OT 最大/最小: {}月 {:.2f}℃ / {}月 {:.2f}℃".format(
        monthly.idxmax(), monthly.max(), monthly.idxmin(), monthly.min()))
    dow = df_year.groupby("dayofweek")[TARGET].mean()
    print("    曜日別平均OT 最大/最小: {} {:.2f}℃ / {} {:.2f}℃ (差 {:.2f}℃)".format(
        "月火水木金土日"[dow.idxmax()], dow.max(), "月火水木金土日"[dow.idxmin()], dow.min(),
        dow.max() - dow.min()))

    out_csv = os.path.join(OUT_DIR, "{}_1year_features.csv".format(name))
    df_year.to_csv(out_csv, index=False)

    # --- 2. 冬季急落の検証 ---
    thr, events = detect_drop_events(df_year)
    print("\n[2] 11〜1月のOT急落検証 — 判定閾値(年間下位1%): OT < {:.2f}℃".format(thr))
    print("    検出イベント数: {}".format(len(events)))
    paths = [plot_winter_overview(df_year, thr, name)]

    verdicts = []
    for rank, ev in enumerate(sorted(events, key=lambda x: x["ot_min"])[:5], start=1):
        ret, mean_ret, verdict, ot_base = judge_cause(df_year, ev)
        verdicts.append((ev, mean_ret, verdict))
        print("\n    ● イベント{}: {:%m/%d %H:%M} 〜 {:%m/%d %H:%M} ({}時間)".format(
            rank, ev["start"], ev["end"], ev["duration_h"]))
        print("      OT: 最低 {:.2f}℃ / 期間平均 {:.2f}℃ (前後72hの平均 {:.2f}℃ → {:+.2f}℃)".format(
            ev["ot_min"], ev["ot_mean"], ot_base, ev["ot_mean"] - ot_base))
        print("      負荷の残存率(|イベント平均| / |前後72h平均|):")
        print("        " + "  ".join("{}={:5.1f}%".format(c, ret[c] * 100) for c in FEATURES))
        print("      → 平均残存率 {:.1f}%  ⇒ 判定: {}".format(mean_ret * 100, verdict))
        paths.append(plot_drop_event(df_year, ev, name, rank))

    # --- 3. ラグ相関 ---
    lag_df = lag_correlation(df_year)
    roll_df = rolling_load_correlation(df_year)
    lag_df.to_csv(os.path.join(OUT_DIR, "{}_lag_correlation.csv".format(name)))

    print("\n[3] クロス相関分析 (lag 0〜{}h)".format(MAX_LAG))
    print("    変数    lag0     最適lag   その相関   (改善幅)")
    best_rows = []
    for c in FEATURES:
        best = lag_df[c].abs().idxmax()
        gain = abs(lag_df[c].loc[best]) - abs(lag_df[c].loc[0])
        best_rows.append({"feature": c, "corr_lag0": lag_df[c].loc[0],
                          "best_lag_h": best, "corr_best": lag_df[c].loc[best], "gain": gain})
        print("    {:5s}  {:+.3f}     {:>3d}h     {:+.3f}    ({:+.3f})".format(
            c, lag_df[c].loc[0], best, lag_df[c].loc[best], gain))
    best_df = pd.DataFrame(best_rows)
    top = best_df.iloc[best_df["corr_best"].abs().idxmax()]
    print("    → 最有力: {} (lag {}h, r={:+.3f})".format(
        top["feature"], int(top["best_lag_h"]), top["corr_best"]))

    print("    過去k時間の平均負荷とOTの相関(累積熱入力の代理):")
    for c in FEATURES:
        k = roll_df[c].abs().idxmax()
        print("      {:5s} 最良ウィンドウ {:>3d}h  r={:+.3f}  (瞬時値 r={:+.3f})".format(
            c, k, roll_df[c].loc[k], lag_df[c].loc[0]))
    paths.append(plot_lag_curves(lag_df, roll_df, name))
    paths.append(plot_year_overview(df_year, name))

    return {"name": name, "info": info, "thr": thr, "events": events,
            "verdicts": verdicts, "lag_df": lag_df, "roll_df": roll_df,
            "best_df": best_df, "paths": paths, "df_year": df_year}


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    results = [analyze("ETTh1", "./data/ETTh1.csv"), analyze("ETTh2", "./data/ETTh2.csv")]

    print("\n" + "=" * 78)
    print("■ 保存したファイル")
    print("=" * 78)
    for r in results:
        for p in r["paths"]:
            print("   ", p)
        print("    outputs/{0}_1year_features.csv, outputs/{0}_lag_correlation.csv".format(r["name"]))
    return results


if __name__ == "__main__":
    main()
