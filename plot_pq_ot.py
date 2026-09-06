# -*- coding: utf-8 -*-
"""
有効電力(UFL) / 無効電力(ULL) / オイル温度(OT) の関係を H・M・L 区分ごとに時系列で可視化する。

各図は3段構成(High / Middle / Low)で、
  左軸: OT(赤)
  右軸: 有効電力 P(青・実線), 無効電力 Q(橙・破線), 皮相電力 S=sqrt(P^2+Q^2)(灰・細線)
ラベルは文字化け回避のため英語表記。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEVELS = [("High", "HUFL", "HULL"), ("Middle", "MUFL", "MULL"), ("Low", "LUFL", "LULL")]
FIG_DIR = "figures"

# (ファイル名, 表示名, 開始, 終了, 平滑化窓[h])
WINDOWS = [
    ("summer_week", "Summer Week (2016-08-01 - 08-08)", "2016-08-01", "2016-08-08", None),
    ("winter_week", "Winter Week (2017-01-15 - 01-22)", "2017-01-15", "2017-01-22", None),
    ("year", "1 Year (7d rolling mean)", "2016-07-01", "2017-07-01", 168),
]


def load(csv_path):
    df = pd.read_csv(csv_path, sep=",", header=0)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def plot_window(df, name, key, title, start, end, smooth):
    d = df[(df["date"] >= start) & (df["date"] < end)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    for ax1, (level, p_col, q_col) in zip(axes, LEVELS):
        p, q = d[p_col], d[q_col]
        s = np.sqrt(p ** 2 + q ** 2)
        ot = d["OT"]
        if smooth:
            p, q, s = p.rolling(smooth).mean(), q.rolling(smooth).mean(), s.rolling(smooth).mean()
            ot = ot.rolling(smooth).mean()

        ax1.plot(d["date"], ot, color="red", linewidth=2.2, label="OT (Oil Temperature)", zorder=3)
        ax1.set_ylabel("OT [degC]", color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()
        ax2.plot(d["date"], p, color="#1f77b4", linewidth=1.4,
                 label="{} (Active power P)".format(p_col))
        ax2.plot(d["date"], q, color="#ff7f0e", linewidth=1.4, linestyle="--",
                 label="{} (Reactive power Q)".format(q_col))
        ax2.plot(d["date"], s, color="gray", linewidth=0.9, alpha=0.7,
                 label="S = sqrt(P^2+Q^2)")
        ax2.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax2.set_ylabel("Power Load", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")

        # 相関を注記(平滑化前の生データで計算)
        raw = d
        r_p = raw["OT"].corr(raw[p_col])
        r_q = raw["OT"].corr(raw[q_col])
        r_s = raw["OT"].corr(np.sqrt(raw[p_col] ** 2 + raw[q_col] ** 2))
        ax1.set_title(
            "{} level  |  corr(OT, P)={:+.3f}   corr(OT, Q)={:+.3f}   corr(OT, S)={:+.3f}".format(
                level, r_p, r_q, r_s),
            fontsize=11, loc="left")

        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", ncol=4, fontsize=8)
        ax1.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("{}: Active / Reactive Power vs Oil Temperature — {}".format(name, title),
                 fontsize=13, y=0.995)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "{}_PQ_OT_{}.png".format(name, key))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = []
    for name, csv in [("ETTh1", "./data/ETTh1.csv"), ("ETTh2", "./data/ETTh2.csv")]:
        df = load(csv)
        print("=" * 70)
        print("■ {}".format(name))
        for key, title, start, end, smooth in WINDOWS:
            paths.append(plot_window(df, name, key, title, start, end, smooth))

        # 1年分での相関サマリ
        d = df[(df["date"] >= "2016-07-01") & (df["date"] < "2017-07-01")]
        print("  区分   P(有効)  Q(無効)   S(皮相)  S^2(発熱代理)  S^2の最適lag")
        for level, p_col, q_col in LEVELS:
            s = np.sqrt(d[p_col] ** 2 + d[q_col] ** 2)
            best = max(range(25), key=lambda L: abs(d["OT"].corr((s ** 2).shift(L))))
            print("  {:7s} {:+.3f}   {:+.3f}   {:+.3f}   {:+.3f}        {:2d}h ({:+.3f})".format(
                level, d["OT"].corr(d[p_col]), d["OT"].corr(d[q_col]), d["OT"].corr(s),
                d["OT"].corr(s ** 2), best, d["OT"].corr((s ** 2).shift(best))))

    print("\n保存:")
    for p in paths:
        print("   ", p)


if __name__ == "__main__":
    main()
