# -*- coding: utf-8 -*-
"""
指定した変数ペアの相関と、オイル温度(OT)との時系列関係を可視化する。

対象ペア(相関値は2年全体):
  ETTh1: MUFL-HUFL (0.99) / MULL-HULL (0.93)
  ETTh2: HULL-HUFL (0.67) / MULL-HULL (0.91)

各ペアにつき1枚、4パネル構成:
  A: 散布図(月で着色) — 相関そのもの
  B: 散布図(OTで着色) — 相関構造とOTの関係
  C: 2年間の時系列(7日移動平均) + OT
  D: 1ヶ月の生データ時系列 + OT
ラベルは文字化け回避のため英語表記。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = "figures"
FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

PAIRS = {
    "ETTh1": [("MUFL", "HUFL"), ("MULL", "HULL")],
    "ETTh2": [("HULL", "HUFL"), ("MULL", "HULL")],
}

DETAIL_START, DETAIL_END = "2016-08-01", "2016-09-01"

KIND = {"HUFL": "Active P", "MUFL": "Active P", "LUFL": "Active P",
        "HULL": "Reactive Q", "MULL": "Reactive Q", "LULL": "Reactive Q"}


def load(name):
    d = pd.read_csv("./data/{}.csv".format(name), sep=",", header=0)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def plot_pair(d, name, a, b):
    r = d[a].corr(d[b])
    # 停止期間(全負荷0)を除いた相関も併記
    live = d[~(d[FEATURES] == 0).all(axis=1)]
    r_live = live[a].corr(live[b])

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    # --- A: 散布図(月で着色) ---
    sc = axA.scatter(d[b], d[a], c=d["date"].dt.month, cmap="twilight",
                     s=4, alpha=0.35, linewidths=0)
    lim = np.polyfit(d[b], d[a], 1)
    xs = np.linspace(d[b].min(), d[b].max(), 100)
    axA.plot(xs, np.polyval(lim, xs), color="black", linewidth=1.6,
             label="fit: y={:.3f}x{:+.3f}".format(*lim))
    axA.set_xlabel("{} [{}]".format(b, KIND[b]))
    axA.set_ylabel("{} [{}]".format(a, KIND[a]))
    axA.set_title("A) {} vs {}   r = {:+.3f}  (excl. shutdown: {:+.3f})".format(a, b, r, r_live),
                  fontsize=11)
    axA.legend(fontsize=9, loc="upper left")
    axA.grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=axA)
    cb.set_label("Month")

    # --- B: 散布図(OTで着色) ---
    sc2 = axB.scatter(d[b], d[a], c=d["OT"], cmap="coolwarm", s=4, alpha=0.5, linewidths=0)
    axB.set_xlabel("{} [{}]".format(b, KIND[b]))
    axB.set_ylabel("{} [{}]".format(a, KIND[a]))
    axB.set_title("B) same scatter, colored by Oil Temperature", fontsize=11)
    axB.grid(True, alpha=0.3)
    cb2 = fig.colorbar(sc2, ax=axB)
    cb2.set_label("OT [degC]")

    # --- C: 2年間の時系列(7日移動平均) ---
    axC.plot(d["date"], d[a].rolling(168).mean(), color="#1f77b4", linewidth=1.5, label=a)
    axC.plot(d["date"], d[b].rolling(168).mean(), color="#2ca02c", linewidth=1.5,
             linestyle="--", label=b)
    axC.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    axC.set_ylabel("Power Load (7d rolling mean)", color="#1f77b4")
    axC.tick_params(axis="y", labelcolor="#1f77b4")
    axC.set_xlabel("Date")
    axC2 = axC.twinx()
    axC2.plot(d["date"], d["OT"].rolling(168).mean(), color="red", linewidth=2, label="OT")
    axC2.set_ylabel("OT [degC]", color="red")
    axC2.tick_params(axis="y", labelcolor="red")
    l1, lb1 = axC.get_legend_handles_labels()
    l2, lb2 = axC2.get_legend_handles_labels()
    axC.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=9, ncol=3)
    axC.set_title("C) 2-year time series (7d rolling mean) with OT", fontsize=11)
    axC.grid(True, alpha=0.3)

    # --- D: 1ヶ月の生データ ---
    w = d[(d["date"] >= DETAIL_START) & (d["date"] < DETAIL_END)]
    axD.plot(w["date"], w[a], color="#1f77b4", linewidth=1.1, label=a)
    axD.plot(w["date"], w[b], color="#2ca02c", linewidth=1.1, linestyle="--", label=b)
    axD.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    axD.set_ylabel("Power Load", color="#1f77b4")
    axD.tick_params(axis="y", labelcolor="#1f77b4")
    axD.set_xlabel("Date")
    axD2 = axD.twinx()
    axD2.plot(w["date"], w["OT"], color="red", linewidth=1.8, label="OT")
    axD2.set_ylabel("OT [degC]", color="red")
    axD2.tick_params(axis="y", labelcolor="red")
    l1, lb1 = axD.get_legend_handles_labels()
    l2, lb2 = axD2.get_legend_handles_labels()
    axD.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=9, ncol=3)
    axD.set_title("D) Raw hourly detail (Aug 2016) with OT", fontsize=11)
    axD.grid(True, alpha=0.3)
    for lab in axD.get_xticklabels():
        lab.set_rotation(20)

    fig.suptitle("{}: {} vs {}  (r = {:+.3f})   —   relation to Oil Temperature".format(
        name, a, b, r), fontsize=14)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "{}_pair_{}_{}.png".format(name, a, b))
    fig.savefig(path, dpi=105)
    plt.close(fig)

    print("  {} - {}: r={:+.3f} (停止除外 {:+.3f})  | OTとの相関 {}={:+.3f} {}={:+.3f}".format(
        a, b, r, r_live, a, d["OT"].corr(d[a]), b, d["OT"].corr(d[b])))
    return path


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = []
    for name, pairs in PAIRS.items():
        print("=" * 66)
        print(name, " (2年全体, n={})".format(len(load(name))))
        d = load(name)
        for a, b in pairs:
            paths.append(plot_pair(d, name, a, b))
    print("\n保存:")
    for p in paths:
        print("   ", p)


if __name__ == "__main__":
    main()
