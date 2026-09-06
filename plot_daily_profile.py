# -*- coding: utf-8 -*-
"""
四季それぞれ任意の1日について、OT・有効電力(P)・無効電力(Q)の24時間推移を可視化する。

目的: 「昼に多く電力を流し、夜に減らす」という一般的な想定が成立するかの検証。
単日だけでは偶然の可能性があるため、その季節の平均日内プロファイルを薄線で重ねる。

ラベルは文字化け回避のため英語表記。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = "figures"
P_COL, Q_COL = "HUFL", "HULL"

# (季節名, 代表日, その季節とみなす月)  代表日はすべて水曜・非祝日・31日以外
SEASONS = [
    ("Spring", "2017-04-19", [3, 4, 5]),
    ("Summer", "2016-08-10", [6, 7, 8]),
    ("Autumn", "2016-10-19", [9, 10, 11]),
    ("Winter", "2017-01-18", [12, 1, 2]),
]


def load(name):
    d = pd.read_csv("./data/{}.csv".format(name), sep=",", header=0)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"].dt.day != 31]          # 月末31日は定数埋めの合成値
    return d.reset_index(drop=True)


def plot_station(name):
    d = load(name)
    d["hour"] = d["date"].dt.hour
    d["month"] = d["date"].dt.month
    # 1年分を季節平均の母集団にする
    year = d[(d["date"] >= "2016-07-01") & (d["date"] <= "2017-06-30 23:00")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    summary = []

    for ax1, (season, day, months) in zip(axes.ravel(), SEASONS):
        one = d[d["date"].dt.normalize() == pd.Timestamp(day)].sort_values("hour")
        prof = year[year["month"].isin(months)].groupby("hour")[[P_COL, Q_COL, "OT"]].mean()

        h = one["hour"].values

        # OT (左軸)
        ax1.plot(h, one["OT"], color="red", linewidth=2.4, marker="o", markersize=3,
                 label="OT ({})".format(day), zorder=4)
        ax1.plot(prof.index, prof["OT"], color="red", linewidth=1.2, alpha=0.35,
                 linestyle=":", label="OT (season mean)")
        ax1.set_ylabel("OT [degC]", color="red")
        ax1.tick_params(axis="y", labelcolor="red")
        ax1.set_xlabel("Hour of day")
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xlim(0, 23)

        # P, Q (右軸)
        ax2 = ax1.twinx()
        ax2.plot(h, one[P_COL], color="#1f77b4", linewidth=2.0, marker="s", markersize=3,
                 label="{} = Active P".format(P_COL), zorder=3)
        ax2.plot(prof.index, prof[P_COL], color="#1f77b4", linewidth=1.2, alpha=0.35,
                 linestyle=":", label="P (season mean)")
        ax2.plot(h, one[Q_COL], color="#ff7f0e", linewidth=2.0, linestyle="--",
                 marker="^", markersize=3, label="{} = Reactive Q".format(Q_COL), zorder=3)
        ax2.plot(prof.index, prof[Q_COL], color="#ff7f0e", linewidth=1.2, alpha=0.35,
                 linestyle=":", label="Q (season mean)")
        ax2.axhline(0, color="black", linewidth=1.0, alpha=0.5)
        ax2.set_ylabel("Power (P, Q)", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")

        # 昼(9-17時)と夜(21-5時)の平均を比較
        dayp = one[(one["hour"] >= 9) & (one["hour"] <= 17)][P_COL].mean()
        nightp = one[(one["hour"] >= 21) | (one["hour"] <= 5)][P_COL].mean()
        pk, tr = int(one.loc[one[P_COL].idxmax(), "hour"]), int(one.loc[one[P_COL].idxmin(), "hour"])
        verdict = "DAY > NIGHT (as expected)" if dayp > nightp else "NIGHT > DAY (inverted!)"
        summary.append({"season": season, "day": day, "P_day": dayp, "P_night": nightp,
                        "peak_h": pk, "trough_h": tr, "verdict": verdict})

        ax1.axvspan(9, 17, color="gold", alpha=0.10, zorder=0)
        ax1.set_title("{}  {}   |  P day(9-17h)={:.2f}  night(21-5h)={:.2f}  ->  {}".format(
            season, day, dayp, nightp, verdict), fontsize=10, loc="left")

        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="best", fontsize=7.5, ncol=2)
        ax1.grid(True, alpha=0.25)

    fig.suptitle(
        "{}: One representative day per season — OT, Active power (P) and Reactive power (Q)\n"
        "solid = the chosen day,  dotted = seasonal mean profile,  shaded = daytime 9-17h".format(name),
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(FIG_DIR, "{}_daily_profile.png".format(name))
    fig.savefig(path, dpi=110)
    plt.close(fig)

    print("=" * 78)
    print("■ {}".format(name))
    print(pd.DataFrame(summary).to_string(index=False))

    # 季節平均でも同じ判定になるか(単日の偶然でないかの確認)
    print("  [季節平均での検証] 昼(9-17h) vs 夜(21-5h) の平均 P")
    for season, day, months in SEASONS:
        s = year[year["month"].isin(months)]
        dp = s[(s["hour"] >= 9) & (s["hour"] <= 17)][P_COL].mean()
        np_ = s[(s["hour"] >= 21) | (s["hour"] <= 5)][P_COL].mean()
        print("   {:7s} 昼={:7.2f}  夜={:7.2f}  比(昼/夜)={:.2f}  {}".format(
            season, dp, np_, dp / np_ if np_ else float("nan"),
            "昼>夜" if dp > np_ else "夜>昼 (逆転)"))
    return path


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for n in ["ETTh1", "ETTh2"]:
        print("\n->", plot_station(n))


if __name__ == "__main__":
    main()
