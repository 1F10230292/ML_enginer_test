# -*- coding: utf-8 -*-
"""
HUFL と OT を四季それぞれ3週間ずつ切り出し、土日・中国の祝日を重ねて可視化する。

ねらい:
  - 季節ごとの負荷と油温のトレンド比較
  - 春節/国慶節など長期休暇での負荷低下の有無 → 需要家構成(工業 or 生活)の検証

祝日は中国の法定節假日。振替出勤日(调休)も別色で表示する。
ラベルは文字化け回避のため英語表記。
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG_DIR = "figures"

# 3週間ずつの窓 (季節名, 表示名, 開始, 終了)
SEASONS = [
    ("spring", "Spring", "2017-04-17", "2017-05-08"),
    ("summer", "Summer", "2016-07-25", "2016-08-15"),
    ("autumn", "Autumn", "2016-09-14", "2016-10-05"),
    ("winter", "Winter", "2017-01-16", "2017-02-06"),
]

# 中国の法定節假日 (期間中に該当するもの)
HOLIDAYS = [
    ("2016-09-15", "2016-09-17", "Mid-Autumn Festival"),
    ("2016-10-01", "2016-10-07", "National Day"),
    ("2017-01-27", "2017-02-02", "Spring Festival (CNY)"),
    ("2017-04-29", "2017-05-01", "Labour Day"),
]

# 振替出勤日(调休): 土日だが出勤日
MAKEUP_WORKDAYS = ["2016-09-18", "2016-10-08", "2016-10-09", "2017-01-22", "2017-02-04"]


def load(name):
    d = pd.read_csv("./data/{}.csv".format(name), sep=",", header=0)
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


def day_type(ts):
    """各日を workday / weekend / holiday / makeup に分類する。"""
    day = ts.normalize()
    for s, e, _ in HOLIDAYS:
        if pd.Timestamp(s) <= day <= pd.Timestamp(e):
            return "holiday"
    if day.strftime("%Y-%m-%d") in MAKEUP_WORKDAYS:
        return "makeup"
    return "weekend" if day.dayofweek >= 5 else "workday"


def plot_station(name):
    d = load(name)
    fig, axes = plt.subplots(4, 1, figsize=(17, 16))

    stats = []
    for ax1, (key, label, start, end) in zip(axes, SEASONS):
        w = d[(d["date"] >= start) & (d["date"] < end)].copy()
        w["dtype"] = w["date"].map(day_type)

        # 土日・祝日の背景
        for day in pd.date_range(start, end, freq="D")[:-1]:
            t = day_type(day)
            if t == "weekend":
                ax1.axvspan(day, day + pd.Timedelta(days=1), color="gray", alpha=0.15, zorder=0)
            elif t == "holiday":
                ax1.axvspan(day, day + pd.Timedelta(days=1), color="orange", alpha=0.22, zorder=0)
            elif t == "makeup":
                ax1.axvspan(day, day + pd.Timedelta(days=1), color="green", alpha=0.14, zorder=0)

        # 祝日名を注記
        for s, e, hname in HOLIDAYS:
            s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1)
            if s_ts < pd.Timestamp(end) and e_ts > pd.Timestamp(start):
                mid = max(s_ts, pd.Timestamp(start)) + (
                    min(e_ts, pd.Timestamp(end)) - max(s_ts, pd.Timestamp(start))) / 2
                ax1.annotate("{}\n{} - {}".format(hname, s, e),
                             xy=(mid, 0.97), xycoords=("data", "axes fraction"),
                             ha="center", va="top", fontsize=9, color="darkorange",
                             fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="orange", alpha=0.85))

        ax1.plot(w["date"], w["OT"], color="red", linewidth=1.9, label="OT", zorder=3)
        ax1.set_ylabel("OT [degC]", color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()
        ax2.plot(w["date"], w["HUFL"], color="#1f77b4", linewidth=1.2, linestyle="-",
                 alpha=0.85, label="HUFL (Active P)", zorder=2)
        ax2.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax2.set_ylabel("HUFL", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")

        # 日付+曜日の目盛
        days = pd.date_range(start, end, freq="D")[:-1]
        ax1.set_xticks(days)
        ax1.set_xticklabels(["{}\n{}".format(dd.strftime("%m/%d"), dd.strftime("%a")) for dd in days],
                            fontsize=8)
        ax1.set_xlim(pd.Timestamp(start), pd.Timestamp(end))

        # 区分別の平均
        g = w.groupby("dtype")[["HUFL", "OT"]].mean()
        parts = []
        for t in ["workday", "weekend", "holiday", "makeup"]:
            if t in g.index:
                parts.append("{}: HUFL={:.2f} OT={:.1f}".format(t, g.loc[t, "HUFL"], g.loc[t, "OT"]))
                stats.append({"season": label, "type": t,
                              "HUFL": g.loc[t, "HUFL"], "OT": g.loc[t, "OT"]})
        ax1.set_title("{}  {} - {}   |   {}".format(label, start, end, "   ".join(parts)),
                      fontsize=10, loc="left")

        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "{}: HUFL (Active Power) vs Oil Temperature — 3 weeks per season\n"
        "gray = weekend,  orange = Chinese public holiday,  green = makeup workday (tiaoxiu)".format(name),
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    path = os.path.join(FIG_DIR, "{}_seasonal_holiday.png".format(name))
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path, pd.DataFrame(stats)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for name in ["ETTh1", "ETTh2"]:
        path, st = plot_station(name)
        print("=" * 72)
        print("■ {}  -> {}".format(name, path))
        piv = st.pivot(index="season", columns="type", values="HUFL")
        piv_ot = st.pivot(index="season", columns="type", values="OT")
        print("  平均 HUFL (区分別)")
        print(piv.round(2).to_string())
        print("  平均 OT (区分別)")
        print(piv_ot.round(1).to_string())
        if "holiday" in piv.columns and "workday" in piv.columns:
            r = (piv["holiday"] / piv["workday"]).dropna()
            print("  祝日/平日のHUFL比: " + "  ".join(
                "{}={:.3f}".format(k, v) for k, v in r.items()))


if __name__ == "__main__":
    main()
