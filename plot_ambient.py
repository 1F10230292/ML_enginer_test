# -*- coding: utf-8 -*-
"""
OT と中国の実測外気温の関係を1年分の時系列で可視化する。

外気温データ出典:
  Open-Meteo Historical Weather API (ERA5 reanalysis)
  https://open-meteo.com/  — 無償・非営利利用可 (CC BY 4.0)

ETTデータセットには外気温が含まれないため外部データを補う。
変電所の正確な所在地は非公開のため、華北の複数都市を候補として取得し、
ETTh1の設備停止期間(2016-12-05〜12-07、無負荷で油温が外気温まで低下)を
アンカーにして整合性を確認する。
"""
import json
import os
import urllib.parse
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR, OUT_DIR = "figures", "outputs"
START, END = "2016-07-01", "2017-06-30"

CITIES = {
    "Beijing": (39.9042, 116.4074),
    "Shijiazhuang": (38.0428, 114.5149),
    "Taiyuan": (37.8706, 112.5489),
    "Jinan": (36.6512, 117.1201),
    "Zhengzhou": (34.7466, 113.6254),
    "Shenyang": (41.8057, 123.4315),
}


def fetch_temp(city, lat, lon):
    """Open-Meteo から時別気温を取得(ローカルにキャッシュ)。"""
    cache = os.path.join(OUT_DIR, "ambient_{}.csv".format(city))
    if os.path.exists(cache):
        d = pd.read_csv(cache)
        d["date"] = pd.to_datetime(d["date"])
        return d

    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "hourly": "temperature_2m", "timezone": "Asia/Shanghai",
    })
    url = "https://archive-api.open-meteo.com/v1/archive?" + q
    with urllib.request.urlopen(url, timeout=90) as r:
        j = json.load(r)
    d = pd.DataFrame({
        "date": pd.to_datetime(j["hourly"]["time"]),
        "temp": j["hourly"]["temperature_2m"],
    })
    os.makedirs(OUT_DIR, exist_ok=True)
    d.to_csv(cache, index=False)
    return d


def load_ett(name):
    d = pd.read_csv("./data/{}.csv".format(name), sep=",", header=0)
    d["date"] = pd.to_datetime(d["date"])
    # 毎月31日は全項目が定数の合成値 → 除外
    d = d[d["date"].dt.day != 31]
    return d[(d["date"] >= START) & (d["date"] <= END + " 23:00")].reset_index(drop=True)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    temps = {}
    for city, (lat, lon) in CITIES.items():
        temps[city] = fetch_temp(city, lat, lon)
        print("取得: {:14s} {}件".format(city, len(temps[city])))

    ett = {n: load_ett(n) for n in ["ETTh1", "ETTh2"]}

    # --- 停電期間をアンカーにした照合 ---
    print("\n[アンカー] ETTh1 停止期間 2016-12-06〜12-07 の油温(=外気温に漸近)と各都市の実測気温")
    anchor = ett["ETTh1"]
    seg = anchor[(anchor["date"] >= "2016-12-06 12:00") & (anchor["date"] <= "2016-12-07 12:00")]
    print("   ETTh1 油温(無負荷時): 平均 {:+.2f}℃  最低 {:+.2f}℃".format(
        seg["OT"].mean(), seg["OT"].min()))
    for city, t in temps.items():
        s = t[(t["date"] >= "2016-12-06 12:00") & (t["date"] <= "2016-12-07 12:00")]
        print("   {:14s} 平均 {:+.2f}℃  最低 {:+.2f}℃   差(平均) {:+.2f}℃".format(
            city, s["temp"].mean(), s["temp"].min(), s["temp"].mean() - seg["OT"].mean()))

    # --- 相関 ---
    print("\n[相関] OT と各都市の外気温 (1年, 生データ / 7日移動平均)")
    rows = []
    for n, d in ett.items():
        for city, t in temps.items():
            m = d.merge(t, on="date", how="inner")
            r_raw = m["OT"].corr(m["temp"])
            r_smooth = m["OT"].rolling(168).mean().corr(m["temp"].rolling(168).mean())
            rows.append({"station": n, "city": city, "r_raw": r_raw, "r_7d": r_smooth})
    rdf = pd.DataFrame(rows)
    print(rdf.pivot(index="city", columns="station", values=["r_raw", "r_7d"]).round(3).to_string())
    rdf.to_csv(os.path.join(OUT_DIR, "ambient_correlation.csv"), index=False)

    # --- 作図: 北京を代表として1年分 ---
    ref = "Beijing"
    t = temps[ref]
    paths = []
    for n, d in ett.items():
        m = d.merge(t, on="date", how="inner")

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        ax1 = axes[0]
        ax1.plot(m["date"], m["OT"], color="red", alpha=0.30, linewidth=0.6, label="OT (hourly)")
        ax1.plot(m["date"], m["OT"].rolling(168).mean(), color="red", linewidth=2.2,
                 label="OT (7d rolling mean)")
        ax1.set_ylabel("Oil Temperature [degC]", color="red")
        ax1.tick_params(axis="y", labelcolor="red")

        ax1b = ax1.twinx()
        ax1b.plot(m["date"], m["temp"], color="#1f77b4", alpha=0.30, linewidth=0.6,
                  label="Ambient (hourly)")
        ax1b.plot(m["date"], m["temp"].rolling(168).mean(), color="#1f77b4", linewidth=2.2,
                  label="Ambient (7d rolling mean)")
        ax1b.set_ylabel("Ambient air temperature [degC] ({})".format(ref), color="#1f77b4")
        ax1b.tick_params(axis="y", labelcolor="#1f77b4")

        r = m["OT"].corr(m["temp"])
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax1b.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=9, ncol=2)
        ax1.set_title("{}: Oil Temperature vs Ambient Air Temperature ({})   r = {:+.3f}".format(
            n, ref, r), fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 下段: 温度上昇分 (OT - 外気温)
        ax2 = axes[1]
        rise = m["OT"] - m["temp"]
        ax2.plot(m["date"], rise, color="gray", alpha=0.35, linewidth=0.6,
                 label="OT - Ambient (hourly)")
        ax2.plot(m["date"], rise.rolling(168).mean(), color="darkgreen", linewidth=2.2,
                 label="OT - Ambient (7d rolling mean)")
        ax2.axhline(0, color="black", linewidth=0.9)
        ax2.axhline(rise.mean(), color="orange", linestyle="--", linewidth=1.3,
                    label="mean = {:+.1f} degC".format(rise.mean()))
        ax2.set_ylabel("Temperature rise over ambient [degC]")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.set_title("Temperature rise above ambient (= heat generated by the transformer itself)",
                      fontsize=12)
        ax2.grid(True, alpha=0.3)

        fig.suptitle("{}  2016-07-01 - 2017-06-30   |   ambient: Open-Meteo ERA5, {}".format(
            n, ref), fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = os.path.join(FIG_DIR, "{}_OT_vs_ambient.png".format(n))
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)

        print("\n[{}] 外気温に対する上昇分 (OT - Ambient)".format(n))
        print("   平均 {:+.1f}℃ / 最小 {:+.1f}℃ / 最大 {:+.1f}℃".format(
            rise.mean(), rise.min(), rise.max()))
        mm = m.assign(rise=rise).groupby(m["date"].dt.month)["rise"].mean()
        print("   月別平均: " + "  ".join("{}月={:+.1f}".format(k, v) for k, v in mm.items()))

    print("\n保存:")
    for p in paths:
        print("   ", p)


if __name__ == "__main__":
    main()
