"""matplotlib 绘图统一样式：中文字体与负号显示。

Windows 自带微软雅黑；macOS/Linux 环境自动回退到可用字体。
"""

from __future__ import annotations

import matplotlib


def apply_plot_style() -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
