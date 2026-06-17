import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
from pathlib import Path

data = {
    "模型": [
        "Qwen3.5-Omni-Plus",
        "Qwen3-Omni-30B-A3B-Instruct",
        "Fun-Audio-Chat-8B",
        "GPT-Audio-Mini",
        "GPT-Audio",
    ],
    "内容主题与意图理解": [87.4, 90.2, 84.6, 84.9, 77.5],
    "事件场景与说话人信息识别": [73.7, 60.0, 45.0, 28.4, 30.3],
    "语用含义与语言现象推理": [90.8, 90.2, 89.2, 85.9, 89.0],
    "情绪态度识别": [48.1, 37.0, 37.5, 10.0, 4.8],
    "韵律声学感知与推理": [67.9, 42.2, 26.8, 25.0, 26.2],
}

df = pd.DataFrame(data)

# macOS 中文黑体字体路径
# 优先使用华文黑体；如果没有，再回退到苹方或 Noto
font_path_candidates = [
    "/System/Library/Fonts/STHeiti Medium.ttc",      # 华文黑体，中黑
    "/System/Library/Fonts/STHeiti Light.ttc",       # 华文黑体，细黑
    "/System/Library/Fonts/PingFang.ttc",            # 苹方，macOS 常见中文黑体风格
    "/Library/Fonts/NotoSansCJK-Regular.ttc",        # Noto Sans CJK
    "/Library/Fonts/Noto Sans CJK SC-Regular.otf",
]

font_path = next((p for p in font_path_candidates if Path(p).exists()), None)

if font_path is None:
    raise FileNotFoundError(
        "没有找到可用中文黑体字体。请检查 macOS 字体路径，或安装 Noto Sans CJK SC。"
    )

# 显式注册并使用字体
fm.fontManager.addfont(font_path)

font_prop = fm.FontProperties(fname=font_path)
bold_font_prop = fm.FontProperties(

    fname=font_path,

    weight="bold",

    size=24

)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

print("使用字体文件：", font_path)
print("字体名称：", font_prop.get_name())


def wrap_model_name(name):
    mapping = {
        "Qwen3.5-Omni-Plus": "Qwen3.5-Omni\n-Plus",
        "Qwen3-Omni-30B-A3B-Instruct": "Qwen3-Omni\n-30B-A3B-Instruct",
        "Fun-Audio-Chat-8B": "Fun-Audio\n-Chat-8B",
        "GPT-Audio-Mini": "GPT-Audio\n-Mini",
        "GPT-Audio": "GPT-Audio",
    }

    return mapping.get(name, name)


def add_outer_labels(ax, angles, labels, radius=1.25, fontsize=10):
    ax.set_xticklabels([])

    for angle, label in zip(angles, labels):
        x = 0.5 + 0.5 * radius * np.cos(np.pi / 2 - angle)
        y = 0.5 + 0.5 * radius * np.sin(np.pi / 2 - angle)

        if x > 0.56:
            ha = "left"
        elif x < 0.44:
            ha = "right"
        else:
            ha = "center"

        if y > 0.56:
            va = "bottom"
        elif y < 0.44:
            va = "top"
        else:
            va = "center"

        ax.text(
            x,
            y,
            wrap_model_name(label),
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontproperties=font_prop,
            clip_on=False,
        )


models = df["模型"].tolist()

dimensions = [
    "内容主题与意图理解",
    "事件场景与说话人信息识别",
    "语用含义与语言现象推理",
    "情绪态度识别",
    "韵律声学感知与推理",
]

n = len(models)

angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
closed_angles = angles + angles[:1]

fig = plt.figure(figsize=(12.8, 9.4), dpi=180)
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles)
add_outer_labels(ax, angles, models, radius=1.25, fontsize=10)

ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(
    ["20", "40", "60", "80", "100"],
    fontsize=9,
    fontproperties=font_prop,
)

ax.set_rlabel_position(90)
ax.grid(True, linewidth=0.8, alpha=0.45)

for dim in dimensions:
    values = df[dim].tolist() + df[dim].tolist()[:1]

    ax.plot(
        closed_angles,
        values,
        linewidth=2,
        marker="o",
        markersize=4,
        label=dim,
    )

    ax.fill(closed_angles, values, alpha=0.055)

# 雷达图上方标题：加大、加粗
ax.set_title(
    "各模型在不同听力任务上的正确率",
    fontsize=32,
    fontproperties=bold_font_prop,
    pad=92,
)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.30),
    ncol=2,
    frameon=False,
    prop=font_prop,
)

fig.subplots_adjust(left=0.10, right=0.90, top=0.72, bottom=0.25)

output_path = "listening_task_accuracy_radar_title_32_bold.png"

fig.savefig(output_path, bbox_inches="tight")
plt.show()

print(f"已保存：{output_path}")
print(f"字体文件：{font_path}")
print("标题已设置为 32 号、加粗：不同模型在各听力任务上的正确率")