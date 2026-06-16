"""
Fresnel 反射与透射 — 振幅系数 · 功率系数 · 布儒斯特角 · 全内反射
原生 Streamlit 组件，单文件。
"""

import io
import base64
import os as _os
import urllib.request as _request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

# ---- 中文字体: 全平台统一 Noto Sans SC ----
_FONT_DIR = _os.path.join(_os.path.expanduser("~"), ".fonts")
_os.makedirs(_FONT_DIR, exist_ok=True)
_FONT_FILE = _os.path.join(_FONT_DIR, "NotoSansSC-Regular.otf")

if not _os.path.exists(_FONT_FILE):
    try:
        _url = ("https://raw.githubusercontent.com/notofonts/noto-cjk/main/"
                "Sans/SubsetOTF/SC/NotoSansSC-Regular.otf")
        _request.urlretrieve(_url, _FONT_FILE)
    except Exception:
        pass

if _os.path.exists(_FONT_FILE):
    fm.fontManager.addfont(_FONT_FILE)
    plt.rcParams["font.family"] = "Noto Sans SC"
else:
    for _f in ["PingFang SC", "Heiti SC", "STHeiti",
               "Noto Sans CJK SC", "WenQuanYi Micro Hei",
               "Arial Unicode MS", "SimHei"]:
        if any(_f.lower() in name.lower() for name in fm.get_font_names()):
            plt.rcParams["font.family"] = _f
            break

plt.rcParams["axes.unicode_minus"] = False

# ---- 颜色常量 ----
C_S = "#2563eb"          # s 偏振 (TE)
C_P = "#0d9488"          # p 偏振 (TM)
C_BREWSTER = "#d97706"   # 布儒斯特角标记
C_CRITICAL = "#db2777"   # 临界角 / 全内反射
SLATE_300 = "#cbd5e1"
SLATE_400 = "#94a3b8"


# ============================================================
# 物理计算
# ============================================================

def brewster_angle(n1, n2):
    """布儒斯特角 [°] — p 偏振反射率为零"""
    return float(np.degrees(np.arctan(n2 / n1)))


def critical_angle(n1, n2):
    """临界角 [°] — 仅当 n1 > n2 时存在"""
    if n1 > n2:
        return float(np.degrees(np.arcsin(n2 / n1)))
    return None


def normal_reflectance(n1, n2):
    """法向入射反射率"""
    return ((n1 - n2) / (n1 + n2)) ** 2


def compute_fresnel(n1, n2, theta_deg):
    """
    向量化 Fresnel 计算。

    参数:
        n1, n2 : 折射率
        theta_deg : 入射角 [°], 标量或 array

    返回:
        R_s, R_p : 反射率
        T_s, T_p : 透射率
        theta_t  : 折射角 [°]
        is_tir   : 是否全内反射
        r_s, r_p : 复振幅反射系数 (TIR 时为复数, 含相位)
        t_s, t_p : 复振幅透射系数
    """
    theta_i = np.radians(np.asarray(theta_deg, dtype=float))
    sin_i = np.sin(theta_i)
    cos_i = np.cos(theta_i)

    # Snell 定律
    ratio = n1 / n2
    sin_t = ratio * sin_i
    is_tir = sin_t >= 1.0

    # cos(θ_t) : 正常折射为实数; TIR 时为纯虚数
    # np.where 会同时计算两个分支; 抑制其中一方的无效 sqrt 警告
    with np.errstate(invalid="ignore"):
        cos_t = np.where(is_tir, 1j * np.sqrt(sin_t ** 2 - 1),
                         np.sqrt(1 - sin_t ** 2))

    # ---- 振幅系数 ----
    denom_s = n1 * cos_i + n2 * cos_t
    denom_p = n2 * cos_i + n1 * cos_t

    # 处理可能的分母为零（仅发生在 n1==n2 且 cos_t==cos_i，此情况下无反射）
    safe_s = np.where(denom_s == 0, 1.0, denom_s)
    safe_p = np.where(denom_p == 0, 1.0, denom_p)

    r_s = np.where(denom_s == 0, 0.0, (n1 * cos_i - n2 * cos_t) / safe_s)
    r_p = np.where(denom_p == 0, 0.0, (n2 * cos_i - n1 * cos_t) / safe_p)
    t_s = np.where(denom_s == 0, 1.0, (2 * n1 * cos_i) / safe_s)
    t_p = np.where(denom_p == 0, 1.0, (2 * n1 * cos_i) / safe_p)

    # ---- 功率系数 ----
    R_s = np.abs(r_s) ** 2
    R_p = np.abs(r_p) ** 2

    T_s = np.where(is_tir, 0.0,
                   (n2 * np.real(cos_t)) / (n1 * cos_i) * np.abs(t_s) ** 2)
    T_p = np.where(is_tir, 0.0,
                   (n2 * np.real(cos_t)) / (n1 * cos_i) * np.abs(t_p) ** 2)

    # 折射角 : TIR 时记为 90°
    with np.errstate(invalid="ignore"):
        theta_t = np.where(is_tir, 90.0, np.degrees(np.arcsin(sin_t)))

    return R_s, R_p, T_s, T_p, theta_t, is_tir, r_s, r_p, t_s, t_p


# ============================================================
# 绑图工具
# ============================================================

def fmt_complex(z):
    """把复振幅系数格式化为可读字符串; 传播区(虚部≈0)只显示实数"""
    z = complex(z)
    if abs(z.imag) < 1e-4:
        return f"{z.real:+.4f}"
    sign = "+" if z.imag >= 0 else "−"
    return f"{z.real:+.4f} {sign} {abs(z.imag):.4f}i"


def svg_block(fig, width_px=800):
    """将 matplotlib Figure 编码为 base64 SVG 嵌入 Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    svg_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return (f'<img src="data:image/svg+xml;base64,{svg_data}" '
            f'style="width:100%;max-width:{width_px}px"/>')


# ============================================================
# 绑图: 反射率 / 透射率
# ============================================================

def plot_reflection(angles, R_s, R_p, theta_b, theta_c, n1, n2):
    """反射率 vs 入射角"""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    ax.plot(angles, R_s, color=C_S, linewidth=2.0, label="R_s  (s / TE)")
    ax.plot(angles, R_p, color=C_P, linewidth=2.0, label="R_p  (p / TM)")
    ax.set_xlabel("入射角 [°]", fontsize=11)
    ax.set_ylabel("反射率 R", fontsize=11)
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.04, 1.04)   # 留少量上下留白, 0/1 不压在框线上
    ax.margins(x=0)
    ax.grid(True, alpha=0.3, color=SLATE_400)

    # 关键角: 仅画竖虚线, 角度值写进图例(图注), 不再用箭头标注
    if theta_b is not None:
        ax.axvline(theta_b, color=C_BREWSTER, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_B = {theta_b:.1f}°  布儒斯特角")
        ax.scatter([theta_b], [0], c=C_BREWSTER, s=45, zorder=5,
                   marker="o", edgecolors="white", linewidths=1.2)

    if theta_c is not None:
        ax.axvline(theta_c, color=C_CRITICAL, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_C = {theta_c:.1f}°  临界角")
        ax.axvspan(theta_c, 90, alpha=0.06, color=C_CRITICAL)

    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
    plt.tight_layout()
    return fig


def plot_transmission(angles, T_s, T_p, theta_b, theta_c, n1, n2):
    """透射率 vs 入射角"""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    ax.plot(angles, T_s, color=C_S, linewidth=2.0, label="T_s  (s / TE)")
    ax.plot(angles, T_p, color=C_P, linewidth=2.0, label="T_p  (p / TM)")
    ax.set_xlabel("入射角 [°]", fontsize=11)
    ax.set_ylabel("透射率 T", fontsize=11)
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.04, 1.04)   # 留少量上下留白, 0/1 不压在框线上
    ax.margins(x=0)
    ax.grid(True, alpha=0.3, color=SLATE_400)

    if theta_b is not None:
        ax.axvline(theta_b, color=C_BREWSTER, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_B = {theta_b:.1f}°  布儒斯特角")
    if theta_c is not None:
        ax.axvline(theta_c, color=C_CRITICAL, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_C = {theta_c:.1f}°  临界角")
        ax.axvspan(theta_c, 90, alpha=0.06, color=C_CRITICAL)

    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.85)
    plt.tight_layout()
    return fig


def plot_amplitude(angles, r_s, r_p, theta_b, theta_c, is_tir, n1, n2):
    """带符号的振幅反射系数 r vs 入射角 (符号即相位: 负号 = π 相变)"""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    # TIR 区 r 为复数(|r|=1), 实部会振荡, 故只画传播区的带符号实值
    rs = np.where(is_tir, np.nan, np.real(r_s))
    rp = np.where(is_tir, np.nan, np.real(r_p))

    ax.axhline(0, color=SLATE_400, linewidth=1.0, alpha=0.7)
    ax.plot(angles, rs, color=C_S, linewidth=2.0, label="r_s  (s / TE)")
    ax.plot(angles, rp, color=C_P, linewidth=2.0, label="r_p  (p / TM)")
    ax.set_xlabel("入射角 [°]", fontsize=11)
    ax.set_ylabel("振幅反射系数 r", fontsize=11)
    ax.set_xlim(0, 90)
    ax.set_ylim(-1.06, 1.06)   # 留少量留白, ±1 不压在框线上
    ax.margins(x=0)
    ax.grid(True, alpha=0.3, color=SLATE_400)

    if theta_b is not None:
        ax.axvline(theta_b, color=C_BREWSTER, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_B = {theta_b:.1f}°  (r_p 变号)")
        ax.scatter([theta_b], [0], c=C_BREWSTER, s=45, zorder=5,
                   marker="o", edgecolors="white", linewidths=1.2)
    if theta_c is not None:
        ax.axvline(theta_c, color=C_CRITICAL, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_C = {theta_c:.1f}°  (|r|=1, 起相位)")
        ax.axvspan(theta_c, 90, alpha=0.06, color=C_CRITICAL)

    ax.legend(loc="best", fontsize=8.5, framealpha=0.85)
    plt.tight_layout()
    return fig


def plot_t_amplitude(angles, t_s, t_p, theta_c, is_tir, n1, n2):
    """振幅透射系数 t vs 入射角 (恒正, 无相位翻转; 内反射时可 > 1)"""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    ts = np.where(is_tir, np.nan, np.real(t_s))
    tp = np.where(is_tir, np.nan, np.real(t_p))

    ax.axhline(0, color=SLATE_400, linewidth=1.0, alpha=0.7)
    ax.plot(angles, ts, color=C_S, linewidth=2.0, label="t_s  (s / TE)")
    ax.plot(angles, tp, color=C_P, linewidth=2.0, label="t_p  (p / TM)")
    ax.set_xlabel("入射角 [°]", fontsize=11)
    ax.set_ylabel("振幅透射系数 t", fontsize=11)
    ax.set_xlim(0, 90)
    ax.set_ylim(bottom=-0.04)  # 留少量留白; t≥0, 内反射时可 >1, 上界自适应
    ax.margins(x=0)
    ax.grid(True, alpha=0.3, color=SLATE_400)

    if theta_c is not None:
        ax.axvline(theta_c, color=C_CRITICAL, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_C = {theta_c:.1f}°  (起全内反射)")
        ax.axvspan(theta_c, 90, alpha=0.06, color=C_CRITICAL)

    ax.legend(loc="best", fontsize=8.5, framealpha=0.85)
    plt.tight_layout()
    return fig


def plot_phase(angles, r_s, r_p, theta_b, theta_c, n1, n2):
    """反射相位 φ = arg(r) vs 入射角。
    传播区: φ 只能取 0 或 ±180° (实数 r 的符号); 全反射区: φ 连续变化。"""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ps = np.degrees(np.angle(r_s))
    pp = np.degrees(np.angle(r_p))

    ax.axhline(0, color=SLATE_400, linewidth=1.0, alpha=0.6)
    ax.plot(angles, ps, color=C_S, linewidth=2.0, label="φ_s  (s / TE)")
    ax.plot(angles, pp, color=C_P, linewidth=2.0, label="φ_p  (p / TM)")
    ax.set_xlabel("入射角 [°]", fontsize=11)
    ax.set_ylabel("反射相位 φ [°]", fontsize=11)
    ax.set_xlim(0, 90)
    ax.set_ylim(-195, 195)
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.margins(x=0)
    ax.grid(True, alpha=0.3, color=SLATE_400)

    if theta_b is not None:
        ax.axvline(theta_b, color=C_BREWSTER, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_B = {theta_b:.1f}°")
    if theta_c is not None:
        ax.axvline(theta_c, color=C_CRITICAL, linestyle="--", linewidth=1.4,
                   alpha=0.7, label=f"θ_C = {theta_c:.1f}°")
        ax.axvspan(theta_c, 90, alpha=0.06, color=C_CRITICAL)

    ax.legend(loc="best", fontsize=8.5, framealpha=0.85)
    plt.tight_layout()
    return fig


# ============================================================
# 绑图: 光线示意图
# ============================================================

def _draw_evanescent(ax, delta_lam, s_vis):
    """TIR: 画倏逝波(指数衰减场)、1/e 穿透深度 δ 与古斯-汉欣位移 D"""
    # 视觉穿透深度: 按实际 δ/λ 缩放限幅(示意, 非严格比例)
    d_vis = float(np.clip((delta_lam or 0.2) * 1.6, 0.4, 1.2))
    A = 0.95                      # 界面处(z=0)的最大振幅(横向长度表示场强)

    # 倏逝场: 入射点(法线 x=0)正下方振幅最大, 沿 +z(向下)指数衰减 exp(z/δ);
    #         沿 +x 是无衰减行波 → 用横向"梳齿"长度表示各深度的场强
    z = np.linspace(0, -1.5, 200)
    env = A * np.exp(z / d_vis)
    ax.fill_betweenx(z, 0.0, env, color=C_P, alpha=0.16, zorder=1)
    ax.plot(env, z, color=C_P, linewidth=1.3, alpha=0.85, zorder=2)
    for zi in np.linspace(-0.09, -1.42, 11):
        ax.plot([0.0, A * np.exp(zi / d_vis)], [zi, zi],
                color=C_P, linewidth=1.3, alpha=0.5, zorder=2)
    ax.annotate("", xy=(1.5, -0.05), xytext=(A, -0.05),   # 沿界面 +x 行波
                arrowprops=dict(arrowstyle="-|>", color=C_P, lw=1.1,
                                linestyle="dotted", mutation_scale=11))
    ax.text(1.18, -1.2, "倏逝波", fontsize=9, color=C_P, ha="center",
            fontweight="bold")

    # 1/e 穿透深度 δ: 法线左侧竖向标尺 + 指到场衰减到 1/e 处的虚线
    xm = -0.2
    ax.annotate("", xy=(xm, -d_vis), xytext=(xm, 0),
                arrowprops=dict(arrowstyle="<->", color=C_CRITICAL, lw=1.3))
    ax.plot([xm, A / np.e], [-d_vis, -d_vis],
            color=C_CRITICAL, linewidth=0.8, linestyle=":", alpha=0.7)
    lbl = f"δ = {delta_lam:.2f}λ" if delta_lam is not None else "δ"
    ax.text(xm - 0.1, -d_vis / 2, lbl, fontsize=8.5, color=C_CRITICAL,
            ha="right", va="center", fontweight="bold")

    # 古斯-汉欣横向位移 D: 界面上入射点 0 → 反射出射点 s_vis
    if s_vis > 0.05:
        ax.annotate("", xy=(s_vis, 0.16), xytext=(0, 0.16),
                    arrowprops=dict(arrowstyle="<|-|>", color=C_S, lw=1.4,
                                    mutation_scale=11))
        ax.text(s_vis / 2, 0.27, "D", fontsize=9, color=C_S,
                ha="center", fontweight="bold")


def plot_ray_diagram(theta_i_deg, theta_t_deg, is_tir, n1, n2,
                     delta_lam=None, gh_lam=None):
    """光线示意图; TIR 时叠加倏逝波穿透深度与古斯-汉欣位移"""
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    th_i = np.radians(theta_i_deg)
    th_t = np.radians(theta_t_deg) if not is_tir else np.pi / 2
    L = 1.6

    # TIR 时反射光出射点相对入射点横移 (古斯-汉欣位移); 钳到小幅度, 防止箭头出框(示意)
    s_vis = float(np.clip((gh_lam or 0.0) * 0.3, 0.15, 0.4)) if is_tir else 0.0

    # 界面与法线
    ax.axhline(0, color="k", linewidth=1.8, zorder=3)
    ax.axvline(0, color=SLATE_300, linestyle="--", linewidth=1.2,
               alpha=0.8, zorder=1)
    ax.text(1.75, 0.38, f"n1 = {n1}", fontsize=11, ha="center",
            fontweight="bold")
    ax.text(1.75, -0.4, f"n2 = {n2}", fontsize=11, ha="center",
            fontweight="bold")
    ax.text(-2.0, -0.14, "界面", fontsize=9, color=SLATE_400,
            ha="left", va="top")
    # 入射光 (打到原点)
    xi, yi = -L * np.sin(th_i), L * np.cos(th_i)
    ax.annotate("", xy=(0, 0), xytext=(xi, yi),
                arrowprops=dict(arrowstyle="-|>", color=C_BREWSTER,
                                lw=1.6, alpha=0.9, mutation_scale=16))
    ax.text(xi * 1.08 - 0.08, yi * 1.08, "入射光", fontsize=9,
            color=C_BREWSTER, ha="right", fontweight="bold")
    # 反射光 (TIR 时从横移 s_vis 处出射)
    xr, yr = s_vis + L * np.sin(th_i), L * np.cos(th_i)
    ax.annotate("", xy=(xr, yr), xytext=(s_vis, 0),
                arrowprops=dict(arrowstyle="-|>", color=C_S, lw=1.6,
                                alpha=0.9, mutation_scale=16))
    ax.text(xr * 1.04 + 0.05, yr * 1.08, "反射光", fontsize=9,
            color=C_S, fontweight="bold")
    # 折射光 / 倏逝波
    if is_tir:
        # 标出入射点(几何反射点, 在法线上)与反射光出射点(右移 D)
        ax.scatter([0], [0], c=C_BREWSTER, s=24, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.scatter([s_vis], [0], c=C_S, s=24, zorder=4,
                   edgecolors="white", linewidths=0.8)
        _draw_evanescent(ax, delta_lam, s_vis)
    else:
        xt, yt = L * np.sin(th_t), -L * np.cos(th_t)
        ax.annotate("", xy=(xt, yt), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=C_P,
                                    lw=1.6, alpha=0.9, mutation_scale=16))
        ax.text(xt * 1.08 + 0.05, yt * 1.08 - 0.05, "折射光",
                fontsize=9, color=C_P, fontweight="bold")
    # 角度弧线 (相对法线测量, 标注置于弧的角平分线外侧); TIR 时省略以免遮挡倏逝波标注
    arc_r = 0.34
    if not is_tir and theta_i_deg > 1.5:
        arc = np.linspace(0, th_i, 40)
        ax.plot(arc_r * np.sin(arc), arc_r * np.cos(arc),
                color=SLATE_400, linewidth=1.0)
        ax.text((arc_r + 0.22) * np.sin(th_i / 2),
                (arc_r + 0.22) * np.cos(th_i / 2),
                f"{theta_i_deg:.0f}°", fontsize=9, color=SLATE_400,
                ha="center", va="center")
    if not is_tir and theta_t_deg > 1.5:
        arc = np.linspace(0, th_t, 40)
        ax.plot(arc_r * np.sin(arc), -arc_r * np.cos(arc),
                color=SLATE_400, linewidth=1.0)
        ax.text((arc_r + 0.22) * np.sin(th_t / 2),
                -(arc_r + 0.22) * np.cos(th_t / 2),
                f"{theta_t_deg:.0f}°", fontsize=9, color=SLATE_400,
                ha="center", va="center")
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


# ============================================================
# Streamlit UI
# ============================================================

@st.dialog("从麦克斯韦方程到倏逝波", width="large")
def show_theory():
    # ---------- 0. 坐标与几何约定 ----------
    st.caption(
        "- **界面**：取 $z=0$ 平面。介质 1（$n_1$，入射侧）位于 $z>0$，"
        "介质 2（$n_2$，透射侧）位于 $z<0$。\n"
        "- **法线**：沿 $z$ 轴，垂直于界面。\n"
        "- **入射面**：由入射波矢与法线张成的平面，取为 $x\\text{–}z$ 平面。\n"
        "- **$x$ 轴**：在界面内、且落在入射面内（即入射波矢沿界面的投影方向，"
        "倏逝波沿此方向传播）。\n"
        "- **$y$ 轴**：垂直于入射面（同样落在界面内）。\n"
        "- 由此分解偏振：**s 偏振 (TE)** 的 $\\mathbf E\\parallel y$（垂直入射面）；"
        "**p 偏振 (TM)** 的 $\\mathbf H\\parallel y$。两者的切向方向即 $x,y$。")

    # ---------- 1. 出发点 ----------
    st.markdown("### ● 出发点：麦克斯韦方程与平面波")
    st.markdown(
        "在无源、线性、各向同性介质中，麦克斯韦方程组为"
    )
    st.latex(r"\nabla\cdot\mathbf D=0,\quad \nabla\cdot\mathbf B=0,\quad "
             r"\nabla\times\mathbf E=-\frac{\partial\mathbf B}{\partial t},\quad "
             r"\nabla\times\mathbf H=\frac{\partial\mathbf D}{\partial t}")
    st.markdown(
        "取单色平面波 $\\mathbf E=\\mathbf E_0\\,e^{i(\\mathbf k\\cdot\\mathbf r-\\omega t)}$，"
        "代入波动方程得色散关系 $|\\mathbf k| = n\\,\\omega/c$。"
        "把旋度方程在界面上沿无穷小回路积分，得到**边界条件**：电场与磁场的**切向分量连续**。"
    )
    st.latex(r"\mathbf E_\parallel\ \text{连续},\qquad \mathbf H_\parallel\ \text{连续}")

    # ---------- 2. Snell ----------
    st.markdown("### ● 相位匹配 → Snell 定律")
    st.markdown(
        "界面上入射、反射、透射三波的相位必须处处同步，即沿界面的波矢分量相等 "
        "$k_{ix}=k_{rx}=k_{tx}$。由 $k_x=n\\frac{\\omega}{c}\\sin\\theta$ 立即得到"
    )
    st.latex(r"n_1\sin\theta_i = n_2\sin\theta_t")

    # ---------- 3. s 偏振 ----------
    st.markdown("### ● s 偏振 (TE)：$\\mathbf E\\perp$ 入射面")
    st.markdown(
        "设 $E$ 沿 $y$ 方向。切向电场连续给出 $E_i+E_r=E_t$；"
        "磁场切向分量 $H_x\\propto nE\\cos\\theta$ 连续给出"
    )
    st.latex(r"E_i+E_r=E_t,\qquad n_1\cos\theta_i\,(E_i-E_r)=n_2\cos\theta_t\,E_t")
    st.markdown("联立解出振幅系数")
    st.latex(r"r_s=\frac{E_r}{E_i}="
             r"\frac{n_1\cos\theta_i-n_2\cos\theta_t}{n_1\cos\theta_i+n_2\cos\theta_t},"
             r"\qquad "
             r"t_s=\frac{E_t}{E_i}=\frac{2n_1\cos\theta_i}{n_1\cos\theta_i+n_2\cos\theta_t}")

    # ---------- 4. p 偏振 ----------
    st.markdown("### ● p 偏振 (TM)：$\\mathbf H\\perp$ 入射面")
    st.markdown(
        "此时 $H$ 沿 $y$。$H$ 切向连续给 $H_i+H_r=H_t$（即 $n_1(E_i+E_r)=n_2E_t$），"
        "$E$ 切向连续给 $\\cos\\theta_i(E_i-E_r)=\\cos\\theta_t\\,E_t$。联立得"
    )
    st.latex(r"r_p=\frac{n_2\cos\theta_i-n_1\cos\theta_t}{n_2\cos\theta_i+n_1\cos\theta_t},"
             r"\qquad "
             r"t_p=\frac{2n_1\cos\theta_i}{n_2\cos\theta_i+n_1\cos\theta_t}")
    st.info("以上四式即 **Fresnel 方程**，全部由两条切向连续边界条件 + Snell 定律导出。")

    # ---------- 5. 功率系数 ----------
    st.markdown("### ● 功率系数与能量守恒")
    st.markdown(
        "坡印廷矢量法向分量 $S_z\\propto n\\cos\\theta\\,|E|^2$，反射/透射功率比为"
    )
    st.latex(r"R=|r|^2,\qquad T=\frac{n_2\cos\theta_t}{n_1\cos\theta_i}\,|t|^2,"
             r"\qquad R+T=1")
    st.markdown(
        "其中 $T$ 的前因子来自两介质中**光束截面与坡印廷流的不同**——"
        "这正是为何振幅系数 $t$ 可以大于 1，而功率仍守恒。法向入射极限：")
    st.latex(r"R(0^\circ)=\left(\frac{n_1-n_2}{n_1+n_2}\right)^2")

    # ---------- 6. 布儒斯特角 ----------
    st.markdown("### ● 布儒斯特角：$r_p=0$")
    st.markdown("令 $r_p$ 分子为零 $n_2\\cos\\theta_i=n_1\\cos\\theta_t$，结合 Snell 消去 $\\theta_t$：")
    st.latex(r"\boxed{\ \tan\theta_B=\frac{n_2}{n_1}\ }\qquad(\theta_B+\theta_t=90^\circ)")
    st.markdown(
        "物理上：当反射光线与折射光线垂直时，介质中沿反射方向振荡的偶极子不向该方向辐射，"
        "故 $p$ 分量反射为零，**反射光成为纯 $s$ 偏振**。")

    # ---------- 7. 临界角与全反射 ----------
    st.markdown("### ● 临界角与全内反射 ($n_1>n_2$)")
    st.markdown("由 Snell，$\\sin\\theta_t=\\tfrac{n_1}{n_2}\\sin\\theta_i$。当它达到 1 时折射角为 $90^\\circ$：")
    st.latex(r"\boxed{\ \sin\theta_C=\frac{n_2}{n_1}\ }")
    st.markdown(
        "当 $\\theta_i>\\theta_C$，$\\sin\\theta_t>1$，于是 $\\cos\\theta_t$ 变为**纯虚数**")
    st.latex(r"\cos\theta_t=i\sqrt{\Big(\tfrac{n_1}{n_2}\Big)^2\sin^2\theta_i-1}\ \equiv\ i\,q")
    st.markdown(
        "代入 Fresnel 公式，分子与分母成共轭，故 $|r_s|=|r_p|=1$ —— **全反射**，"
        "但 $r$ 带有相位 $r=e^{i\\phi}$。")

    # ---------- 8. 倏逝波 ----------
    st.markdown("### ● 倏逝波（消逝波）原理")
    st.markdown(
        "介质 2 中任何平面波都必须满足**色散关系**（即 $|\\mathbf k|=n_2\\omega/c$ 的分量形式）")
    st.latex(r"k_{tx}^2+k_{tz}^2=\Big(\frac{n_2\,\omega}{c}\Big)^2")
    st.markdown(
        "而切向波矢沿界面守恒 $k_{tx}=k_{ix}=\\dfrac{\\omega}{c}n_1\\sin\\theta_i$，"
        "代入解出法向分量：")
    st.latex(r"k_{tz}=\frac{\omega}{c}\sqrt{n_2^2-n_1^2\sin^2\theta_i}")
    st.markdown(
        "$\\theta_i>\\theta_C$ 时根号内为负，$k_{tz}$ **变为纯虚数** $k_{tz}=i\\kappa$，其中")
    st.latex(r"\kappa=\frac{\omega}{c}\sqrt{n_1^2\sin^2\theta_i-n_2^2}"
             r"=\frac{2\pi}{\lambda}\sqrt{n_1^2\sin^2\theta_i-n_2^2}")
    st.markdown("于是透射场写成")
    st.latex(r"\mathbf E_t\propto e^{i k_{tx}x}\,e^{-\kappa z}")
    st.markdown(
        "即一支**沿界面 $x$ 传播、沿法向 $z$ 指数衰减**的波——倏逝波。"
        "振幅衰减到 $1/e$ 的**穿透深度**为")
    st.latex(r"\delta=\frac{1}{\kappa}=\frac{\lambda}{2\pi\sqrt{n_1^2\sin^2\theta_i-n_2^2}}")
    st.markdown(
        "全反射并非在界面瞬间发生：光束先渗入介质 2 约 $\\delta$ 深、沿界面前移再返回，"
        "造成反射光的横向偏移——**古斯-汉欣 (Goos–Hänchen) 位移**，由反射相位的色散给出")
    st.latex(r"D=-\frac{d\phi}{dk_x}=-\frac{1}{k_0 n_1\cos\theta_i}\frac{d\phi}{d\theta_i}")
    st.latex(r"D_s=2\delta\tan\theta_i,\qquad "
             r"D_p=\frac{n_2^2}{(n_1^2+n_2^2)\sin^2\theta_i-n_2^2}\,D_s")

    # ---------- 9. s/p 相位差异的根因 ----------
    st.markdown("### ● 为什么全反射后 $\\phi_p$ 总比 $\\phi_s$ 降得更快？")
    st.markdown(
        "两条相位曲线在 $\\theta_C$ 之后分开，根子在 **s、p 在界面上受的边界约束本质不同**：\n\n"
        "- **s 偏振 (TE)**：$\\mathbf E$ 完全平行界面，只受一条「切向 $E$ 连续」约束，干净单纯。\n"
        "- **p 偏振 (TM)**：$\\mathbf E$ 在入射面内，既有切向、又有**法向分量**。法向受「$D_\\perp$ 连续」约束")
    st.latex(r"n_1^2 E_{1\perp}=n_2^2 E_{2\perp}\ \Rightarrow\ "
             r"E_\perp\ \text{在界面跃变}")
    st.markdown(
        "也就是说 $p$ 偏振在法向上**正面撞上了折射率突变**。这条多出来的法向约束，正是 $r_p$ 及其相位"
        "公式里多出 $n^2$ 因子的来源。把两个 TIR 相位写在一起对比最清楚：")
    st.latex(r"\phi_s=-2\arctan\frac{u}{n_1\cos\theta_i},\qquad "
             r"\phi_p=-2\arctan\!\Big(\frac{n_1^2}{n_2^2}\cdot\frac{u}{n_1\cos\theta_i}\Big),"
             r"\quad u=\sqrt{n_1^2\sin^2\theta_i-n_2^2}")
    st.markdown(
        "$p$ 的反正切宗量比 $s$ 多了一个因子 $(n_1/n_2)^2>1$，于是**每个角度上 $|\\phi_p|>|\\phi_s|$**，"
        "$\\phi_p$ 掉得更快。\n\n"
        "**微观图像**：全反射时光在界面外形成倏逝场、沿界面滑行一段再返回。$p$ 偏振的纵向电场会在"
        "低折射率一侧感应出**表面极化电荷**，与近场耦合更强，边界匹配迫使其相位累积更快；"
        "再经 $D=-d\\phi/dk_x$，这同时意味着 $p$ 的古斯-汉欣位移也更大。")
    st.info(
        "**易错点**：穿透深度 $\\delta=1/\\kappa$ 只依赖 $\\kappa=k_0\\sqrt{n_1^2\\sin^2\\theta_i-n_2^2}$，"
        "**与偏振无关** —— $\\delta_s=\\delta_p$。s、p 的差别不在「钻多深」，而在**相位**以及由相位色散"
        "决定的**横向位移**。")

    # ---------- 10. GH 位移的第一性原理: 角谱分解 ----------
    st.markdown("### ● 古斯-汉欣位移的第一性原理：角谱分解")
    st.markdown(
        "这个横向位移最早由**牛顿**预见，直到 1947 年才由 **Goos 与 Hänchen** 首次实验观测到"
        "（故得名）。它**不依赖任何微观极化机制**，单凭波动光学就能严格证出——"
        "核心是**角谱分解 + 傅里叶位移定理**。")
    st.markdown(
        "**① 真实光束是平面波的叠加。** 几何光学把光当成一根无限细的射线，但有限宽度的光束"
        "（如高斯光束）经傅里叶分解必然对应一段**连续角谱** $\\Phi(k_x)$："
        "朝 $\\theta_0$ 射去的光，本质是 $\\theta_0\\pm\\Delta\\theta$ 一族平面波的叠加，"
        "沿界面波矢 $k_x=k_0 n_1\\sin\\theta$。")
    st.latex(r"\psi_{\text{in}}(x)=\int \Phi(k_x)\,e^{ik_x x}\,dk_x")
    st.markdown(
        "**② 每个分量的全反射相位不同。** TIR 区 $|r|=1$、$r=e^{i\\phi(k_x)}$，而 $\\phi$ 是 $\\theta$"
        "（即 $k_x$）的剧烈函数。反射后各平面波分量的相位被「重新洗牌」：")
    st.latex(r"\psi_{r}(x)=\int e^{i\phi(k_x)}\,\Phi(k_x)\,e^{ik_x x}\,dk_x")
    st.markdown(
        "**③ 相位的角度梯度 = 空间平移。** 在中心 $k_{x0}$ 附近把 $\\phi$ 一阶泰勒展开")
    st.latex(r"\phi(k_x)\approx \phi(k_{x0})+"
             r"\frac{d\phi}{dk_x}\Big|_{k_{x0}}(k_x-k_{x0})")
    st.markdown("把线性项放回积分、常数相位提到积分号外，余下的恰好是一个**平移过的原波包**：")
    st.latex(r"\psi_{r}(x)=e^{i\alpha}\!\int \Phi(k_x)\,"
             r"e^{ik_x\left(x+\frac{d\phi}{dk_x}\right)}dk_x"
             r"=e^{i\alpha}\,\psi_{\text{in}}\!\left(x+\tfrac{d\phi}{dk_x}\right)")
    st.markdown("由**傅里叶位移定理**，反射波包整体沿界面平移")
    st.latex(r"\boxed{\,D=-\frac{d\phi}{dk_x}"
             r"=-\frac{1}{k_0 n_1\cos\theta_0}\frac{d\phi}{d\theta_0}\,}")
    st.markdown(
        "常数项 $\\phi(k_{x0})$ 只改变整体相位、不移动位置；真正把反射光束**质心**「顶」过去的，"
        "是 $\\phi$ 对 $k_x$ 的**斜率**。这与空间域得到的同一个 $D=-d\\phi/dk_x$ 完全吻合——"
        "**空间域直觉、频域严格证明、微观边界根因，三者指向同一结果。**")

    st.caption("把 θᵢ 拖到临界角以上，下方「逐点查看」会实时显示 δ、相移 φ 与 G–H 位移。")


def main():
    st.set_page_config(page_title="The Fresnel Equations",
                       page_icon="🔬", layout="wide")

    # ======================== 侧边栏: 介质参数 ========================
    # 常用材料 (名称, 折射率), 按 n 升序
    MATERIALS = [
        ("空气", 1.00),
        ("冰", 1.31),
        ("水", 1.33),
        ("乙醇", 1.36),
        ("熔融石英", 1.46),
        ("有机玻璃 PMMA", 1.49),
        ("玻璃 BK7", 1.50),
        ("聚碳酸酯", 1.59),
        ("蓝宝石", 1.77),
        ("重火石 SF11", 1.78),
        ("立方氧化锆", 2.16),
        ("金刚石", 2.42),
        ("碳化硅 SiC", 2.65),
    ]
    CUSTOM = "自定义…"

    def mat_label(name, val):
        return f"{name} ({val:.2f})"

    OPTIONS = [mat_label(n, v) for n, v in MATERIALS] + [CUSTOM]
    VAL_BY_LABEL = {mat_label(n, v): v for n, v in MATERIALS}

    def label_for_value(val):
        """折射率匹配到材料标签; 无匹配则归为自定义 (容差 < 半个步进, 兼顾浮点漂移)"""
        for name, v in MATERIALS:
            if abs(v - val) < 1e-3:
                return mat_label(name, v)
        return CUSTOM

    # 初始化默认值 (n₁=空气=1.0, n₂=玻璃 BK7=1.5)
    if "n1" not in st.session_state:
        st.session_state.n1 = 1.0
        st.session_state.n1_sel = mat_label("空气", 1.00)
    if "n2" not in st.session_state:
        st.session_state.n2 = 1.5
        st.session_state.n2_sel = mat_label("玻璃 BK7", 1.50)

    def on_select(prefix):
        """选下拉材料 → 填入折射率 (选「自定义」则保持当前值)"""
        label = st.session_state[f"{prefix}_sel"]
        if label != CUSTOM:
            st.session_state[prefix] = VAL_BY_LABEL[label]

    def on_input(prefix):
        """手动改折射率 → 下拉同步到对应材料或「自定义」"""
        st.session_state[f"{prefix}_sel"] = label_for_value(st.session_state[prefix])

    with st.sidebar:
        st.header("介质折射率")

        st.selectbox("n₁  入射介质", OPTIONS, key="n1_sel",
                     on_change=on_select, args=("n1",),
                     help="选材料自动填入折射率，或在下方手动输入")
        n1 = round(st.number_input("折射率 n₁", min_value=1.0, max_value=3.0,
                                   step=0.01, key="n1", format="%.2f",
                                   on_change=on_input, args=("n1",)), 2)

        st.selectbox("n₂  透射介质", OPTIONS, key="n2_sel",
                     on_change=on_select, args=("n2",),
                     help="选材料自动填入折射率，或在下方手动输入")
        n2 = round(st.number_input("折射率 n₂", min_value=1.0, max_value=3.0,
                                   step=0.01, key="n2", format="%.2f",
                                   on_change=on_input, args=("n2",)), 2)

        st.divider()
        theta_b = brewster_angle(n1, n2)
        theta_c = critical_angle(n1, n2)
        st.caption("关键角度")
        ac1, ac2 = st.columns(2)
        ac1.metric("布儒斯特角", f"{theta_b:.1f}°",
                   help=r"p 偏振反射率为零：$\tan\theta_B = n_2/n_1$")
        if theta_c is not None:
            ac2.metric("临界角", f"{theta_c:.1f}°",
                       help=r"全内反射起始角 $(n_1>n_2)$：$\sin\theta_C = n_2/n_1$")
        else:
            ac2.caption("无临界角\n(n₁ < n₂)")
        st.divider()
        st.caption("法向入射参考  (θᵢ = 0°)")
        R0 = normal_reflectance(n1, n2)
        nc1, nc2 = st.columns(2)
        nc1.metric("R", f"{R0:.4f}")
        nc2.metric("T", f"{1 - R0:.4f}")

    # ======================== 标题 ========================
    _tl, _tc, _tr = st.columns([1, 6, 1], vertical_alignment="center")
    with _tc:
        st.markdown(
            "<style>@import url('https://fonts.googleapis.com/css2?"
            "family=Source+Serif+4:wght@500;600&display=swap');</style>"
            "<h1 style='text-align:center;margin:0;"
            "font-family:\"Source Serif 4\",\"Tiempos\",Georgia,"
            "\"Times New Roman\",serif;"
            "font-weight:600;letter-spacing:0.3px'>The Fresnel Equations</h1>",
            unsafe_allow_html=True,
        )
    with _tr:
        if st.button("📖", key="theory_btn"):
            show_theory()
    st.divider()
    st.caption(
        "以下分析均基于**单色平面波**在理想平界面上的反射与透射。"
        f"当前:  n₁ = {n1}  →  n₂ = {n2}"
    )

    # 全角度计算
    angles = np.linspace(0, 89.9, 600)
    R_s, R_p, T_s, T_p, theta_t_arr, is_tir_arr, r_s_arr, r_p_arr, \
        t_s_arr, t_p_arr = compute_fresnel(n1, n2, angles)

    # 反射相位 φ(θ) 及古斯-汉欣横向位移 (以波长 λ 为单位; 仅 TIR 区有意义)
    #   φ = arg(r),  D = -dφ/dk_x = -(dφ/dθ)/(k₀ n₁ cosθ)
    th_rad = np.radians(angles)
    cos_arr = np.cos(th_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_s_arr = np.unwrap(np.angle(r_s_arr))
        phi_p_arr = np.unwrap(np.angle(r_p_arr))
        gh_s_arr = -np.gradient(phi_s_arr, th_rad) / (2 * np.pi * n1 * cos_arr)
        gh_p_arr = -np.gradient(phi_p_arr, th_rad) / (2 * np.pi * n1 * cos_arr)

    # ---- 反射率 / 透射率: 公式在上, 曲线在下 ----
    rcol, tcol = st.columns(2, gap="medium")
    with rcol:
        st.markdown("##### 振幅反射系数")
        st.latex(r"r_s = \frac{n_1\cos\theta_i - n_2\cos\theta_t}"
                 r"{n_1\cos\theta_i + n_2\cos\theta_t}")
        st.latex(r"r_p = \frac{n_2\cos\theta_i - n_1\cos\theta_t}"
                 r"{n_2\cos\theta_i + n_1\cos\theta_t}")
        st.caption("反射率  $R = |r|^2$")
        fig_r = plot_reflection(angles, R_s, R_p, theta_b, theta_c, n1, n2)
        st.markdown(svg_block(fig_r, 520), unsafe_allow_html=True)
    with tcol:
        st.markdown("##### 振幅透射系数")
        st.latex(r"t_s = \frac{2n_1\cos\theta_i}"
                 r"{n_1\cos\theta_i + n_2\cos\theta_t}")
        st.latex(r"t_p = \frac{2n_1\cos\theta_i}"
                 r"{n_2\cos\theta_i + n_1\cos\theta_t}")
        st.caption(r"透射率  $T = \frac{n_2\cos\theta_t}{n_1\cos\theta_i}\,|t|^2$")
        fig_t = plot_transmission(angles, T_s, T_p, theta_b, theta_c, n1, n2)
        st.markdown(svg_block(fig_t, 520), unsafe_allow_html=True)

    # ---- 振幅系数 + 反射相位: 折叠, 想深究符号/相位时再看 ----
    with st.expander("振幅系数 r, t 与反射相位 φ（功率图丢掉的信息）"):
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.caption("振幅反射系数 r（带符号）")
            fig_a = plot_amplitude(angles, r_s_arr, r_p_arr,
                                   theta_b, theta_c, is_tir_arr, n1, n2)
            st.markdown(svg_block(fig_a, 360), unsafe_allow_html=True)
        with c2:
            st.caption("振幅透射系数 t")
            fig_ta = plot_t_amplitude(angles, t_s_arr, t_p_arr,
                                      theta_c, is_tir_arr, n1, n2)
            st.markdown(svg_block(fig_ta, 360), unsafe_allow_html=True)
        with c3:
            st.caption("反射相位 φ = arg(r)")
            fig_ph = plot_phase(angles, r_s_arr, r_p_arr,
                                theta_b, theta_c, n1, n2)
            st.markdown(svg_block(fig_ph, 360), unsafe_allow_html=True)

        st.markdown(
            "$R=|r|^2$ 丢掉了 $r$ 的**符号 / 相位**，而它正是关键物理：\n\n"
            "- **传播区**（$\\theta<\\theta_C$）$r$ 为实数，相位只能是 $0$ 或 $\\pm180°$："
            "负号 = π 相变（半波损失）；$r_p$ 在 $\\theta_B$ 处**穿零变号**，对应 φ 的跳变。\n"
            "- **全反射后相位怎么变**（右图阴影区）：$\\theta>\\theta_C$ 时 $|r|=1$、$r=e^{i\\phi}$，"
            "φ 从临界角处的 $0°$ **连续地**降到掠射 $90°$ 处的 $-180°$，解析式为"
        )
        st.latex(
            r"\phi_s=-2\arctan\frac{\sqrt{n_1^2\sin^2\theta_i-n_2^2}}{n_1\cos\theta_i},"
            r"\qquad "
            r"\phi_p=-2\arctan\frac{n_1\sqrt{n_1^2\sin^2\theta_i-n_2^2}}{n_2^{2}\cos\theta_i}"
        )
        st.markdown(
            "- $\\phi_p$ 始终比 $\\phi_s$ 降得快，二者之差 $\\Delta\\phi=\\phi_p-\\phi_s$ 在中间角度出现极大值"
            "—— **菲涅尔棱镜**正是用两次全反射累计 $\\Delta\\phi=90°$ 做成 1/4 波片。\n"
            "- $t$ 恒正、无相位翻转；内反射（$n_1>n_2$）时 $t$ 可大于 1，"
            "但功率仍满足 $R+T=1$（$T$ 含 $\\tfrac{n_2\\cos\\theta_t}{n_1\\cos\\theta_i}$ 因子）。"
        )

    # ---- 逐点查看 ----
    st.divider()
    st.subheader("🔍 逐点查看")
    theta_sel = st.slider("入射角 θᵢ  [°]", 0.0, 90.0, 45.0, 0.5,
                          key="angle_slider",
                          help="拖动滑块查看任意入射角下的系数")
    idx = np.argmin(np.abs(angles - theta_sel))
    _rs, _rp = float(R_s[idx]), float(R_p[idx])
    _ts, _tp = float(T_s[idx]), float(T_p[idx])
    _tt = float(theta_t_arr[idx])
    _tir = bool(is_tir_arr[idx])

    # 全内反射: 预先算好倏逝波量 (供光线图与下方指标共用)
    delta_lam = phi_s = phi_p = dphi = gh_s = gh_p = None
    if _tir:
        u = np.sqrt(n1 ** 2 * np.sin(np.radians(theta_sel)) ** 2 - n2 ** 2)
        delta_lam = 1.0 / (2 * np.pi * u)               # 透射深度 / λ
        phi_s = np.degrees(np.angle(r_s_arr[idx]))
        phi_p = np.degrees(np.angle(r_p_arr[idx]))
        dphi = (phi_p - phi_s + 180) % 360 - 180        # 相对相移 → (-180,180]
        gh_s, gh_p = float(gh_s_arr[idx]), float(gh_p_arr[idx])

    ray_col, metric_col = st.columns([1, 1.5], gap="medium")
    with ray_col:
        fig_ray = plot_ray_diagram(
            theta_sel, _tt, _tir, n1, n2,
            delta_lam=delta_lam,
            gh_lam=(0.5 * (gh_s + gh_p) if _tir else None))
        st.markdown(svg_block(fig_ray, 460), unsafe_allow_html=True)
    with metric_col:
        st.caption(f"θᵢ = {theta_sel:.1f}°  时的功率系数")
        mc1, mc2 = st.columns(2)
        mc1.metric("R_s", f"{_rs:.4f}")
        mc2.metric("R_p", f"{_rp:.4f}")
        mc1.metric("T_s", f"{_ts:.4f}")
        mc2.metric("T_p", f"{_tp:.4f}")

        # 复振幅系数 r, t (全反射时为复数, 含相位)
        crs, crp = complex(r_s_arr[idx]), complex(r_p_arr[idx])
        cts, ctp = complex(t_s_arr[idx]), complex(t_p_arr[idx])
        with st.expander("振幅系数 r, t"):
            st.markdown(
                "| 系数 | s 偏振 (TE) | p 偏振 (TM) |\n"
                "|:--|:--|:--|\n"
                f"| $r$ | {fmt_complex(crs)} | {fmt_complex(crp)} |\n"
                f"| $\\lvert r\\rvert\\,\\angle\\,\\phi$ | "
                f"{abs(crs):.4f} ∠{np.degrees(np.angle(crs)):+.1f}° | "
                f"{abs(crp):.4f} ∠{np.degrees(np.angle(crp)):+.1f}° |\n"
                f"| $t$ | {fmt_complex(cts)} | {fmt_complex(ctp)} |\n"
                f"| $\\lvert t\\rvert$ | {abs(cts):.4f} | {abs(ctp):.4f} |"
            )
            if _tir:
                st.caption("全反射区 $r$ 为复数、$\\lvert r\\rvert=1$，"
                           "相位即上面的 $\\phi$；$t$ 为倏逝场的复振幅。")

        if not _tir:
            st.caption(f"折射角  θₜ = {_tt:.1f}°")
            ok_s = abs(_rs + _ts - 1.0) < 0.005
            ok_p = abs(_rp + _tp - 1.0) < 0.005
            if ok_s and ok_p:
                st.success("R + T = 1  ✓  能量守恒")
            else:
                st.warning(f"s: R+T={_rs + _ts:.4f}  p: R+T={_rp + _tp:.4f}")
        else:
            # ---- 全内反射: 倏逝波 (透射深度 / 相移 / 古斯-汉欣位移) ----
            st.success("全内反射 — 无透射光, 仅界面外的倏逝波")
            st.caption("倏逝波特性")
            ec1, ec2 = st.columns(2)
            ec1.metric("透射深度 δ", f"{delta_lam:.3f} λ",
                       help=r"倏逝波振幅衰减到 $1/e$ 的深度："
                            r"$\delta=\dfrac{\lambda}{2\pi\sqrt{n_1^2\sin^2\theta_i-n_2^2}}$")
            ec2.metric("相对相移 Δφ = φ_p − φ_s", f"{dphi:+.1f}°",
                       help=r"$p$、$s$ 反射的相位差 $\Delta\phi=\phi_p-\phi_s$")
            ec1.metric("古斯-汉欣位移 (s)", f"{gh_s:.3f} λ")
            ec2.metric("古斯-汉欣位移 (p)", f"{gh_p:.3f} λ")
            ec1.metric("反射相位 φ_s", f"{phi_s:+.1f}°")
            ec2.metric("反射相位 φ_p", f"{phi_p:+.1f}°")
            st.caption(
                "• 古斯-汉欣位移 $D$ 已**脱离几何光学范畴**：它只对**有限宽度的光束**"
                "（如高斯光束）才成立，理想的单根射线不会发生平移。\n\n"
                "• 表中长度均以**真空波长 λ** 为单位 —— λ 指入射光在真空中的波长 "
                "($\\lambda = c/\\nu$)。")

if __name__ == "__main__":
    main()
