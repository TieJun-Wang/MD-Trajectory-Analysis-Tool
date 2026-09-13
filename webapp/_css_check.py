# -*- coding: utf-8 -*-
"""确认构建产物里包含折叠面板与"整页不滚动"的样式。"""
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "http://127.0.0.1:8000"

html = urllib.request.urlopen(BASE + "/", timeout=20).read().decode("utf-8")
m = re.search(r'href="(/assets/[^"]+\.css)"', html)
css = urllib.request.urlopen(BASE + m.group(1), timeout=20).read().decode("utf-8")
print(f"构建产物 {m.group(1)}：{len(css)} bytes\n")

flat = css.replace(" ", "")
need = [
    (".acc-item", "折叠面板容器"),
    (".acc-item.open", "展开态占满剩余高度"),
    (".acc-head", "折叠标题栏"),
    (".acc-body", "折叠内容区"),
    (".ops", "左操作台"),
    (".layout", "主体分栏"),
    (".ptab", "参数表（表格约束布局）"),
    ("table-layout:auto", "参数表列宽由内容决定"),
    ("appearance:textfield", "隐藏数字输入箭头（省宽、外观一致）"),
    (".chip-pick", "组分胶囊多选"),
    (".fgroup", "分析功能紧凑分组"),
    (".pgroup", "参数按分析项分组的小标题"),
    (".notes-foot", "说明栏底部提示"),
    (".logview", "运行日志页"),
    (".tabbody", "标签页内容区"),
    (".badge-live", "实时更新徽标"),
    ("@keyframeslivepulse", "徽标脉冲动画"),
    (".chart-wait", "等待中的图表提示"),
    (".ops.collapsed", "操作台折叠态"),
    (".rail", "折叠竖条"),
    (".rail-l", "左侧竖条"),
    (".rail-r", "右侧竖条"),
    (".notes", "右侧图表说明栏"),
    (".notes.collapsed", "说明栏折叠态"),
    (".notes-body", "说明栏内容"),
    (".notes-scroll", "说明栏内部滚动"),
    (".chartarea.fs", "全屏（CSS 退路）"),
    (".chartarea:fullscreen", "全屏（原生 API）"),
    ("button.mini.active", "「仅 XX」预设高亮"),
    ("writing-mode:vertical-rl", "竖条文字竖排"),
    ("transition:flex-basis", "折叠/展开的宽度过渡"),
    ("--notes-width", "说明栏宽度变量"),
    (".notes-clip", "说明栏折叠载体（缩到 0，竖条才点得到）"),
    (".inforeport", "体系信息完整报告容器"),
    ("overflow-x:hidden|overflow:hiddenauto", "锁定横向滚动（压缩后为简写）"),
    ("overflow-wrap:anywhere", "超宽内容自动换行"),
    ("table.inforows", "信息表格样式"),
    ("table-layout:fixed", "表格列宽按比例分配，不被内容撑开"),
    (".infosec-title", "信息小节标题"),
    (".infosec-note", "信息小节备注"),
    (".notelist", "说明条目列表（悬挂缩进）"),
    (".noterow", "说明条目（换行后与首行对齐）"),
    (".notebullet", "说明项目符号列"),
    ("table.notes-table", "窄栏统计量表格"),
    (".topbar-live", "顶栏实时进度"),
    (".tbl-fill", "顶栏进度条填充"),
    (".statnotes", "统计量页说明列表（与 .notes 容器区分开）"),
    (".navmodule", "图表/数据表导航的模块层"),
    (".navmodule-head", "模块标题"),
    (".navmodule .navgroup", "模块内分析项缩进"),
    (".stattabs", "统计量单一模块标签栏"),
    (".stattabbody", "模块标签内容区"),
    (".tabhint", "标签页提示文字"),
    (".explabel", "导出区行标签"),
    (".lastout", "最近导出（目录 + 时间）"),
    (".dirpicker", "目标目录选择器"),
    (".dp-item", "目录选择器条目"),
    (".chinline", "模块/格式胶囊勾选"),
    (".exppath", "目标目录显示框"),
    (".filetabs", "导出的「可选文件」标签栏"),
    (".filegroup", "导出文件按模块分板块"),
    (".filegroup-head", "板块标题栏（含全选/全不选）"),
    ("table.filetable", "可选文件表"),
]

#: 这些样式**必须已经不存在**（结构改掉了，留着就是死代码）
gone = [
    (".chartfoot", "旧的「图表下方固定块」（说明已移到右侧栏）"),
    (".ops-toggle", "旧的顶栏收起按钮（已统一到最左侧竖条）"),
    ("white-space:pre;", "旧的定宽文本报告（报告已改为表格）"),
    (".logo", "顶栏 logo 方块（改为两行标题）"),
    (".statmodule", "统计量旧的分板块样式（已改为单一模块标签）"),
]

bad = []
for key, desc in need:
    # 允许写成多个候选（用 "|" 分隔）：压缩器会把
    # `overflow-y:auto; overflow-x:hidden` 合并成简写 `overflow:hidden auto`，
    # 语义完全一样，所以两种写法都要认。
    alts = [a.replace(" ", "") for a in key.split("|")]
    hit = any(a in flat for a in alts)
    print(f"  {'有' if hit else '缺!'}  {key:34} {desc}")
    if not hit:
        bad.append(key)

print()
for key, desc in gone:
    gone_ok = key.replace(" ", "") not in flat
    print(f"  {'已移除' if gone_ok else '残留!'}  {key:34} {desc}")
    if not gone_ok:
        bad.append(f"{key}(残留)")

rule = re.search(r"html,body,#root\{([^}]*)\}", css)
print(f"\n  html,body,#root 规则: {rule.group(1) if rule else '未找到'}")
print(f"  overflow:hidden 出现 {css.count('overflow:hidden')} 次"
      f"（整页 + 各分栏锁死，不产生页面级滚动）")

# ---------------------------------------------------- 快捷按钮行是否排得下
# 「全选 + 仅链构象/仅界面/仅结晶/仅辅助」必须在一行内放下（用户明确要求）。
# 这里用真实 CSS 数值做一次静态估算：中文字宽 ≈ font-size，逐项相加。
def _px(pattern, default):
    m = re.search(pattern, css)
    return float(m.group(1)) if m else default


ops_w = _px(r"--ops-width:\s*(\d+(?:\.\d+)?)px", 372.0)
body_pad = re.search(r"\.acc-body\{[^}]*?padding:\s*([^;}]+)", css)
pad_lr = 24.0
if body_pad:
    parts = [p for p in body_pad.group(1).split() if p.endswith("px")]
    try:
        vals = [float(p[:-2]) for p in parts]
        # CSS padding 简写：1/2/3/4 个值
        pad_lr = 2 * (vals[1] if len(vals) >= 2 else vals[0])
    except (ValueError, IndexError):
        pass

btn = re.search(r"\.btnrow button\.mini\{([^}]*)\}", css)
btn_pad, btn_fs, btn_border = 12.0, 11.0, 2.0
if btn:
    b = btn.group(1)
    m = re.search(r"padding:\s*(\d+(?:\.\d+)?)px\s+(\d+(?:\.\d+)?)px", b)
    if m:
        btn_pad = 2 * float(m.group(2))
    m = re.search(r"font-size:\s*(\d+(?:\.\d+)?)px", b)
    if m:
        btn_fs = float(m.group(1))

row = re.search(r"\.btnrow\{([^}]*)\}", css)
gap, nowrap = 4.0, False
if row:
    g = row.group(1)
    m = re.search(r"gap:\s*(\d+(?:\.\d+)?)px", g)
    if m:
        gap = float(m.group(1))
    nowrap = "flex-wrap:nowrap" in g.replace(" ", "")

LABELS = ["全选", "仅构象", "仅结构", "仅取向", "仅输运"]
# ↑ 必须与 `mdta.pipeline.GROUP_SHORT`（经 /api/analyses 下发）一致；
#   按钮文字 = "仅" + 缩写，完整分组名放在 tooltip 上。


def _text_w(s, fs):
    """粗估文本宽度：CJK ≈ 1.0 em，ASCII ≈ 0.55 em。"""
    w = 0.0
    for ch in s:
        w += fs * (0.55 if ord(ch) < 0x2E80 else 1.0)
    return w


need = sum(_text_w(t, btn_fs) + btn_pad + btn_border for t in LABELS) + gap * (len(LABELS) - 1)
avail = ops_w - pad_lr
print(f"\n  快捷按钮行（{len(LABELS)} 个）:")
print(f"    标签: {LABELS}")
print(f"    估算占用 {need:.1f}px / 可用 {avail:.1f}px"
      f"（操作台 {ops_w:.0f}px − 左右内边距 {pad_lr:.0f}px）")
print(f"    不换行(nowrap) = {nowrap}")
if need > avail:
    bad.append(f"快捷按钮行放不下({need:.0f}>{avail:.0f})")
    print("    -> 放不下！")
else:
    print(f"    -> 排得下，余量 {avail - need:.1f}px")

# ------------------------------------------- 参数区布局（表格约束）
# 用户建议并采纳：参数改用 <table class="ptab"> 约束排版——列在各行之间共享宽度，
# 因此同一列的输入框左边缘天然对齐（flex 布局下标签长短不一 → 输入框起点参差不齐）。
# "提示文字不被截断"已升级为**元素级保证**（每个输入框带内联 min-width），
# 由 _ui_render.mjs 直接读渲染结果断言，这里只校验表格本身的基础规则。
ptab = re.search(r"\.ptab\{([^}]*)\}", flat)
th_rule = re.search(r"\.ptabth\{([^}]*)\}", flat)
spin = re.search(r"\.ptabinput\[type=number\]::-webkit-inner-spin-button\{([^}]*)\}", flat)
print("\n  参数区布局（表格约束）:")
if not ptab:
    bad.append("找不到 .ptab 规则（参数区没有用表格约束布局）")
else:
    print(f"    .ptab: {ptab.group(1)}")
    if "width:100%" not in ptab.group(1):
        bad.append(".ptab 没有 width:100%")
    if not th_rule or "white-space:nowrap" not in th_rule.group(1):
        bad.append(".ptab th 缺 nowrap（标签会折行）")
    else:
        print(f"    .ptab th: {th_rule.group(1)}")
    if not spin or "-webkit-appearance:none" not in spin.group(1):
        bad.append("数字输入的上下箭头没隐藏（占宽且外观不齐）")
    else:
        print(f"    数字输入箭头: {spin.group(1)}")
    if not bad:
        print("    -> 表格约束布局 + 标签不折行 + 箭头已隐藏 ✓")

# 提示文字的宽度保证在 JSX 里（内联 min-width），这里只确认公式里的内边距常数一致：
IN_CHROME = 14.0
print(f"    提示文字余量由 JSX 内联 min-width 保证（+{IN_CHROME:.0f}px 内边距/边框）；"
      f"渲染级断言见 _ui_render.mjs 的 checkPlaceholders")

# nowrap 三条声明必须齐全。**逐条查、不查拼接串**：压缩器会重排声明顺序
# （实测产物里是 white-space;text-overflow;…;overflow），拼接匹配会误报。
span_rule = re.search(r"\.grid3\.field>span\{([^}]*)\}", flat)
_decls = ["white-space:nowrap", "overflow:hidden", "text-overflow:ellipsis"]
if not span_rule:
    bad.append("找不到 .grid3 .field>span 规则（参数标签可能折行）")
else:
    _miss = [d for d in _decls if d not in span_rule.group(1)]
    if _miss:
        bad.append(f"参数标签缺声明 {_miss}（长标签会折成两行）")
    else:
        print("    -> 标签 nowrap / overflow / ellipsis 三条齐全，强制单行 ✓")

# ------------------------------- 说明栏：纵向可滚、横向必须锁死
# 只写 overflow-y:auto 时另一轴会由 visible 计算成 auto，于是超宽就冒出横向滚动条。
for sel, label in ((r"\.notes-scroll\{", "说明栏滚动区"),
                   (r"\.inforeport\{", "体系信息报告区")):
    m = re.search(sel + r"([^}]*)", css)
    body = m.group(1) if m else ""
    locked = ("overflow-x:hidden" in body.replace(" ", "")
              or "overflow:hiddenauto" in body.replace(" ", ""))
    print(f"\n  {label} 锁定横向滚动 = {locked}")
    if not m:
        bad.append(f"{label}(找不到规则)")
    elif not locked:
        bad.append(f"{label}(未锁定横向滚动)")
        print(f"    -> 规则内容: {body[:90]}")

# --------------------- 顶栏深靛紫底：底与字的明暗必须反过来
# 只改底色不调文字色，就会出现「深底上写深字」看不清；这里直接算相对亮度来卡。
def _luminance(hexcolor: str) -> float:
    h = hexcolor.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _first_hex(rule_body: str, prop: str | None = None) -> str | None:
    """取规则里的第一个 ``#rrggbb``。

    ``prop`` 指定时只在**该属性**的值里找 —— 压缩器会重排属性顺序，
    直接取"规则里第一个 hex"很容易抓到 ``color`` 而不是 ``background``。
    """
    body = rule_body
    if prop:
        m = re.search(rf"(?:^|;){prop}\s*:\s*([^;}}]*)", rule_body)
        if not m:
            return None
        body = m.group(1)
    m = re.search(r"#([0-9a-fA-F]{3,8})", body)
    return f"#{m.group(1)}" if m else None


print("\n  顶栏深靛紫底 + 反白文字（明暗对比）:")
top = re.search(r"\.topbar\{([^}]*)\}", css)
top_body = top.group(1) if top else ""
# 底纹写在 background（渐变）里，必须按属性名取
top_hex = _first_hex(top_body, "background") or _first_hex(top_body)
lum_bg = _luminance(top_hex) if top_hex else None
print(f"    .topbar 底色取样 {top_hex} → 相对亮度 "
      f"{lum_bg:.4f}" if lum_bg is not None else "    .topbar 未取到底色")
if lum_bg is None:
    bad.append("顶栏底色(取不到)")
elif lum_bg > 0.35:
    bad.append(f"顶栏底色不够深({lum_bg:.3f})")
else:
    print("    ✓ 底色够深（深靛紫）")

for sel, label in ((r"\.t1\{", "第一行标题 .t1"),
                   (r"\.t2\{", "第二行副标题 .t2")):
    m = re.search(sel + r"([^}]*)", css)
    body = m.group(1) if m else ""
    hx = _first_hex(body)
    lum = _luminance(hx) if hx else None
    if lum is None:
        print(f"    ✗ {label}: 取不到颜色")
        bad.append(f"{label}(取不到颜色)")
    else:
        contrast_ok = (lum + 0.05) / ((lum_bg or 0) + 0.05) > 4.0
        print(f"    {'✓' if contrast_ok else '✗'} {label} 颜色 {hx} → 亮度 "
              f"{lum:.4f}，与底色对比度 {(lum + 0.05) / ((lum_bg or 0) + 0.05):.1f}:1")
        if not contrast_ok:
            bad.append(f"{label} 在深底上对比度不足")

# 状态色也要够亮（在线/离线/版本）
for sel, label in ((r"\.status \.ok\{", "在线 status .ok"),
                   (r"\.status \.ver\{", "版本 .ver")):
    m = re.search(sel + r"([^}]*)", css)
    body = m.group(1) if m else ""
    hx = _first_hex(body)
    lum = _luminance(hx) if hx else 0.0
    okc = lum > 0.35
    print(f"    {'✓' if okc else '✗'} {label} 颜色 {hx} → 亮度 {lum:.4f}")
    if not okc:
        bad.append(f"{label} 在深底上偏暗")


# ⚠️ 注意：这一节只能证明**规则写在样式表里**，不能证明它**作用到了元素上**。
#    `.notes > .notes-body` 曾经就因为起不到作用而漏过两次（选择器匹配不上）——
#    真正"生效"的验证在 webapp/_test_layout.mjs（真实浏览器 + CDP 量 computed style）。
print("\n  说明栏高度链（仅检查规则存在；是否生效见 _test_layout.mjs）:")
chain = re.search(r"\.notes\s*>\s*\.notes-clip\{([^}]*)", css)
chain_body = chain.group(1).replace(" ", "") if chain else ""
body_rule = re.search(r"\.notes-clip\s*>\s*\.notes-body\{([^}]*)", css)
body_body = body_rule.group(1).replace(" ", "") if body_rule else ""
checks = [
    ("notes-clip 是 flex 容器", "display:flex" in chain_body),
    ("notes-clip 纵向排列", "flex-direction:column" in chain_body),
    ("notes-clip 有 min-height:0", "min-height:0" in chain_body),
    ("notes-body 用 .notes-clip 子选择器（与 DOM 一致）", bool(body_rule)),
    ("notes-body 有 flex", "flex:" in body_body or "flex-grow" in body_body),
    ("notes-body 有 min-height:0", "min-height:0" in body_body),
    ("notes-body 不再用 height:100%（百分比高度在 stretch 父元素上解析不出来）",
     "height:100%" not in body_body),
]
for label, ok_ in checks:
    print(f"    {'✓' if ok_ else '✗'} {label}")
    if not ok_:
        bad.append(f"说明栏高度链: {label}")

assert not bad, bad
assert rule and "overflow:hidden" in rule.group(1), rule.group(1) if rule else None
print("\n构建产物检查通过。")
