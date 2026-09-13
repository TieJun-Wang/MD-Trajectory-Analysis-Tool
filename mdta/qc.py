# -*- coding: utf-8 -*-
"""科研 QC 检查（横切关注点）。

设计思路
--------
"这个数还能不能用"本来散落在各分析模块里（interface 37 处、dynamics 11 处、
crystallinity 3 处、conformation 1 处判据），人看得见、程序接不住。这里把它们
**统一导出成结构化检查项**，挂到 :class:`mdta.core.AnalysisResult.checks` 上：

    {"名称": "接触概率饱和", "级别": "warn", "结论": "...", "依据": "..."}

三个级别：

- ``ok``：正常，可以引用；
- ``warn``：数仍然可用，但有需要写进论文的保留意见（样本偏少、误差偏乐观…）；
- ``bad``：该量**不可引用**（已被拒绝给出，或口径导致假象）。

为什么不把这套逻辑塞进每个分析里：同一个判据（例如"最小镜像失效"）会同时影响
实时更新、导出与界面展示，写三遍必然漂移。分析与 QC **解耦**后，新增一个分析项
只要往 ``summary`` 里写标准键名，就自动获得 QC 覆盖。
"""

from __future__ import annotations

from typing import Any

#: 结论里出现这些词 → 该量已被拒绝/不可引用（bad）
_BAD_WORDS = ("不给出", "拒绝", "不可引用", "不建议引用")


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None          # NaN → None


def _digest_checks(checks: list[dict]) -> dict:
    out = {"ok": 0, "warn": 0, "bad": 0}
    for c in checks:
        out[c.get("级别", "ok")] = out.get(c.get("级别", "ok"), 0) + 1
    return out


def derive_checks(res) -> list[dict]:
    """按 ``res.summary`` / ``res.notes`` 自动导出 QC 检查项（原地追加）。"""
    s = res.summary
    checks: list[dict] = []
    add = lambda n, v, lvl, d="": checks.append(          # noqa: E731
        {"名称": n, "级别": lvl, "结论": v, "依据": d})

    # ---------------------------------------------------- 分析对象（口径是谁的数）
    if "分子数" in s:
        n_mol = _num(s["分子数"])
        scope = str(s.get("统计口径", ""))
        add("分析对象", f"分析对象含 {int(n_mol) if n_mol else '?'} 个分子"
                        + (f"；{scope}" if scope else ""), "ok",
            "「整组」与「逐分子」是两个不同的量：多分子组分的整组 Rg 接近盒子尺度，"
            "不等于「分子有多大」。")
    if "参与统计的链数" in s:
        add("分析对象", f"参与统计的链数 = {s['参与统计的链数']}", "ok",
            str(s.get("链端来源", "")))
    if "配对模式" in s:
        add("配对口径", f"配对模式 = {s['配对模式']}", "ok",
            "inter=只算不同分子之间，intra=只算同一分子内部，total=都算（旧口径）")
    if "配对对象" in s:
        add("配对对象", str(s["配对对象"]), "ok", "")

    # ------------------------------------------------------------ 样本数 / 误差
    small = [k for k, v in s.items()
             if k.endswith(" n") and (_num(v) is not None and _num(v) < 20)]
    if small:
        add("样本数", "样本数偏少：" + "、".join(
            f"{k[:-2]}={int(_num(s[k]))}" for k in small), "warn",
            "少于 20 个独立样本时，标准误本身不可靠（尤其是时间序列有自相关时）")
    for k, v in s.items():
        if k.endswith("标准误来源") and "不足" in str(v):
            add("误差棒可信度", f"{k[:-6]} 的误差可能偏乐观", "warn", str(v))
    for k, v in s.items():
        if k.endswith("最小镜像失效比例"):
            f = _num(v)
            if f is None:
                continue
            tag = k[:-len("最小镜像失效比例")].strip()
            if f > 0.05:
                add("PBC 追踪", f"{tag}：最小镜像失效比例 {f:.1%}（严重）", "bad",
                    "帧间隔内位移超过最小盒高的一半，跨周期追踪不可靠，D 不可引用；"
                    "请用更密的帧间隔重算")
            elif f > 0:
                add("PBC 追踪", f"{tag}：最小镜像失效比例 {f:.1%}", "warn",
                    "少量粒子跨周期追踪可能失效，D 建议交叉核对")
            else:
                add("PBC 追踪", f"{tag}：最小镜像失效比例 0", "ok",
                    "所有粒子的帧间位移都远小于半盒，跨周期追踪安全")
    for k, v in s.items():
        if k.endswith("α (log-log 斜率)"):
            f = _num(v)
            if f is None:
                continue
            tag = k[:-len("α (log-log 斜率)")].strip()
            if abs(f - 1.0) > 0.15:
                add("扩散标度", f"{tag}：MSD∝t^{f:.2f}，偏离扩散标度 α=1", "warn",
                    "拟合区间未落在扩散区，D 只能作定性参考（应延长拟合窗口或加长轨迹）")
            else:
                add("扩散标度", f"{tag}：MSD∝t^{f:.2f}（≈1）", "ok",
                    "拟合区间落在扩散区，D 可引用")
    for k, v in s.items():
        if k.endswith("拟合可靠性"):
            txt = str(v)
            lvl = "bad" if any(w in txt for w in _BAD_WORDS) else (
                "warn" if txt.startswith("MSD") or "偏离" in txt else "ok")
            add("拟合可靠性", f"{k[:-len('拟合可靠性')].strip()}：{txt}", lvl, "")

    # ---------------------------------------------------------------- 判据类 QC
    for k, v in s.items():
        if k.endswith("第一峰是否显著"):
            tag = k[:-len("第一峰是否显著")].strip()
            g = _num(s.get(f"{tag} 第一峰高度 g_max"))
            pos = _num(s.get(f"{tag} 第一峰位置 (Å)"))
            if str(v).startswith("否"):
                add("结构峰显著性", f"{tag}：没有显著第一峰"
                    + (f"（最大 g={g:.3f} @ {pos:.2f} Å）" if g is not None else ""),
                    "warn",
                    "按判据 g_max ≥ 1.2 未通过，因此**不给配位数**。高分子熔体的"
                    "分子间第一壳层本来就宽而浅（g≈1.1–1.2），该阈值对聚合物偏严")
            else:
                add("结构峰显著性", f"{tag}：第一峰显著"
                    + (f"（g={g:.3f} @ {pos:.2f} Å）" if g is not None else ""), "ok", "")
    if "接触概率是否饱和" in s:
        if str(s["接触概率是否饱和"]).startswith("是"):
            add("指标区分度", "接触概率饱和（恒为 1，无区分度）", "warn",
                "稠密体系里「存在接触的帧比例」必然为 1；请改用「平均接触数」"
                "「接触对占据率」或「配位数分布」")
        else:
            add("指标区分度", "接触概率未饱和，有区分度", "ok", "")
    if "平均接触对数" in s and _num(s.get("平均接触对数")) == 0.0:
        dmin = _num(s.get("最小原子间距 (Å)"))
        if dmin is not None:
            add("接触统计有效性", "平均接触对数 = 0，但最小原子间距只有 "
                f"{dmin:.2f} Å", "bad",
                "两组原子很可能属于**同一个分子**（共价相连），inter 口径把配对全部"
                "剔除；应改用 total 口径或换一组配对")
    rel = s.get("界面判据可靠性") or s.get("界面宽度说明")
    if rel is not None:
        txt = str(rel)
        lvl = "bad" if ("不能解释为真实界面" in txt or "未通过" in txt) else "warn"
        add("界面判据", txt, lvl, str(s.get("界面说明", "")))
    if "1D 台阶模型是否适用" in s:
        ok = str(s["1D 台阶模型是否适用"]).startswith("是")
        add("界面模型适用性", f"1D 台阶模型是否适用：{s['1D 台阶模型是否适用']}",
            "ok" if ok else "warn",
            "不适用时界面宽度只是密度涨落的特征尺度，不是真实界面")

    # ------------------------------------------------------- 定义是否成立/权重
    for k, v in s.items():
        if "是否可定义" in k and str(v).startswith("否"):
            add("量是否有定义", f"{k}：否（已拒绝给出）", "bad",
                str(s.get(f"{k.split('是否可定义')[0]}说明", "")))
    if "权重 (取向/二面角/局部)" in s:
        w = str(s["权重 (取向/二面角/局部)"])
        if w.rstrip().endswith("0.000"):
            add("指数构成", "局部结构分量权重为 0（未给 g_ref），指数只由取向与二面角构成",
                "warn", "RDF 堆积峰高度依赖数密度、无法跨体系比较，故默认不计入")
        else:
            add("指数构成", f"权重 = {w}", "ok", "")
    if "指数名称" in s:
        add("指标性质", str(s["指数名称"]), "warn",
            "该指数用于**相对**比较，不可当作绝对结晶度；定量结晶度需按研究体系确定判据")

    # ------------------------------------------------- BOO / 邻域定义是否合理
    nb = _num(s.get("平均邻居数"))
    if nb is not None:
        cut = _num(s.get("邻域半径 (Å)"))
        if nb < 4:
            add("邻域定义", f"平均邻居数只有 {nb:.2f}（邻域半径 {cut} Å）", "bad",
                "稠密体系的每个原子应有 6–14 个邻居。这么低说明邻域半径圈到的是"
                "**共价键内的原子**（1-2 配对），此时的 q_l 与由它派生的结晶分数"
                "**全部不可用**；请把 cutoff 调到第一非键壳层（聚合物熔体约 5 Å）"
                "或留空让工具自动推定。")
        elif nb > 24:
            add("邻域定义", f"平均邻居数高达 {nb:.1f}（邻域半径 {cut} Å）", "warn",
                "邻域半径过大会把第二、第三壳层一起圈进来，q_l 会被平均掉、"
                "对局部晶序不再灵敏。")
        elif nb < 6:
            add("邻域定义", f"平均邻居数偏少 {nb:.1f}（邻域半径 {cut} Å）", "warn",
                "稠密液体的第一壳层应有 6–14 个邻居。偏少说明自动推定的邻域半径"
                "卡在第一壳层**内侧**（分子液体的壳层比原子液体更宽），"
                "建议显式指定 cutoff 后重算，例如聚合物熔体取 5 Å 左右。")
        else:
            add("邻域定义", f"平均邻居数 {nb:.1f}（邻域半径 {cut} Å）", "ok",
                "落在稠密体系合理的 6–14 个邻居范围内")

    # ------------------------------------------------------------- 单粒子/退化
    for k, v in s.items():
        if k.endswith("帧数") and _num(v) is not None:
            n_fr = int(_num(v))
            add("样本规模", f"{k[:-2]} = {n_fr}",
                "ok" if n_fr >= 20 else "warn",
                "帧数与自相关时间共同决定统计误差；帧数偏少时标准误不可靠")
    for note in res.notes:
        if "trans 判据" in str(note):
            add("构象判据", str(note), "ok",
                "同一轨迹上「二面角分析」与「结构有序度」共用这一个阈值")
    for k, v in s.items():
        if k.endswith("追踪粒子数") and _num(v) == 1 and any(
                "各向异性" in x for x in s):
            add("各向异性统计", f"{k[:-len('追踪粒子数')].strip()}：只有 1 个追踪粒子",
                "warn", "D∥/D⊥ 来自单个粒子，没有系综平均，数值不能当各向异性结论")
    if not res.curves and not res.summary:
        add("产出", "该分析没有产出任何曲线与统计量", "bad",
            "通常是体系缺少所需组分/原子，或参数把结果全部否决")

    # -------------------------------------------------------------------- 说明
    for note in res.notes:
        t = str(note)
        if "没有可用的质量信息" in t:
            add("质量信息", "缺少质量，已退化为等权（几何）口径", "warn", t)
        elif "缺键" in t or "没有键连接信息" in t:
            add("拓扑信息", "缺少键连接信息", "warn", t)
        elif "仅 1 个分子" in t or "只含 1 个分子" in t:
            add("分析对象", "该组分只含 1 个分子：整组量就是该分子的量", "ok", t)

    # 去重（同一名称+结论只留一条）
    seen: set[tuple[str, str]] = set()
    uniq: list[dict] = []
    for c in checks:
        key = (c["名称"], c["结论"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    res.checks.extend(uniq)
    return uniq


def checks_digest(checks: list[dict]) -> dict:
    """统计各等级检查项数量，供界面/导出显示总览。"""
    return _digest_checks(list(checks or []))


def worst_level(checks: list[dict]) -> str:
    """整体结论：只要有一条 bad 就是 bad，否则有 warn 就是 warn，否则 ok。"""
    levels = {c.get("级别", "ok") for c in (checks or [])}
    if "bad" in levels:
        return "bad"
    if "warn" in levels:
        return "warn"
    return "ok"
