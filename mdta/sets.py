# -*- coding: utf-8 -*-
"""**分析集合（Set）** —— 类 Materials Studio Forcite 的「Set」概念。

Forcite 里可以定义一个 *Set* 作为分析对象（例如"只看碳原子"、"只含链状分子"），
本模块提供同样的事：用一组可组合的筛选条件，从体系里挑出一批原子/分子，
作为一个独立成分参与 RDF / 接触 / 界面 / 密度 / MSD 等分析。

筛选条件（全部 **AND**，未填 = 不限）
------------------------------------
``scope``       限定范围：``"*"``（整个体系）或若干**组分名**（如 ``["polymer[ETH]"]``）
``elements``    元素白名单（``["C"]`` = 只统计碳原子；大小写不敏感）
``shape``       分子形态：``"all"`` / ``"chain"``（键图无环）/ ``"ring"``（含环）
``min_atoms``   每个分子最少原子数（**按分子**统计，不是整组）
``max_atoms``   每个分子最多原子数
``pure``        只要"整分子都由 ``elements`` 组成"的分子（例如纯碳的分子）

两种**语义**（自动判定，界面会显示是哪一种）
------------------------------------------
* **分子集合**：挑出来的分子是**完整**的（筛选没切碎分子，例如"链状分子"、按原子数筛），
  于是可以按**分子质心**追踪 —— MSD 得到的是这些分子的平动扩散；
* **原子子集**：筛选切碎了分子（例如"只取 C 原子"），只能按**原子**追踪，
  标签会写成 ``C 原子`` 之类，避免把"碳原子的 MSD"误读成"分子的 MSD"。

用法
----
>>> from mdta.sets import build_set, suggest_presets
>>> ag, info = build_set(u, {"name": "C-only", "elements": ["C"]})
>>> info["kind"]            # 'atom' —— 只挑了部分原子
>>> info["n_atoms"], info["n_molecules"]
"""
from __future__ import annotations

import re
from collections import OrderedDict

import numpy as np


class SetError(ValueError):
    """集合定义有误（元素名不认识、范围为空等）。"""


#: 常见元素符号（用于把 ``types`` 里的原子类型名归一化）
_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")


def _elements_of(universe) -> np.ndarray:
    """逐原子元素符号（尽量给全；拿不到就用原子名/类型首字母推）。"""
    from .topology import _atom_types

    n = universe.atoms.n_atoms
    try:
        el = np.asarray(universe.atoms.elements, dtype=str)
        if el.size == n and any(str(x).strip() for x in el):
            return np.array([str(x).strip().upper() for x in el], dtype=object)
    except Exception:                                    # noqa: BLE001
        pass
    t = _atom_types(universe)
    if t is not None and t.size == n:
        return np.array([str(x).strip().upper() for x in t], dtype=object)
    return np.array([""] * n, dtype=object)


def available_elements(universe) -> list[str]:
    """体系里出现过的元素（去空、去重、按字母排序），供界面做成多选项。"""
    el = _elements_of(universe)
    vals = sorted({str(e) for e in el if str(e).strip()})
    return [v for v in vals if _ELEMENT_RE.match(v)][:40]


def _molecule_index(universe) -> np.ndarray:
    """逐原子分子编号（优先 molnums / 键连通分量，复用 topology 的成熟实现）。"""
    from .topology import molecule_ids

    ids = molecule_ids(universe.atoms)
    if ids is None:
        return np.zeros(universe.atoms.n_atoms, dtype=int)
    return np.asarray(ids, dtype=int)


def molecule_profile(universe) -> dict:
    """**每个分子**的形态学档案：原子数 / 键数 / 是否含环 / 是否链状。

    ``has_ring`` 用**独立环数**（cyclomatic number）判定：
    ``cycles = E − V + 1``（单个连通分量），>0 就说明有环。桥/环之分这里不细分 ——
    对"链状 vs 环状"这个用途足够，而且不依赖 rdkit / networkx。
    """
    u = universe
    mol = _molecule_index(u)
    n_mol = int(mol.max()) + 1 if mol.size else 0
    atoms = np.bincount(mol, minlength=n_mol)
    bonds = np.zeros(n_mol, dtype=int)
    try:
        if u.bonds is not None and len(u.bonds):
            # ⚠️ 必须用 `indices`（原子索引），不能 `for ia, ib in u.bonds` ——
            # 后者给的是 Atom 对象，`int(Atom)` 直接抛 TypeError（会被 except 吞掉，
            # 于是所有分子都被判成"无环"；这正是第一版环判定失效的原因）。
            idx = np.asarray(u.bonds.indices, dtype=int).reshape(-1, 2)
            same = mol[idx[:, 0]] == mol[idx[:, 1]]
            np.add.at(bonds, mol[idx[same, 0]], 1)
    except Exception:                                    # noqa: BLE001
        pass
    cycles = np.maximum(bonds - atoms + 1, 0)
    return {
        "mol": mol,
        "n_mol": n_mol,
        "n_atoms": atoms,
        "n_bonds": bonds,
        "cycles": cycles,
        "has_ring": cycles > 0,
        "is_chain": (cycles == 0) & (atoms > 0),
    }


def _resolve_scope(universe, scope, components: dict | None) -> np.ndarray:
    """把 ``scope`` 变成原子掩码（True = 在范围内）。"""
    n = universe.atoms.n_atoms
    if not scope:
        return np.ones(n, dtype=bool)
    names = [str(x) for x in (scope if isinstance(scope, (list, tuple, set))
                              else [scope]) if str(x).strip()]
    if not names or any(x in ("*", "all", "()") for x in names):
        return np.ones(n, dtype=bool)
    comps = components or {}
    mask = np.zeros(n, dtype=bool)
    for name in names:
        ag = comps.get(str(name))
        if ag is None:
            continue
        idx = np.asarray(ag.indices, dtype=int)
        mask[idx] = True
    if not mask.any():
        raise SetError(f"范围 {list(scope)} 没有选中任何原子（检查组分名）")
    return mask


def build_set(universe, spec: dict, components: dict | None = None) -> tuple:
    """按 ``spec`` 生成 ``(AtomGroup, info)``。-- 见模块文档的字段说明。

    ``info`` 里带 ``kind``（``"molecule"`` / ``"atom"``）、``n_atoms``、
    ``n_molecules``、``label``、以及实际生效的筛选条件，方便界面回显。
    """
    u = universe
    spec = dict(spec or {})
    name = str(spec.get("name") or "set").strip() or "set"

    scope = spec.get("scope")
    elements = [str(e).strip().upper() for e in (spec.get("elements") or []) if str(e).strip()]
    shape = str(spec.get("shape") or "all").lower()
    min_atoms = int(spec.get("min_atoms") or 0)
    max_atoms = int(spec.get("max_atoms") or 0)
    pure = bool(spec.get("pure"))

    prof = molecule_profile(u)
    mol = prof["mol"]
    n_mol = prof["n_mol"]

    # ---- 1) 分子层面的筛选（范围 / 形态 / 规模 / 纯度）
    keep_mol = np.ones(n_mol, dtype=bool)
    scope_mask = _resolve_scope(u, scope, components)
    mol_in_scope = np.zeros(n_mol, dtype=bool)
    np.logical_or.at(mol_in_scope, mol[scope_mask], True)
    keep_mol &= mol_in_scope

    if shape == "chain":
        keep_mol &= prof["is_chain"]
    elif shape in ("ring", "cyclic"):
        keep_mol &= prof["has_ring"]
    if min_atoms > 0:
        keep_mol &= prof["n_atoms"] >= min_atoms
    if max_atoms > 0:
        keep_mol &= prof["n_atoms"] <= max_atoms

    # 元素白名单（按原子过滤，先算出来）
    el = _elements_of(u)
    elem_mask = np.zeros(u.atoms.n_atoms, dtype=bool)
    if elements:
        want = set(elements)
        elem_mask = np.isin(el, list(want))

    if pure and elements:
        # 整分子都由 elements 组成：分子内**所有**原子都命中白名单
        counts = np.bincount(mol[elem_mask], minlength=n_mol)
        keep_mol &= counts >= prof["n_atoms"]

    if not keep_mol.any():
        raise SetError("这组筛选条件没有选中任何分子/原子（放宽元素、形态或原子数范围试试）")

    picked_mol = np.nonzero(keep_mol)[0]
    mol_mask = np.isin(mol, picked_mol)

    # ---- 2) 决定语义：有没有被"元素筛选"切碎分子
    if elements:
        # 选中分子里，有多少分子是**全原子**都留下
        atoms_per_mol = prof["n_atoms"][picked_mol]
        kept_per_mol = np.bincount(mol[mol_mask & elem_mask], minlength=n_mol)[picked_mol]
        intact = np.all(kept_per_mol >= atoms_per_mol) if picked_mol.size else False
    else:
        intact = True

    final_mask = mol_mask & elem_mask if elements else mol_mask
    if not final_mask.any():
        raise SetError("筛选后没有剩下任何原子")

    ag = u.atoms[final_mask]
    kind = "molecule" if intact else "atom"
    n_pick = int(final_mask.sum())

    bits = []
    if scope and scope not in ("*", "all", "()"):
        bits.append("范围 " + "/".join(str(s) for s in scope))
    if elements:
        bits.append("元素 " + "+".join(elements))
    if shape == "chain":
        bits.append("链状(无环)")
    elif shape in ("ring", "cyclic"):
        bits.append("含环")
    if min_atoms or max_atoms:
        bits.append(f"分子 {min_atoms or 0}–{max_atoms or '∞'} 原子")
    if pure:
        bits.append("整分子只含所选元素")
    label = f"{name}（{('、'.join(bits)) or '全部原子'}｜{n_pick:,} 原子"
    label += f" / {len(picked_mol):,} 分子）" if kind == "molecule" else "）"

    info = {
        "name": name,
        "label": label,
        "kind": kind,
        "n_atoms": n_pick,
        "n_molecules": int(len(picked_mol)),
        "elements": elements,
        "shape": shape,
        "scope": scope or "*",
        "min_atoms": min_atoms,
        "max_atoms": max_atoms,
        "pure": pure,
    }
    return ag, info


def suggest_presets(universe, components: dict | None = None) -> list[dict]:
    """给界面的一键预设（都是"常见于 MSD/RDF 的切分方式"）。

    每个预设 = ``{"name", "spec", "hint"}``；只返回**能选中原子**的那些
    （例如体系里没有环状分子，就不列"含环"）。
    """
    els = available_elements(universe)
    out: list[dict] = []

    def _try(name, spec, hint):
        try:
            _ag, info = build_set(universe, {**spec, "name": name}, components)
        except Exception:                                # noqa: BLE001
            return
        if info["n_atoms"] <= 0:
            return
        out.append({"name": name, "spec": spec, "hint": hint,
                    "preview": info["label"]})

    heavy = [e for e in els if e != "H"]
    for e in [x for x in ("C", "N", "O") if x in heavy] or heavy[:2]:
        _try(e, {"elements": [e]}, f"只统计元素 {e} 的原子（按原子追踪）")

    _try("链状分子", {"shape": "chain"},
         "键图无环的分子（烷烃类、线型聚合物），按分子质心追踪")
    _try("含环分子", {"shape": "ring"},
         "带环的分子（芳环/糖环等），按分子质心追踪")
    _try("大分子", {"min_atoms": 20},
         "每个分子 ≥20 原子（把单体/溶剂排除），按分子质心追踪")
    if els:
        _try("重原子", {"elements": [e for e in els if e != "H"]},
             "排除氢，只看重原子（按原子追踪）")
    return out


__all__ = ["SetError", "available_elements", "build_set", "molecule_profile",
           "suggest_presets"]
