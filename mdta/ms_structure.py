# -*- coding: utf-8 -*-
"""Materials Studio 文档格式（``.car`` / ``.xsd`` / ``.xtd``）的读取。

MDAnalysis **不认识**这三种 MS 自己的格式，所以这里自己解析：

============  ============================================================
``.car``     BIOSYM/Material Studio 结构文件（ASCII）：一行一个原子，
             `PBC a b c α β γ (P1)` 给晶胞；**通常没有键表**（需要按坐标猜键）。
``.xsd``     MS 结构文档（XML）：``<Atom3d XYZ="分数坐标">`` + ``<Bond Connects="a,b">``
             + ``<SpaceGroup AVector/BVector/CVector>``（盒矢量，Å）。
             文件里还有一大堆 ``ImageOf`` 的**周期镜像拷贝**，必须剔除。
``.xtd``     MS 轨迹文档：与 ``.xsd`` 同构（原子/键/盒都在），只是多了
             ``<Trajectory End="6001" Type="trj"/>`` —— 说明**坐标不在这个文件里**，
             真正的帧在旁边的 ``.trj``（本工具的 ``mdta.ms_trj`` 负责转换）。
============  ============================================================

关键点（实测用户体系 ``122C12-450K.xsd``）：

* 真实原子 **5596**（H 3172 + C 2424），另有 5355 个 ``ImageOf`` 镜像拷贝要剔除；
* ``XYZ`` 是**分数坐标**，要乘晶胞基矢 ``AVector/BVector/CVector`` 才得到 Å；
* 键表 ``<Bond Connects="156,157">`` 可直接当拓扑（比 MS 导出的 PDB 的 CONECT 完整）；
* 分子划分交给现有的"键连通分量"逻辑，不必解析 ``<Molecule>`` 树。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: 本模块处理的格式后缀
MS_STRUCTURE_SUFFIXES = (".car",)
MS_DOCUMENT_SUFFIXES = (".xsd", ".xtd")
ALL_SUFFIXES = MS_STRUCTURE_SUFFIXES + MS_DOCUMENT_SUFFIXES

_TAG_RE = {
    "atom": re.compile(r"<Atom3d ([^>]+?)/?>"),
    "bond": re.compile(r"<Bond ([^>]+?)/?>"),
    "spacegroup": re.compile(r"<SpaceGroup ([^>]+?)/?>"),
    "trajectory": re.compile(r"<Trajectory ([^>]+?)/?>"),
    "molecule": re.compile(r"<(?:Molecule|LinearChain) ([^>]+?)/?>"),
}
_ATTR_RE = re.compile(r'([A-Za-z_][\w.]*)="([^"]*)"')

#: 解析结果缓存（键 = 绝对路径 + 大小 + mtime）——一次打开里同一份文档会被读 2–3 次
_DOC_CACHE: dict = {}


def _attrs(text: str) -> dict[str, str]:
    return {k: v for k, v in _ATTR_RE.findall(text)}


def _vec(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    try:
        return np.array([float(x) for x in s.split(",")], dtype=float)
    except ValueError:
        return None


def cell_from_vectors(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """三个晶胞基矢 → MDAnalysis 的 ``[a, b, c, alpha, beta, gamma]``（Å / 度）。"""
    la, lb, lc = (float(np.linalg.norm(v)) for v in (a, b, c))
    ang = []
    for u, v in ((b, c), (a, c), (a, b)):
        cosv = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
        ang.append(float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))))
    return np.array([la, lb, lc, *ang], dtype=np.float64)


def read_car(path: str) -> dict:
    """读 BIOSYM / Material Studio 的 ``.car``（原子 + 晶胞，通常**没有**键表）。"""
    txt = Path(path).read_text(encoding="latin1", errors="replace")
    lines = txt.splitlines()
    cell = None
    names: list[str] = []
    elements: list[str] = []
    xyz: list[list[float]] = []
    resid: list[int] = []
    types: list[str] = []
    for ln in lines:
        if ln.startswith("PBC"):
            nums = ln.split()[1:7]
            try:
                cell = np.array([float(x) for x in nums[:3] + nums[3:6]], dtype=float)
            except ValueError:
                cell = None
            continue
        if not ln or ln[0] in "!;" or ln.startswith(("end", "CONECT", "PBC")):
            continue
        f = ln.split()
        if len(f) < 8:
            continue
        try:
            x, y, z = float(f[1]), float(f[2]), float(f[3])
        except ValueError:
            continue
        names.append(f[0])
        xyz.append([x, y, z])
        types.append(f[6] if len(f) > 6 else "")
        elements.append(f[7] if len(f) > 7 else f[6])
        try:
            resid.append(int(f[5]))
        except (ValueError, IndexError):
            resid.append(1)
    n = len(xyz)
    return {
        "format": "car",
        "n_atoms": n,
        "names": names,
        "elements": elements,
        "types": types,
        "resids": resid,
        "xyz": np.asarray(xyz, dtype=float),
        "cell": cell,
        "bonds": None,
        "n_images_skipped": 0,
        "trajectory_hint": None,
        "title": lines[2] if len(lines) > 2 else "",
    }


def read_ms_document(path: str) -> dict:
    """读 ``.xsd`` / ``.xtd``（XML）：真实原子、键表、晶胞、镜像剔除、轨迹声明。"""
    txt = Path(path).read_text(encoding="latin1", errors="replace")

    # ---- 晶胞（SpaceGroup 的 A/B/C 三个基矢，单位 Å）
    cell = None
    cell_vecs = None
    sg = _TAG_RE["spacegroup"].search(txt)
    if sg:
        at = _attrs(sg.group(1))
        A, B, C = _vec(at.get("AVector")), _vec(at.get("BVector")), _vec(at.get("CVector"))
        if A is not None and B is not None and C is not None:
            cell_vecs = (A, B, C)
            cell = cell_from_vectors(A, B, C)

    # ---- 原子：剔除 ImageOf（周期镜像拷贝）
    atoms: list[dict] = []
    n_images = 0
    for m in _TAG_RE["atom"].finditer(txt):
        at = _attrs(m.group(1))
        if "ImageOf" in at:
            n_images += 1
            continue
        atoms.append(at)
    ids = [a.get("ID") for a in atoms]
    id2idx = {aid: i for i, aid in enumerate(ids)}

    frac = np.array([_vec(a.get("XYZ")) if a.get("XYZ") else np.zeros(3)
                     for a in atoms], dtype=float)
    if cell_vecs is not None and frac.size:
        cart = (frac[:, 0:1] * cell_vecs[0][None, :]
                + frac[:, 1:2] * cell_vecs[1][None, :]
                + frac[:, 2:3] * cell_vecs[2][None, :])
    else:                                  # 没有晶胞时当笛卡尔坐标
        cart = frac.copy()

    # ---- 键表：只要两端都是真实原子的（镜像键也要剔除）
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    n_bond_images = 0
    for m in _TAG_RE["bond"].finditer(txt):
        at = _attrs(m.group(1))
        if "ImageOf" in at:
            n_bond_images += 1
            continue
        conn = at.get("Connects", "")
        parts = [p.strip() for p in conn.split(",") if p.strip()]
        if len(parts) != 2:
            continue                      # 多中心连接跳过（MS 的桥键）
        i, j = id2idx.get(parts[0]), id2idx.get(parts[1])
        if i is None or j is None:
            continue
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    # ---- 轨迹声明（只有 .xtd 有）
    traj_hint = None
    tr = _TAG_RE["trajectory"].search(txt)
    if tr:
        at = _attrs(tr.group(1))
        traj_hint = {"n_frames": int(at["End"]) if at.get("End", "").isdigit() else None,
                     "type": at.get("Type"), "id": at.get("ID")}

    n_mol_nodes = len(_TAG_RE["molecule"].findall(txt))
    return {
        "format": Path(path).suffix.lower().lstrip("."),
        "n_atoms": len(atoms),
        "names": [a.get("Name", "") for a in atoms],
        "elements": [a.get("Components", "") or a.get("ForcefieldType", "") for a in atoms],
        "types": [a.get("ForcefieldType", "") for a in atoms],
        "resids": None,
        "xyz": cart,
        "cell": cell,
        "bonds": pairs or None,
        "n_bonds_image_skipped": n_bond_images,
        "n_images_skipped": n_images,
        "n_molecule_nodes": n_mol_nodes,
        "trajectory_hint": traj_hint,
        "title": Path(path).stem,
    }


def read_ms_structure(path: str) -> dict:
    """按后缀分派：``.car`` / else（``.xsd`` / ``.xtd``）。**带缓存**（按 路径+大小+mtime）。

    为什么要缓存：一次打开里这份文档会被解析 2–3 次（原子数预检 → 建 Universe →
    转换后的重试），而 `.xsd` 有 4 MB 文本、每次 ~0.3 s。缓存键带上大小与 mtime，
    文件被改写后自动失效。
    """
    p = Path(path)
    try:
        st = p.stat()
        key = (str(p.resolve()), st.st_size, int(st.st_mtime))
    except OSError:
        key = None
    if key is not None and key in _DOC_CACHE:
        return _DOC_CACHE[key]
    suf = p.suffix.lower()
    if suf in MS_STRUCTURE_SUFFIXES:
        doc = read_car(path)
    elif suf in MS_DOCUMENT_SUFFIXES:
        doc = read_ms_document(path)
    else:
        raise ValueError(f"不是 Materials Studio 结构/文档格式：{path}")
    if key is not None:
        if len(_DOC_CACHE) > 4:
            _DOC_CACHE.clear()
        _DOC_CACHE[key] = doc
    return doc


def _molecule_index(n_atoms: int, bonds) -> np.ndarray:
    """按键表做并查集，返回每个原子的分子下标（没有键表时全体一个分子）。"""
    if not bonds:
        return np.zeros(n_atoms, dtype=np.int64)
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in bonds:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n_atoms)], dtype=np.int64)
    _uniq, inv = np.unique(roots, return_inverse=True)
    return inv.astype(np.int64)


def _molecule_kinds(mol: np.ndarray, elements: list[str], types: list[str]) -> list[str]:
    """给每个分子一个 3 字标签（当 resname / segid 用），让界面能按组分区分。

    实测用户体系：122 条十二烷（C+H）与 1 片石墨烯（纯 C，960 原子）混在一个
    文件里；若所有分子同名，界面的"组分"会把它们并成一组，用户就选不出聚合物。
    判据（保守）：纯碳且 ≥100 原子 → ``GRA``；含 H → ``ETH``；其余 → ``MOL``。
    """
    n_mol = int(mol.max()) + 1 if mol.size else 0
    kinds: list[str] = []
    for m in range(n_mol):
        idx = np.flatnonzero(mol == m)
        el = {str(elements[i]).upper() for i in idx}
        ff = {str(types[i]).lower()[:2] for i in idx}
        if el <= {"C"} and idx.size >= 100:
            kinds.append("GRA")
        elif "H" in el:
            kinds.append("ETH")
        elif {"C4"} <= ff:
            kinds.append("GRA")
        else:
            kinds.append("MOL")
    return kinds


def build_universe(doc: dict, *, time_ps: float = 0.0):
    """把解析结果变成 MDAnalysis ``Universe``（元素/质量/键/分子/段 + 1 帧坐标）。

    分子划分与组分命名也在这里定好：每个键连通分量 = 一个 residue = 一个 molecule，
    并按分子类型给 ``resname`` / ``segid``（见 :func:`_molecule_kinds`），
    这样界面上的"组分 / 链类型"就能直接选出"聚合物 122 条 / 石墨烯 1 片"。
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    n = int(doc["n_atoms"])
    if n == 0:
        raise ValueError("文件里没有解析出原子")
    coords = np.asarray(doc["xyz"], dtype=np.float32)[None, :, :]
    cell = doc.get("cell")
    dims = (np.asarray(cell, dtype=np.float32)
            if cell is not None else np.zeros(6, dtype=np.float32))
    if dims.shape == (3,):                 # .car 的 PBC 行只给了 3 个长度
        dims = np.array([*dims, 90.0, 90.0, 90.0], dtype=np.float32)

    names = [str(x) or "X" for x in (doc.get("names") or ["X"] * n)]
    elems = [str(x).upper() or "X" for x in (doc.get("elements") or ["X"] * n)]
    types = [str(x) or "X" for x in (doc.get("types") or ["X"] * n)]
    bonds = doc.get("bonds") or []

    mol = _molecule_index(n, bonds)
    n_mol = int(mol.max()) + 1 if n else 0
    kinds = _molecule_kinds(mol, elems, types)
    # 每种分子类型一个 **segment**：界面的"组分"按 segid 细分（polymer[ETH] /
    # polymer[GRA]），这样用户就能直接选出"122 条十二烷"而不带那片石墨烯。
    seg_names = sorted(set(kinds))
    seg_of_kind = {k: i for i, k in enumerate(seg_names)}
    res_segindex = np.array([seg_of_kind[k] for k in kinds], dtype=np.int64)

    u = mda.Universe.empty(n, n_residues=max(n_mol, 1),
                           atom_resindex=mol,
                           n_segments=len(seg_names),
                           residue_segindex=res_segindex,
                           trajectory=True)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("elements", elems)
    u.add_TopologyAttr("types", types)
    # resnames / resids / segids 都是**残基级/段级**（这里 1 个分子 = 1 个残基）
    u.add_TopologyAttr("resnames", kinds)
    u.add_TopologyAttr("resids", np.arange(1, max(n_mol, 1) + 1))
    u.add_TopologyAttr("segids", seg_names)
    if bonds:
        u.add_TopologyAttr("bonds", [tuple(int(x) for x in p) for p in bonds])
    u.trajectory = MemoryReader(coords, order="fac", dimensions=dims)
    return u


__all__ = ["MS_STRUCTURE_SUFFIXES", "MS_DOCUMENT_SUFFIXES", "ALL_SUFFIXES",
           "read_car", "read_ms_document", "read_ms_structure", "build_universe",
           "cell_from_vectors"]
