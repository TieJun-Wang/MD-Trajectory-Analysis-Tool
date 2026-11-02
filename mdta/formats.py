# -*- coding: utf-8 -*-
"""**文件格式后缀表**（全工具唯一的定义处）。

为什么单独一个模块
------------------
后缀白名单原来散落在 5 个地方：`mdta/io.py`、`webapp/backend/session.py`、
桌面 GUI（`mdta_gui.py` 的 filetypes）、CLI（`mdta_cli.py` 的 glob）、
以及前端提示文字。加一个格式就要改 5 处，**必然漏**（`.lammpstrj` 就漏过：
它在名字列表里，但 MDAnalysis 这个版本认格式名不认后缀，实际读不了）。
现在集中在这里，其余地方 import 本模块。

表的来源是**实测**，不是抄文档
------------------------------
逐个候选后缀调用 MDAnalysis 的公开 API ``get_reader_for`` / ``get_parser_for``
问一遍（脚本：``test/_probe_suffixes.py``），只写本机真能读的格式；``test/selftest.py`` 里有
一条测试盯着"表 ↔ MDAnalysis 实际能力"是否一致，将来升级 MDAnalysis 会立刻报出来。

三类角色的区分（重要）
----------------------
- ``TOPOLOGY_SUFFIXES``：含**原子身份**（名字/元素/键表）的格式，能当"拓扑"用；
- ``TRAJECTORY_SUFFIXES``：有坐标读取器的格式，能当"轨迹"用；
- ``BOTH_SUFFIXES``：两者都能（`.pdb` / `.gro` / `.xyz` 等多帧文本格式）——
  这类文件在界面上两个槽都能放。

⚠️ 纯坐标格式（`.xtc` / `.trr` / `.dcd` / `.nc` / `.mdcrd` …）**只算轨迹**：
MDAnalysis 用一个 MinimalParser 也能把它们"当拓扑"读进来，但那时体系没有原子名、
元素与质量，后续每一项质量相关的分析都会失真（qc.py 的「质量来源」规则会标不可引用）。
所以这里把它们从拓扑槽里排除，并在界面上说明原因。

显式 format 的后缀
------------------
``.lammpstrj`` / ``.dump``（LAMMPS dump）：本机 MDAnalysis 2.10 里
``LAMMPS.DumpReader`` 没有 ``EXTENSIONS`` 属性，按后缀找不到读取器，
必须显式传 ``format="LAMMPSDUMP"`` —— 实测传了就能正常读（2 帧 / 3 原子）。

LAMMPS 的 **data** 文件同理（:data:`EXPLICIT_TOPOLOGY_FORMAT`）：
``.lmp`` / ``.dat`` 这两个最常用的后缀 MDAnalysis **根本没登记**，
而 ``.lammps`` 更坑 —— 它被映射给 **DCDReader**，于是 data 文件被当成 DCD 去读，
报出来的是 ``XDR read error``（与真实原因毫无关系）。所以这四个后缀
（``.data`` / ``.lammps`` / ``.lmp`` / ``.dat``）一律显式传
``topology_format="DATA"``。

后缀不可信的地方还有一个：``in.lammps``（输入脚本）、``log.lammps``（热力学日志）
与 data 文件**共用后缀**。所以 :func:`sniff_lammps_kind` 只看内容前 40 行来分辨
data / 输入脚本 / 日志，:func:`unsupported_hint` 给 ``.cfg``（AtomEye）、
``.restart``（二进制内部状态）、``.yaml``（YAML dump）这类真读不了的格式
一条**替换做法**，而不是一句"读不了"。
"""
from __future__ import annotations

import os

# --------------------------------------------------------------- 拓扑（含原子身份）
TOPOLOGY_SUFFIXES: frozenset[str] = frozenset({
    ".pdb", ".ent",                     # PDB（含多模型）
    ".gro",                             # GROMACS 坐标（原子名 → 质量可推）
    ".xyz",                             # 通用 XYZ
    ".tpr",                             # GROMACS 二进制拓扑（带键/电荷/质量）
    ".psf",                             # NAMD/X-PLOR
    ".prmtop", ".parm7",                # AMBER
    ".top",                             # GROMACS 文本拓扑
    ".mol2",                            # Tripos
    ".pdbqt", ".pqr",                   # AutoDock / PQR
    ".dms",                             # DMS
    ".data", ".lammps",                 # LAMMPS data（MDAnalysis 的 DATAParser）
    # LAMMPS data 的常见写法：`.lmp` / `.dat` 是**最常用**的两个后缀，而 MDAnalysis
    # 只按后缀认 `.data`/`.lammps`（`.lammps` 还会被误认成 DCD！），所以这两个走
    # EXPLICIT_TOPOLOGY_FORMAT / EXPLICIT_FORMAT 显式指定成 DATA。
    ".lmp", ".dat",
    ".fhiaims",                         # FHI-aims
    ".xml",                             # Tinker
    # Materials Studio 自家格式（MDAnalysis 不认识，由 mdta.ms_structure 解析）：
    #   .car  BIOSYM/MS 结构（原子 + 晶胞，通常没有键表 → 按坐标猜键）
    #   .xsd  MS 结构文档（原子 + **完整键表** + 晶胞；实测 5596 原子 / 5891 键）
    #   .xtd  MS 轨迹文档（原子/键/晶胞都在，坐标在旁边的 .trj 里）
    ".car", ".xsd", ".xtd",
    # 压缩版（MDAnalysis 自己解压）
    ".pdb.gz", ".gro.gz", ".xyz.gz", ".tpr.gz",
})

# --------------------------------------------------------------- 轨迹（含纯坐标格式）
TRAJECTORY_SUFFIXES: frozenset[str] = frozenset({
    ".xtc", ".trr",                     # GROMACS
    ".dcd",                             # CHARMM/NAMD
    ".nc", ".ncdf",                     # AMBER NetCDF
    ".trj", ".trz",                     # .trj 有两个来源：CHARMM ASCII / Materials
                                        # Studio 二进制 —— 见 CONVERT_SUFFIXES 与
                                        # mdta.ms_trj（先按 CHARMM 试读，失败再转换）
    ".mdcrd", ".crd",                   # AMBER ASCII
    ".coor", ".inpcrd",                 # CHARMM/AMBER 单帧坐标
    ".lammpstrj", ".dump", ".lammpsdump",   # LAMMPS dump（前者需显式 format）
    ".tng",                             # GROMACS TNG
    ".h5md", ".gsd",                    # HDF5 系（自带拓扑）
    ".lammps", ".data", ".lmp", ".dat",     # LAMMPS data（也能当轨迹，单帧）
    ".gro", ".pdb", ".ent", ".xyz",     # 多帧文本格式
    ".dms", ".mol2",
    ".pdb.gz", ".gro.gz", ".xyz.gz",
})

#: 两个槽都能放的格式（界面上会同时出现在两个候选里）
BOTH_SUFFIXES: frozenset[str] = TOPOLOGY_SUFFIXES & TRAJECTORY_SUFFIXES

#: 界面文件选择器要列出来的全部后缀
ALL_SUFFIXES: frozenset[str] = TOPOLOGY_SUFFIXES | TRAJECTORY_SUFFIXES

#: 后缀 → 必须显式传给 MDAnalysis 的 ``format`` 名（否则按后缀找不到**坐标**读取器）
EXPLICIT_FORMAT: dict[str, str] = {
    ".lammpstrj": "LAMMPSDUMP",
    ".dump": "LAMMPSDUMP",
    ".lammpsdump": "LAMMPSDUMP",         # 自带 EXTENSIONS，显式指定只为稳
    # LAMMPS data：MDAnalysis 只认 `.data`/`.lammps` 后缀，且 `.lammps` 会被
    # **DCDReader** 抢走（实测报 "Failed to load … with parser"）—— 一并显式指定。
    ".lmp": "DATA",
    ".dat": "DATA",
    ".lammps": "DATA",
    ".data": "DATA",
}

#: 后缀 → 必须显式传给 MDAnalysis 的 ``topology_format``（否则按后缀找不到**拓扑**解析器）
#: `mdta.io` 会把它作为 `Universe(..., topology_format=...)` 传下去；
#: 轨迹一侧的 `format` 由 :data:`EXPLICIT_FORMAT` 单独负责（两者互不影响）。
EXPLICIT_TOPOLOGY_FORMAT: dict[str, str] = {
    ".lmp": "DATA",
    ".dat": "DATA",
    ".lammps": "DATA",
    ".data": "DATA",
}

#: **可能需要先转换**的格式：MDAnalysis 读不了时，用外部纯 Python 模块转成 XTC。
#: `.trj` 是双关：CHARMM 的 ASCII 轨迹 MDAnalysis 能读，而 **Materials Studio**
#: 的同名格式不能 —— 所以策略是"先让 MDAnalysis 试读，失败再转换"（见 :mod:`mdta.io`
#: 与 :mod:`mdta.ms_trj`）。
CONVERT_SUFFIXES: frozenset[str] = frozenset({".trj"})

#: 拓扑里没有键表、需要按坐标猜键的格式（见 :mod:`mdta.topology`）
BOND_GUESSABLE_SUFFIXES: frozenset[str] = frozenset({
    ".gro", ".pdb", ".ent", ".xyz", ".mol2", ".pdbqt", ".pqr",
    ".dms", ".data", ".lammps", ".lmp", ".dat", ".lammpsdump",
    ".gro.gz", ".pdb.gz", ".xyz.gz",
    # MS 的 .car 通常不带键表（BIOSYM archive 只在有 CONECT 时带）
    ".car",
})

#: **Materials Studio 自家格式**（MDAnalysis 读不了，由 :mod:`mdta.ms_structure` 解析）
MS_STRUCTURE_SUFFIXES: frozenset[str] = frozenset({".car", ".xsd", ".xtd"})

#: 「文档型」的 MS 文件：文件名与轨迹同名（`x.xtd` ↔ `x.trj`），当用户只给了其中
#: 一个时，可以按这个对应关系自动找另一个（见 :func:`mdta.io` 的自动配对）。
MS_DOCUMENT_SUFFIXES: frozenset[str] = frozenset({".xsd", ".xtd"})

#: 只有坐标、没有原子身份的格式（当拓扑用会丢掉质量/元素 —— QC 会标"不可引用"）
COORDINATE_ONLY_SUFFIXES: frozenset[str] = frozenset({
    ".xtc", ".trr", ".dcd", ".nc", ".ncdf", ".trj", ".trz",
    ".mdcrd", ".crd", ".coor", ".inpcrd", ".lammpstrj", ".dump",
    ".lammpsdump",
})


def suffix_of(path: str) -> str:
    """取小写后缀（含 ``.gz`` 两层，例如 ``sys.pdb.gz`` → ``.pdb.gz``）。

    ⚠️ 不能写成 `os.path.splitext` 一把梭：`sys.pdb.gz` 那样子串会只得到 `.gz`，
    于是压缩文件被判成"不认识的后缀"。这里显式剥掉 `.gz` 再取内层后缀。
    （这条是自检抓出来的：`assert suffix_of("a/b/sys.pdb.gz") == ".pdb.gz"`）
    """
    name = os.path.basename(str(path)).lower()
    if name.endswith(".gz"):
        return os.path.splitext(name[:-3])[1] + ".gz"
    return os.path.splitext(name)[1]


def classify(path: str) -> str | None:
    """``"topology"`` / ``"trajectory"`` / ``"both"``；不认识则 ``None``。"""
    suf = suffix_of(path)
    top = suf in TOPOLOGY_SUFFIXES
    traj = suf in TRAJECTORY_SUFFIXES
    if top and traj:
        return "both"
    if top:
        return "topology"
    if traj:
        return "trajectory"
    return None


def can_be_topology(path: str) -> bool:
    return suffix_of(path) in TOPOLOGY_SUFFIXES


def can_be_trajectory(path: str) -> bool:
    return suffix_of(path) in TRAJECTORY_SUFFIXES


def format_for(path: str) -> str | None:
    """需要显式 ``format``（坐标读取器）时返回格式名，否则 ``None``。"""
    return EXPLICIT_FORMAT.get(suffix_of(path))


def topology_format_for(path: str) -> str | None:
    """需要显式 ``topology_format``（拓扑解析器）时返回格式名，否则 ``None``。

    与 :func:`format_for` 分开：`Universe(拓扑, 轨迹, format=…)` 里的 ``format``
    是**轨迹**读取器，拓扑解析器要用 ``topology_format=`` 单独指定。LAMMPS 的
    `.lmp` / `.dat` 就属于这种情况（MDAnalysis 按后缀找不到 DATA 解析器）。
    """
    return EXPLICIT_TOPOLOGY_FORMAT.get(suffix_of(path))


def is_bond_guessable(path: str) -> bool:
    return suffix_of(path) in BOND_GUESSABLE_SUFFIXES


#: **知道、但本机读不了**的后缀 → 给用户的下一步建议。
#: 这些不进 `ALL_SUFFIXES`（不假装支持），但选到时报错要具体，不能只说"不认识"。
UNSUPPORTED_HINTS: dict[str, str] = {
    ".cfg": "AtomEye 的 `.cfg`（LAMMPS `dump ... cfg` 的输出）在本机 MDAnalysis 2.10 里"
            "**没有读取器**。请让 LAMMPS 改用 `dump ... custom` 输出 `.lammpstrj`"
            "（配合 data 文件），或用 `write_data` 导出 `.lmp` / `.data`。",
    ".restart": "LAMMPS 的二进制 restart 文件不能直接读（它是内部状态转储）。"
                "请在 LAMMPS 里用 `write_data` 导出 data 文件（`.lmp`/`.data`），"
                "或用 `dump` 输出轨迹（`.lammpstrj`）。",
    ".restart.*": "LAMMPS 的二进制 restart 文件不能直接读（它是内部状态转储）。"
                  "请用 `write_data` 导出 data 文件，或 `dump` 输出轨迹。",
    ".yaml": "LAMMPS 的 YAML dump（`dump ... yaml`）本机 MDAnalysis 读不了；"
             "请改用 `dump ... custom` 的 `.lammpstrj`。",
    ".in": "这是 LAMMPS **输入脚本**，不是结构/轨迹文件。"
           "脚本里 `read_data` 指到的 data 文件、`dump` 输出的轨迹才是数据文件。",
    ".log": "这是 LAMMPS 的**日志/热力学输出**，不含坐标。"
            "结构与轨迹请看 `write_data` 的 data 文件与 `dump` 的轨迹文件。",
    ".lammpsdump.gz": "压缩的 dump 先解压再用（本工具暂不解压 LAMMPS dump）。",
}


def unsupported_hint(path: str) -> str | None:
    """已知但读不了的后缀给出具体建议（`None` 表示"完全不认识"）。"""
    suf = suffix_of(path)
    if suf in UNSUPPORTED_HINTS:
        return UNSUPPORTED_HINTS[suf]
    name = os.path.basename(str(path)).lower()
    if name.startswith("restart") or ".restart" in name:
        return UNSUPPORTED_HINTS[".restart"]
    return None


#: LAMMPS 里 `.lammps` / `.lmp` / `.dat` 三种后缀**既可能是数据文件，也可能是
#: 输入脚本或日志**（`in.lammps`、`log.lammps` 很常见）。进解析器之前先"闻一下"内容。
LAMMPS_TEXT_SUFFIXES: frozenset[str] = frozenset({".lammps", ".lmp", ".dat", ".data"})


def sniff_lammps_kind(path: str) -> str | None:
    """猜一个 LAMMPS 文本文件是 ``"data"`` / ``"input"`` / ``"log"``（读前 40 行）。

    为什么要闻内容：`.lammps` 这个后缀在真实项目里**大量用于输入脚本/日志**
    （`in.lammps`、`log.lammps`），直接丢给 DATA 解析器只会得到
    "EOFError / Compressed file ended" 这类看不懂的报错。
    """
    if suffix_of(path) not in LAMMPS_TEXT_SUFFIXES:
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = "".join(fh.readline() for _ in range(40)).lower()
    except OSError:
        return None
    if not head.strip():
        return None
    if any(k in head for k in ("read_data", "pair_style", "compute ", "thermo_style",
                              "dump ", "fix ", "run ")):
        return "input"
    if head.lstrip().startswith("lammps (") or "loop time" in head or "step temp" in head:
        return "log"
    if "atoms" in head and "atom types" in head:
        return "data"
    return None


def is_ms_structure(path: str) -> bool:
    """是不是 Materials Studio 自家格式（``.car`` / ``.xsd`` / ``.xtd``）。"""
    return suffix_of(path) in MS_STRUCTURE_SUFFIXES


def ms_sibling_trajectory(topology: str) -> str | None:
    """MS 文档旁边的**同名轨迹**（`x.xsd`/`x.xtd` ↔ `x.trj` / `x.xtc` / `x.dcd`）。

    实测用户体系：`.xtd` 里写着 ``<Trajectory End="6001" Type="trj"/>``，坐标
    其实在同名的 ``.trj`` 里 —— 所以只给一个 ``.xtd`` 时应该能自动配上它。
    ⚠️ 只考虑**真轨迹**后缀：``.xsd`` / ``.xtd`` / ``.car`` 自己也是"拓扑"，
    拿它们当轨迹会被槽位校验挡下（实测报「.xtd 不能用作轨迹」）。
    """
    if not topology or not is_ms_structure(topology):
        return None
    base = os.path.splitext(str(topology))[0]
    for suf in (".trj", ".xtc", ".dcd", ".trr", ".trz", ".nc"):
        cand = base + suf
        if suffix_of(cand) in TRAJECTORY_SUFFIXES and os.path.isfile(cand):
            return cand
    return None


def _fmt(suffixes) -> str:
    return " ".join(sorted(suffixes))


def topology_help() -> str:
    """给用户的"可用拓扑格式"提示（错误信息与界面提示共用）。"""
    return _fmt(TOPOLOGY_SUFFIXES)


def trajectory_help() -> str:
    return _fmt(TRAJECTORY_SUFFIXES)


__all__ = [
    "ALL_SUFFIXES", "BOND_GUESSABLE_SUFFIXES", "BOTH_SUFFIXES",
    "COORDINATE_ONLY_SUFFIXES", "EXPLICIT_FORMAT", "EXPLICIT_TOPOLOGY_FORMAT",
    "MS_DOCUMENT_SUFFIXES", "MS_STRUCTURE_SUFFIXES", "TOPOLOGY_SUFFIXES",
    "TRAJECTORY_SUFFIXES", "UNSUPPORTED_HINTS", "can_be_topology",
    "can_be_trajectory", "classify", "format_for", "is_bond_guessable",
    "is_ms_structure", "ms_sibling_trajectory", "sniff_lammps_kind", "suffix_of",
    "topology_help", "topology_format_for", "trajectory_help", "unsupported_hint",
]
