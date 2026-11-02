# -*- coding: utf-8 -*-
r"""
msdump.py -- MS_Trajdump 的纯 Python 版本
==========================================

MS_Trajdump 是个 C++ 程序（依赖 Eigen + xdrfile，需要编译）；它的 Python API
（PyMSDump）也只是 C++ 扩展的壳，一样要编译。**这个文件不需要编译**：
.trj 解析器 + XTC 写入器全是纯 Python。

用法::

    # .trj -> .xtc（最常用）
    python msdump.py -f 122C12-450K.trj -s 122C12-450K.pdb -o out.xtc

    # 输出格式由扩展名决定
    python msdump.py -f sys.trj -s sys.pdb -o traj.xyz     # 扩展 xyz（Ovito 可直接读）
    python msdump.py -f sys.trj -s sys.pdb -o traj.gro
    python msdump.py -f sys.trj -o energy.txt              # 逐帧能量
    python msdump.py -f sys.trj -o energy.xvg              # gromacs xvg

    # 只转前 N 帧 / 隔 N 帧取一帧
    python msdump.py -f sys.trj -s sys.pdb -o out.xtc -n 100 --stride 10

    # 帧源也可以是 dcd/xtc（.trj 丢了也能用）
    python msdump.py -f 122C12-450K.dcd -s 122C12-450K.pdb -o out.xtc

对应关系（MS_dump 的选项基本兼容）::

    -f/--file   轨迹文件（.trj / .dcd / .xtc）
    -s/--struct 结构文件（.pdb / .xsd / .xtd），用于原子名、键、晶胞
    -o/--out    输出文件；扩展名决定格式：xtc / xyz / gro / txt / xvg
    -n/--nframes 最多转多少帧
    --stride    每隔多少帧取一帧

与 C++ 版的差异
----------------
* 只输出坐标类格式（xtc/xyz/gro）和能量类（txt/xvg）。**.trr 没做** ——
  TRR 本身不压缩、实现很简单，但需要 5000+ 个原子 × 6 个数组的逐帧写入，
  这里没排进去；需要的话可以用 MDAnalysis 的 TRRWriter。
* 不做 PDB 的 PBC 重排（C++ 版会为了对齐 MS 导出的 PDB 做盒子变换）。
  本工具直接把 .trj 里存的坐标写出去 —— 本体系 MS 的 PDB 和 .trj 本身就是
  一致的正交坐标（已用首/末帧校验到 6e-14 Å），所以不需要变换。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'md_convert'))

from xtcwriter import XTCWriter                      # noqa: E402

# ---------------------------------------------------------------------------
# 帧源：.trj 走自己的解析器；.dcd/.xtc 走 MDAnalysis
# ---------------------------------------------------------------------------
class TrjSource:
    """把 .trj 包成统一的「逐帧迭代」接口。"""

    def __init__(self, path: str, **kw):
        from ms_trj import MSTrajectory
        self._t = MSTrajectory(path)
        self.n_atoms = self._t.n_atoms
        self.n_frames = self._t.n_frames
        self._gen = None

    def frames(self, stride: int = 1, stop: Optional[int] = None):
        stop = self.n_frames if stop is None else min(stop, self.n_frames)
        for fr in self._t.iter_frames(stop=stop, step=stride):
            th = dict(fr.thermo)
            th['_P'] = fr.pressure          # GPa
            th['_V'] = fr.volume            # Å^3
            yield fr.index, fr.step, fr.time, fr.xyz, fr.vel, fr.cell[:3], th

    def close(self):
        self._t.close()


class UASource:
    """把 .dcd/.xtc 包成同样的接口。"""

    def __init__(self, path: str, struct: Optional[str] = None, **kw):
        from ms_io import load_universe
        top = struct or _auto_topology(path)
        self._u = load_universe(top, path) if top else load_universe(path)
        self.n_atoms = len(self._u.atoms)
        self.n_frames = self._u.trajectory.n_frames
        self._top = top

    def frames(self, stride: int = 1, stop: Optional[int] = None):
        n = self.n_frames if stop is None else min(stop, self.n_frames)
        names = None
        for i in range(0, n, stride):
            self._u.trajectory[i]
            ts = self._u.trajectory.ts
            d = ts.dimensions if ts.dimensions is not None else (0, 0, 0, 90, 90, 90)
            yield i, i, float(ts.time), np.array(ts.positions, dtype=np.float64), \
                None, np.array(d[:3], dtype=np.float64), {}

    def close(self):
        pass


def _read_pdb_names(path: str):
    """纯 Python 读 PDB 的原子名/残基名/残基号（不依赖 MDAnalysis，中文路径也行）。"""
    names, resnames, resids = [], [], []
    with open(path, 'r', encoding='latin1') as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            names.append(line[12:16].strip())
            resnames.append(line[17:20].strip() or 'MOL')
            try:
                resids.append(int(line[22:26]))
            except ValueError:
                resids.append(1)
    return names, resnames, resids


def _auto_topology(traj: str) -> Optional[str]:
    d, base = os.path.split(traj)
    stem = base[:base.rindex('.')] if '.' in base else base
    for ext in ('.pdb', '.psf', '.gro'):
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def open_source(path: str, struct: Optional[str] = None):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.trj':
        return TrjSource(path)
    if ext in ('.dcd', '.xtc', '.trr', '.gro', '.pdb'):
        return UASource(path, struct)
    raise SystemExit('不认识的轨迹格式: %s（支持 .trj/.dcd/.xtc）' % ext)


# ---------------------------------------------------------------------------
# 各种输出格式
# ---------------------------------------------------------------------------
def write_xyz(fh, xyz, box, step, time, names):
    """扩展 XYZ：注释行带 Lattice=，Ovito 可以直接读盒子。"""
    n = xyz.shape[0]
    fh.write('%d\n' % n)
    fh.write('step=%d time=%.4f Lattice="%.6f 0.0 0.0 0.0 %.6f 0.0 0.0 0.0 %.6f" '
             'Properties=species:S:1:pos:R:3\n'
             % (step, time, box[0], box[1], box[2]))
    fh.write(''.join('%s %.6f %.6f %.6f\n' % (names[i], xyz[i, 0], xyz[i, 1], xyz[i, 2])
                     for i in range(n)))


def write_gro(fh, xyz, box, step, time, names, resnames, resids):
    """GROMACS .gro（坐标单位 nm）。"""
    n = xyz.shape[0]
    fh.write('step %d, t= %.4f\n%5d\n' % (step, time, n))
    x = xyz / 10.0
    for i in range(n):
        nm = names[i]
        fh.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'
                 % (resids[i] % 100000, resnames[i][:5], nm[:5], (i + 1) % 100000,
                    x[i, 0], x[i, 1], x[i, 2]))
    fh.write('%10.5f%10.5f%10.5f\n' % (box[0] / 10, box[1] / 10, box[2] / 10))


def write_energy(fh, k, step, time, thermo, xvg=False):
    """能量：每行 时间 温度 势能 动能 总能量 压力 体积（kJ/mol、bar）。"""
    if k == 0 and not xvg:
        fh.write('#Time(ps) Temperature(K) Potential(kJ/mol) Kinetic(kJ/mol) '
                 'TotalEnergy(kJ/mol) Pressure(bar) Volume(A^3)\n')
    if k == 0 and xvg:
        fh.write('# gromacs xvg, energies converted to kJ/mol, pressure to bar\n')
        fh.write('@    title "MS trajectory energy"\n@    xaxis  label "Time (ps)"\n'
                 '@    yaxis  label "E (kJ/mol)"\n@TYPE xy\n')
    KC = 4.184                       # kcal/mol -> kJ/mol
    t = thermo
    row = [time, t.get('T_inst', np.nan), t.get('Epot', np.nan) * KC,
           t.get('Ekin', np.nan) * KC, t.get('Etot', np.nan) * KC,
           thermo.get('_P', np.nan) * 10000.0, thermo.get('_V', np.nan)]
    if xvg:
        fh.write(' '.join('%.6g' % v for v in row) + '\n')
    else:
        fh.write(' '.join('%.6f' % v for v in row) + '\n')


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description='MS_Trajdump 的纯 Python 版本（.trj/.dcd/.xtc -> xtc/xyz/gro/txt/xvg）',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('-f', '--file', required=True, help='轨迹文件 .trj/.dcd/.xtc')
    ap.add_argument('-s', '--struct', default=None, help='结构文件 .pdb/.xsd/.xtd')
    ap.add_argument('-o', '--out', default=None,
                    help='输出文件；扩展名决定格式 (xtc/xyz/gro/txt/xvg)。默认 MS_traj.xyz')
    ap.add_argument('-n', '--nframes', type=int, default=0, help='最多转多少帧')
    ap.add_argument('--stride', type=int, default=1, help='每隔多少帧取一帧')
    ap.add_argument('--precision', type=float, default=1000.0,
                    help='XTC 精度（默认 1000，即 3 位小数，单位 nm）')
    args = ap.parse_args(argv)

    src_path = os.path.abspath(args.file)
    if not os.path.exists(src_path):
        raise SystemExit('找不到轨迹文件: %s' % src_path)
    out_path = os.path.abspath(args.out or 'MS_traj.xyz')
    fmt = os.path.splitext(out_path)[1].lower().lstrip('.')
    if fmt not in ('xtc', 'xyz', 'gro', 'txt', 'xvg'):
        raise SystemExit('不支持的输出格式 .%s（支持 xtc/xyz/gro/txt/xvg）' % fmt)

    # 原子名 / 残基号：优先从结构文件取，没有就用元素符号占位
    names = resnames = resids = None
    if args.struct:
        sp = os.path.abspath(args.struct)
        ext = os.path.splitext(sp)[1].lower()
        if ext in ('.xsd', '.xtd'):
            from ms_xtd import read_ms_structure
            st = read_ms_structure(sp)
            names = [a.name for a in st.atoms]
            mol_of = [1] * st.n_atoms
            for mi, mol in enumerate(st.molecules, start=1):
                for ai in mol:
                    mol_of[ai] = mi
            resids = mol_of
            resnames = ['MOL'] * st.n_atoms
            box_struct = np.array(st.cell.lengths)
        else:
            names, resnames, resids = _read_pdb_names(sp)
            box_struct = None
    else:
        box_struct = None

    t0 = time.time()
    src = open_source(src_path, args.struct)
    if names is None:
        names = ['C'] * src.n_atoms
        resnames = ['MOL'] * src.n_atoms
        resids = [1] * src.n_atoms
        print('未提供结构文件：原子名一律用 C（和 MS_dump 的行为一致）')

    n_done = 0
    cfh = None
    try:
        if fmt == 'xtc':
            cfh = XTCWriter(out_path, precision=args.precision)
        elif fmt in ('xyz', 'gro', 'txt', 'xvg'):
            cfh = open(out_path, 'w', encoding='ascii', newline='\n')

        for k, step, tm, xyz, vel, box, thermo in src.frames(
                stride=args.stride, stop=args.nframes or None):
            if cfh is None:
                break
            if fmt == 'xtc':
                cfh.write(xyz, box, step=step, time=tm)
            elif fmt == 'xyz':
                write_xyz(cfh, xyz, box, step, tm, names)
            elif fmt == 'gro':
                write_gro(cfh, xyz, box, step, tm, names, resnames, resids)
            else:
                if '_P' not in thermo:
                    thermo = dict(thermo)
                    pv = src  # 只有 .trj 源才有 p/v，dcd 源没有
                write_energy(cfh, n_done, step, tm, thermo, xvg=(fmt == 'xvg'))
            n_done += 1
            if n_done % 500 == 0 or n_done == 1:
                el = time.time() - t0
                print('  %5d 帧  %.2f 秒  (%.0f 帧/s)' % (n_done, el, n_done / el),
                      flush=True)
    finally:
        if cfh is not None:
            cfh.close()
        src.close()

    dt = time.time() - t0
    print('\n完成：%s' % out_path)
    print('  帧数 %d，原子 %d，用时 %.1f 秒（%.1f 帧/s）'
          % (n_done, src.n_atoms, dt, n_done / max(dt, 1e-9)))
    print('  文件 %.2f MB（每帧 %.1f KB）'
          % (os.path.getsize(out_path) / 1048576,
             os.path.getsize(out_path) / 1024 / max(n_done, 1)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
