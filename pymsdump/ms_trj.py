# -*- coding: utf-8 -*-
"""
ms_trj.py -- Materials Studio / Forcite .trj 轨迹读取器（纯 numpy，无需 Materials Studio）
==========================================================================================

.trj 格式（通过逆向 + 能量校验得到，详见 md_convert/README.md）
----------------------------------------------------------------
整个文件是大端（big-endian）Fortran 无格式记录序列：

    [int32 长度][payload][int32 长度]

文件头（记录 1..8）:
    1  "MDTR" + int32 版本号(2010) + 补零
    2  "Created by Materials Studio Forcite 23.1"
    3  "Molecular mechanics Forcite calculation"
    4  32 B  杂项
    5  int32 类型=1, int32 原子数, int32 原子数, 8 字节体系名 ("MStudio ")
    6  int32 原子数, 然后 原子数 个 int32 = 1,2,3,...,N   (原子编号表)
    7  int32 7, "NOTITLE"
    8  int32 5, "NOPAR"

之后是逐帧数据。每帧 = 5 个元数据记录 + 6 个长度 8*N 的 double 数组：

    元数据 (共 952 B):
      492 B : [float64 时间(ps)][int32 步数][61×float64 温度/能量...]
      96  B : [float64 P_inst, V, 0,0,0,0][float64 P_avg, V, 0,0,0,0]
      32  B : 保留字段
      176 B : [0,0, a, b, c, cos α, cos β, cos γ, ...]   晶胞
      116 B : [int32 0][14×float64 保留字段]
    数组:
      1,2,3 : x, y, z   (Å, 未包裹／分子保持完整)
      4,5,6 : vx, vy, vz (Å/ps)

原子顺序 = XSD/XTD 中 primary Atom3d 的 UserID 顺序（1..N）。
坐标换算：XSD 的 XYZ 是分数坐标，cart = (fx*a, fy*b, fz*c)。

校验（04_validate.py）：
  * 第 1 帧坐标 == .xtd 分数坐标 × (a,b,c)，最大偏差 2.4e-4 Å
  * 第 1 帧数组 4-6 动能 == 122C12-450K.txt 的 7504.925 kcal/mol（完全一致）
  * 第 1/6001 帧的温度/压力/势能 == 122C12-450K.txt 与 Status.txt
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = ['MSTrajectory', 'Frame', 'THERMO_KEYS']

BIG_INT = struct.Struct('>i')
BIG_DBL = struct.Struct('>d')

# 492 B 记录里 61 个 double 的语义（由帧 1 / 帧 6001 与 .txt、Status.txt 对齐得到）
_THERMO_NAMES = [
    'T_inst', 'T_avg', 'dt', 'T_target', 'T_inst2',
    'Epot', 'Ebond', 'Eangle', 'Etorsion', 'Einversion', 'Evdw', 'Eelec',
    '_r12', '_r13', '_r14', '_r15',
    'Ecross', 'Ebend_bend', 'Estretch_stretch', 'Estretch_bond_stretch',
    'Estretch_torsion_stretch', 'Ebend_torsion_bend', 'Etorsion_bend_bend',
    'Eseparated_stretch_stretch', 'Etorsion_stretch', '_r25', '_r26',
    'Evalence', 'Enonbond',
    # 以下为同一批量的“运行平均”
    'Epot_avg', 'Ebond_avg', 'Eangle_avg', 'Etorsion_avg', 'Einversion_avg',
    'Evdw_avg', 'Eelec_avg', '_r36', '_r37', '_r38', '_r39',
    'Ecross_avg', 'Ebend_bend_avg', 'Estretch_stretch_avg',
    'Estretch_bond_stretch_avg', 'Estretch_torsion_stretch_avg',
    'Ebend_torsion_bend_avg', 'Etorsion_bend_bend_avg',
    'Eseparated_stretch_stretch_avg', 'Etorsion_stretch_avg', '_r50', '_r51',
    'Evalence_avg', 'Enonbond_avg',
    'Etot', 'Ekin', 'Etot_avg', 'Ekin_avg',
    '_r57', '_r58', '_r59', '_r60',
]
THERMO_KEYS = tuple(k for k in _THERMO_NAMES if not k.startswith('_'))


@dataclass
class Frame:
    """一帧轨迹。"""
    index: int                      # 0 基帧号
    step: int                       # Forcite 步数
    time: float                     # ps
    cell: Tuple[float, ...] = ()    # a, b, c, alpha, beta, gamma
    xyz: Optional[np.ndarray] = None      # (N,3) float64, Å
    vel: Optional[np.ndarray] = None      # (N,3) float64, Å/ps
    pressure: float = float('nan')
    pressure_avg: float = float('nan')
    volume: float = float('nan')
    thermo: Dict[str, float] = field(default_factory=dict)

    @property
    def temperature(self) -> float:
        return self.thermo.get('T_inst', float('nan'))


class MSTrajectory:
    """流式读取 Forcite .trj。

    示例
    ----
    >>> trj = MSTrajectory('122C12-450K.trj')
    >>> trj.n_atoms, trj.n_frames
    (5596, 6001)
    >>> for fr in trj:            # 顺序流式，内存占用恒定
    ...     if fr.index % 1000 == 0:
    ...         print(fr.index, fr.time, fr.temperature)
    >>> fr = trj.frame(3000)      # 随机访问
    >>> fr.xyz.shape
    (5596, 3)
    """

    def __init__(self, path: str, verbose: bool = False):
        self.path = path
        self.size = os.path.getsize(path)
        self._fh = open(path, 'rb')
        self.verbose = verbose
        self._parse_header()

    # ------------------------------------------------------------------ 头
    def _rec(self, off: int) -> Tuple[int, bytes]:
        self._fh.seek(off)
        raw = self._fh.read(4)
        (ln,) = BIG_INT.unpack(raw)
        data = self._fh.read(ln)
        return ln, data

    def _parse_header(self) -> None:
        off = 0
        recs = []
        magic = None
        while off < self.size:
            ln, data = self._rec(off)
            off0 = off
            off += 8 + ln
            if not recs and data[:4] == b'MDTR':
                magic = data[:8]
            recs.append((off0, ln, data))
            if len(recs) == 5:
                # rec5: int32 1, int32 N, int32 N, name[8]
                self.n_atoms = BIG_INT.unpack(data[4:8])[0]
                self.system_name = data[12:].decode('latin1').strip()
                self._target_array_len = 8 * self.n_atoms
            if magic is None:
                raise ValueError('不是 Materials Studio .trj 文件（缺少 MDTR 魔数）')
            if 'n_atoms' in self.__dict__ and ln == self._target_array_len:
                break
        if 'n_atoms' not in self.__dict__:
            raise ValueError('未找到轨迹原子数记录')
        self.version = BIG_INT.unpack(recs[0][2][4:8])[0]
        # 原子编号表
        _, _, idrec = recs[5]
        self.atom_ids = np.frombuffer(idrec[4:4 + 4 * self.n_atoms], dtype='>i4').astype(np.int64)

        # 帧 1 的 6 个数组紧跟在元数据之后；倒数 5 条记录即每帧的元数据模版
        self._first_array_off = off0
        meta_recs = recs[-6:-1]                 # 492, 96, 32, 176, 116
        self._meta_offsets = [r[0] for r in meta_recs]
        self._meta_lens = [r[1] for r in meta_recs]
        self.meta_size = sum(8 + l for l in self._meta_lens)
        self._arr_len = self._target_array_len
        self._arr_rec = 8 + self._arr_len        # 单个数组记录占用的总字节
        self._block1 = 6 * self._arr_rec         # 帧 1 的 6 个数组
        self._stride = self.meta_size + self._block1
        self._meta0 = self._first_array_off - self.meta_size   # 帧 1 的元数据偏移

        rest = self.size - self._meta0
        if rest % self._stride:
            raise ValueError('文件长度与帧步长不匹配（%d %% %d != 0）'
                             % (rest, self._stride))
        self.n_frames = rest // self._stride

    # --------------------------------------------------------------- 元数据
    def _read_meta(self, k: int) -> Tuple[int, Frame]:
        off = self._meta0 + k * self._stride
        fh = self._fh
        fh.seek(off)
        buf = fh.read(self.meta_size)
        pos = 0
        payloads = []
        for ln in self._meta_lens:
            (l2,) = BIG_INT.unpack_from(buf, pos)
            payloads.append(buf[pos + 4:pos + 4 + l2])
            pos += 8 + l2

        main, pv, _x32, cellrec, _x116 = payloads
        time = BIG_DBL.unpack_from(main, 0)[0]
        (step,) = BIG_INT.unpack_from(main, 8)
        vals = np.frombuffer(main[12:12 + 8 * 61], dtype='>f8')
        thermo = {name: float(vals[i]) for i, name in enumerate(_THERMO_NAMES)
                  if not name.startswith('_') and i < vals.size}

        pv8 = np.frombuffer(pv, dtype='>f8')
        cell8 = np.frombuffer(cellrec, dtype='>f8')
        a, b, c = float(cell8[2]), float(cell8[3]), float(cell8[4])
        ca, cb, cg = float(cell8[5]), float(cell8[6]), float(cell8[7])
        ang = tuple(float(np.degrees(np.arccos(min(1.0, max(-1.0, x)))))
                    for x in (ca, cb, cg))

        fr = Frame(index=k, step=step, time=time, cell=(a, b, c) + ang,
                   pressure=float(pv8[0]), volume=float(pv8[1]),
                   pressure_avg=float(pv8[6]), thermo=thermo)
        return off, fr

    def meta(self, k: int) -> Frame:
        """只读元数据（热力学量、晶胞、时间），不读坐标。"""
        return self._read_meta(k)[1]

    def frame(self, k: int, xyz: bool = True, vel: bool = False) -> Frame:
        """随机访问第 k 帧（0 基）。"""
        if not 0 <= k < self.n_frames:
            raise IndexError('帧号越界: %d (0..%d)' % (k, self.n_frames - 1))
        _, fr = self._read_meta(k)
        if xyz or vel:
            arr = self._arrays_from_raw(self._read_arrays(k))
            if xyz:
                fr.xyz = np.ascontiguousarray(arr[:3].T)
            if vel:
                fr.vel = np.ascontiguousarray(arr[3:].T)
        return fr

    def _read_arrays(self, k: int) -> bytes:
        start = self._meta0 + k * self._stride + self.meta_size
        self._fh.seek(start)
        return self._fh.read(self._block1)

    # ---------------------------------------------------------------- 迭代
    def __iter__(self) -> Iterator[Frame]:
        return self.iter_frames()

    def iter_frames(self, start: int = 0, stop: Optional[int] = None,
                    step: int = 1, vel: bool = False) -> Iterator[Frame]:
        """顺序流式迭代（每帧顺序读 ~270 KB，速度最快，内存恒定）。

        产出的 Frame.xyz / Frame.vel 是**复用缓冲区**上的视图，
        需要留存请自行 .copy()。
        """
        if stop is None:
            stop = self.n_frames
        fh = self._fh
        for k in range(start, stop, step):
            off, fr = self._read_meta(k)
            fh.seek(off + self.meta_size)
            raw = fh.read(self._block1)
            arr = self._arrays_from_raw(raw)
            fr.xyz = arr[:3].T
            fr.vel = arr[3:].T if vel else None
            yield fr

    def _arrays_from_raw(self, raw: bytes) -> np.ndarray:
        """从一整块帧数据里取出 6 个数组（跳过每条记录的长度前缀）。

        返回 (6, N) 的 float64 数组：行 0-2 = x,y,z；行 3-5 = vx,vy,vz。
        """
        out = np.empty((6, self.n_atoms), dtype='>f8')
        for i in range(6):
            out[i] = np.frombuffer(raw, dtype='>f8', count=self.n_atoms,
                                   offset=i * self._arr_rec + 4)
        return out

    # ---------------------------------------------------------------- 其它
    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        return ('<MSTrajectory %s: %d frames, %d atoms, %.3f ns, stride=%d steps>'
                % (os.path.basename(self.path), self.n_frames, self.n_atoms,
                   self.n_frames and self.meta(self.n_frames - 1).time / 1000.0,
                   self.meta(1).step if self.n_frames > 1 else 0))


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '122C12-450K.trj')
    with MSTrajectory(p) as trj:
        print(trj)
        print('原子编号表: 1..%d, 名称表 = %s' % (trj.atom_ids[-1], trj.system_name))
        print('元数据记录长度:', trj._meta_lens, ' 帧步长:', trj._stride, 'B')
        for k in (0, 1, 3000, trj.n_frames - 1):
            fr = trj.frame(k)
            print('frame %5d  step=%8d  t=%8.2f ps  T=%8.3f K  P=%8.4f GPa  '
                  'Epot=%10.3f  Etot=%10.3f  cell=%.3f/%.3f/%.3f'
                  % (fr.index, fr.step, fr.time, fr.temperature, fr.pressure,
                     fr.thermo['Epot'], fr.thermo['Etot'],
                     fr.cell[0], fr.cell[1], fr.cell[2]))
            print('           xyz[0]=', np.round(fr.xyz[0], 5),
                  ' vel[0]=', np.round(fr.vel[0], 5) if fr.vel is not None else None)
