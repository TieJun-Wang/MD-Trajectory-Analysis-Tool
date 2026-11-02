# -*- coding: utf-8 -*-
r"""
xtcwriter.py -- 纯 Python 的 GROMACS XTC 写入器
================================================

从 MS_Trajdump 自带的 xdrfile 源码（``MS_dump/xdrfile/xdrfile.c`` 的
``xdrfile_compress_coord_float`` 与 ``xdrfile_xtc.c``）逐行移植，
**不依赖任何 C 扩展、不需要编译**。

XTC 文件结构（每个帧）::

    int    magic = 1995
    int    natoms
    int    step
    float  time              (ps)
    float  box[9]            (nm, 行主序 3x3；正交盒子就是对角阵)
    int    size = natoms
    float  precision         (通常 1000，即 3 位小数)   ← 仅当 natoms > 9
    int    minint[3]                                    ← 仅当 natoms > 9
    int    maxint[3]                                    ← 仅当 natoms > 9
    int    smallidx                                     ← 仅当 natoms > 9
    int    nbytes            (压缩后字节数)              ← 仅当 natoms > 9
    opaque data[nbytes]      (压缩坐标，4 字节对齐)

坐标单位是 **nm**（GROMACS 约定），所以调用方要把 Å 除以 10。

算法要点
--------
1. 先把坐标乘 precision 并四舍五入成整数（C 里是 ``lf = x*prec ± 0.5`` 再截断）。
2. 记录每维的最小/最大值 minint/maxint，把坐标平移到 [0, sizeint)。
3. 若 sizeint 都 <= 0xffffff，用 ``sizeofints`` 算出一个"三合一"的位宽 bitsize，
   每个原子用 ``encodeints`` 把三个整数打包成一个大整数写进去；
   否则退化为按维各自的位宽 ``encodebits``。
4. 相邻原子如果三个方向差都小于 smallnum，就合并成一个 run，
   用 ``smallidx`` 位宽的小整数差分编码（这正是 XTC 压缩率的主要来源）。
5. 最后 ``buf2[0]`` 是字节数，紧随其后是数据。

校验方式：用自己的 writer 写出 .xtc，再用 MDAnalysis 读回来与原始 .trj 比对，
偏差应 <= 1e-3 Å（就是 precision=1000 的量化步长）。
"""

from __future__ import annotations

import struct
from typing import List, Optional, Sequence

import numpy as np

__all__ = ['XTCWriter', 'write_xtc_frame', 'compress_coord_float']

MAGIC = 1995
DIM = 3
INT_MAX = 2147483647
INT_MIN = -2147483648
UINT32 = 0xFFFFFFFF

# xdrfile.c 里的 magicints 表（原样抄下来，下标 0..71）
MAGICINTS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
]
FIRSTIDX = 9
LASTIDX = len(MAGICINTS)          # 72


# ---------------------------------------------------------------------------
# xdrfile.c 的辅助函数
# ---------------------------------------------------------------------------
def sizeofint(size: int) -> int:
    """表示 size 需要多少位（xdrfile.c: sizeofint）。"""
    num = 1
    n = 0
    while size >= num and n < 32:
        n += 1
        num <<= 1
    return n


def sizeofints(sizes: Sequence[int]) -> int:
    """把若干个小整数乘成一个大整数后需要多少位（xdrfile.c: sizeofints）。"""
    num_of_bytes = 1
    bytes_ = [1] + [0] * 32
    for s in sizes:
        tmp = 0
        bytecnt = 0
        while bytecnt < num_of_bytes:
            tmp = bytes_[bytecnt] * s + tmp
            bytes_[bytecnt] = tmp & 0xFF
            tmp >>= 8
            bytecnt += 1
        while tmp != 0:
            bytes_[bytecnt] = tmp & 0xFF
            tmp >>= 8
            bytecnt += 1
        num_of_bytes = bytecnt
    num_of_bits = 0
    n = 1
    num_of_bytes -= 1
    while bytes_[num_of_bytes] >= n:
        num_of_bits += 1
        n *= 2
    return num_of_bits + num_of_bytes * 8


class _BitBuffer:
    """对应 xdrfile.c 里的 ``int buf2[]``：

    ``buf[0]`` 已写字节数，``buf[1]`` 当前字节内已用位数，
    ``buf[2]`` 累积中的最后一个字节，``buf[3..]`` 是数据。
    """

    __slots__ = ('cnt', 'lastbits', 'lastbyte', 'data')

    def __init__(self) -> None:
        self.cnt = 0
        self.lastbits = 0
        self.lastbyte = 0
        self.data = bytearray()

    def _put(self, idx: int, val: int) -> None:
        """往字节缓冲区的 idx 位置写一个字节。

        C 里是 ``cbuf[cnt++] = ...``，即**覆盖**该位置；而 encodebits 结尾
        会预先写入"还没填满的那个字节"到 cbuf[cnt]，所以下一次写入必须
        覆盖它、而不是追加，否则位流就错位了。
        """
        if idx < len(self.data):
            self.data[idx] = val
        else:
            self.data.append(val)

    def encodebits(self, num_of_bits: int, num: int) -> None:
        """把 num 的低 num_of_bits 位追加进缓冲区（xdrfile.c: encodebits）。"""
        cnt, lastbits, lastbyte = self.cnt, self.lastbits, self.lastbyte
        if num_of_bits < 64:
            num &= (1 << num_of_bits) - 1
        while num_of_bits >= 8:
            lastbyte = ((lastbyte << 8) | ((num >> (num_of_bits - 8)) & 0xFF)) & UINT32
            self._put(cnt, (lastbyte >> lastbits) & 0xFF)
            cnt += 1
            num_of_bits -= 8
        if num_of_bits > 0:
            lastbyte = ((lastbyte << num_of_bits) | num) & UINT32
            lastbits += num_of_bits
            if lastbits >= 8:
                lastbits -= 8
                self._put(cnt, (lastbyte >> lastbits) & 0xFF)
                cnt += 1
        self.cnt, self.lastbits, self.lastbyte = cnt, lastbits, lastbyte
        if lastbits > 0:
            # 预先放好还没填满的那个字节（后续写入会覆盖它）
            self._put(cnt, (lastbyte << (8 - lastbits)) & 0xFF)

    def encodeints(self, num_of_bits: int, sizes: Sequence[int],
                   nums: Sequence[int]) -> None:
        """把若干小整数打包成大整数后写入（xdrfile.c: encodeints）。"""
        bytes_: List[int] = []
        tmp = nums[0]
        while True:
            bytes_.append(tmp & 0xFF)
            tmp >>= 8
            if tmp == 0:
                break
        num_of_bytes = len(bytes_)

        for i in range(1, len(nums)):
            if nums[i] >= sizes[i]:
                raise ValueError('encodeints: num %d >= size %d' % (nums[i], sizes[i]))
            tmp = nums[i]
            bytecnt = 0
            while bytecnt < num_of_bytes:
                tmp = bytes_[bytecnt] * sizes[i] + tmp
                bytes_[bytecnt] = tmp & 0xFF
                tmp >>= 8
                bytecnt += 1
            while tmp != 0:
                bytes_.append(tmp & 0xFF)
                tmp >>= 8
                bytecnt += 1
            num_of_bytes = bytecnt

        if num_of_bits >= num_of_bytes * 8:
            for i in range(num_of_bytes):
                self.encodebits(8, bytes_[i])
            self.encodebits(num_of_bits - num_of_bytes * 8, 0)
        else:
            for i in range(num_of_bytes - 1):
                self.encodebits(8, bytes_[i])
            self.encodebits(num_of_bits - (num_of_bytes - 1) * 8,
                            bytes_[num_of_bytes - 1])


def _round_to_int(x: np.ndarray, precision: float) -> np.ndarray:
    """对应 C 里的 ``lf = x*precision ± 0.5; lint = (int)lf;``（向零截断）。"""
    lf = np.where(x >= 0.0, x * precision + 0.5, x * precision - 0.5)
    return np.trunc(lf).astype(np.int64)


def compress_coord_float(x: np.ndarray, precision: float = 1000.0) -> bytes:
    """仅返回一帧坐标的 XTC payload（调试用；正常写文件请用 write_xtc_frame）。"""
    import io
    buf = io.BytesIO()
    write_xtc_frame(buf, x, [1, 0, 0, 0, 1, 0, 0, 0, 1], 0, 0.0, precision)
    raw = buf.getvalue()
    # 跳过 magic/natoms/step/time(16B) + box(36B) + size/prec/minint/maxint/smallidx(32B)
    return raw[84:]


# ---------------------------------------------------------------------------
# 帧级 / 文件级写入
# ---------------------------------------------------------------------------
def _xdr_opaque(data: bytes) -> bytes:
    """XDR opaque：4 字节长度 + 数据 + 补齐到 4 字节边界。"""
    pad = (-len(data)) % 4
    return struct.pack('>i', len(data)) + data + b'\x00' * pad


def write_xtc_frame(fh, xyz_nm: np.ndarray, box_nm: Sequence[float],
                    step: int, time_ps: float, precision: float = 1000.0) -> None:
    """写一帧。xyz_nm: (N,3) nm；box_nm: 9 个盒矢量分量（行主序，nm）。"""
    natoms = int(xyz_nm.shape[0])
    size = natoms                     # 后面沿用 C 源码里的变量名
    fh.write(struct.pack('>iiif', MAGIC, natoms, int(step), float(time_ps)))
    fh.write(struct.pack('>9f', *[float(v) for v in box_nm]))

    if natoms <= 9:
        fh.write(struct.pack('>i', natoms))
        fh.write(np.ascontiguousarray(xyz_nm, dtype='>f4').tobytes())
        return

    prec = 1000.0 if precision <= 0 else float(precision)
    lint = _round_to_int(xyz_nm, prec)
    minint = lint.min(axis=0)
    maxint = lint.max(axis=0)
    sizeint = (maxint - minint + 1)

    if int(sizeint[0] | sizeint[1] | sizeint[2]) > 0xFFFFFF:
        bitsizeint = [sizeofint(int(s)) for s in sizeint]
        bitsize = 0
    else:
        bitsizeint = [0, 0, 0]
        bitsize = sizeofints([int(s) for s in sizeint])

    if natoms > 1:
        d = np.abs(np.diff(lint, axis=0)).sum(axis=1)
        mindiff = int(d[1:].min()) if natoms > 2 else int(d[0])
    else:
        mindiff = INT_MAX
    smallidx = FIRSTIDX
    while smallidx < LASTIDX and MAGICINTS[smallidx] < mindiff:
        smallidx += 1
    smallidx0 = smallidx      # 头部要写「初始」smallidx，循环里它会变（C 里也是先写）

    maxidx = min(LASTIDX, smallidx + 8)
    minidx = maxidx - 8
    smaller = MAGICINTS[max(FIRSTIDX, smallidx - 1)] // 2
    smallnum = MAGICINTS[smallidx] // 2
    sizesmall = [MAGICINTS[smallidx]] * 3
    larger = MAGICINTS[maxidx] // 2

    # 注意：C 里 buf1 存的是「原始」整数值，平移 minint 是在写每个原子时才做的
    # （tmpcoord[0] = thiscoord[0] - minint[0]），所以这里不能提前减。
    coords = lint.tolist()
    bb = _BitBuffer()
    prevrun = -1
    prevcoord = [0, 0, 0]
    i = 0
    while i < size:
        thiscoord = coords[i]

        if smallidx < maxidx and i >= 1 and \
                abs(thiscoord[0] - prevcoord[0]) < larger and \
                abs(thiscoord[1] - prevcoord[1]) < larger and \
                abs(thiscoord[2] - prevcoord[2]) < larger:
            is_smaller = 1
        elif smallidx > minidx:
            is_smaller = -1
        else:
            is_smaller = 0

        is_small = 0
        if i + 1 < size:
            nxt = coords[i + 1]
            if abs(thiscoord[0] - nxt[0]) < smallnum and \
               abs(thiscoord[1] - nxt[1]) < smallnum and \
               abs(thiscoord[2] - nxt[2]) < smallnum:
                coords[i] = nxt
                coords[i + 1] = thiscoord
                thiscoord = nxt
                is_small = 1

        tmp = [thiscoord[0] - int(minint[0]),
               thiscoord[1] - int(minint[1]),
               thiscoord[2] - int(minint[2])]
        if bitsize == 0:
            bb.encodebits(bitsizeint[0], tmp[0])
            bb.encodebits(bitsizeint[1], tmp[1])
            bb.encodebits(bitsizeint[2], tmp[2])
        else:
            bb.encodeints(bitsize, sizeint, tmp)

        prevcoord = thiscoord
        i += 1

        run = 0
        if is_small == 0 and is_smaller == -1:
            is_smaller = 0
        runbuf = []
        while is_small and run < 8 * 3:
            c = coords[i]
            tmpsum = ((c[0] - prevcoord[0]) ** 2 + (c[1] - prevcoord[1]) ** 2
                      + (c[2] - prevcoord[2]) ** 2)
            if is_smaller == -1 and tmpsum >= smaller * smaller:
                is_smaller = 0
            runbuf.append(c[0] - prevcoord[0] + smallnum)
            runbuf.append(c[1] - prevcoord[1] + smallnum)
            runbuf.append(c[2] - prevcoord[2] + smallnum)
            run += 3
            prevcoord = c
            i += 1
            is_small = 0
            if i < size:
                nc = coords[i]
                if abs(nc[0] - prevcoord[0]) < smallnum and \
                   abs(nc[1] - prevcoord[1]) < smallnum and \
                   abs(nc[2] - prevcoord[2]) < smallnum:
                    is_small = 1

        if run != prevrun or is_smaller != 0:
            prevrun = run
            bb.encodebits(1, 1)
            bb.encodebits(5, run + is_smaller + 1)
        else:
            bb.encodebits(1, 0)

        for k in range(0, run, 3):
            bb.encodeints(smallidx, sizesmall, runbuf[k:k + 3])

        if is_smaller != 0:
            smallidx += is_smaller
            if is_smaller < 0:
                smallnum = smaller
                smaller = MAGICINTS[smallidx - 1] // 2
            else:
                smaller = smallnum
                smallnum = MAGICINTS[smallidx] // 2
            sizesmall = [MAGICINTS[smallidx]] * 3

    if bb.lastbits != 0:
        bb.cnt += 1
    payload = bytes(bb.data[:bb.cnt])

    fh.write(struct.pack('>i', natoms))
    fh.write(struct.pack('>f', prec))
    fh.write(struct.pack('>3i', int(minint[0]), int(minint[1]), int(minint[2])))
    fh.write(struct.pack('>3i', int(maxint[0]), int(maxint[1]), int(maxint[2])))
    fh.write(struct.pack('>i', smallidx0))
    fh.write(_xdr_opaque(payload))


class XTCWriter:
    """纯 Python XTC 写入器。

        with XTCWriter('out.xtc') as w:
            w.write(xyz_angstrom, box_angstrom, step=0, time=0.0)

    坐标/盒子单位是 **Å**（内部转 nm），与 MDAnalysis 的用法一致。
    """

    def __init__(self, filename, precision: float = 1000.0):
        self.filename = filename
        self.precision = precision
        self.fh = open(filename, 'wb')
        self.n_frames = 0

    def write(self, xyz_ang: np.ndarray, box_ang: Sequence[float],
              step: int = 0, time: float = 0.0) -> None:
        box = np.asarray(box_ang, dtype=np.float64).reshape(-1)
        if box.size == 3:                       # 只给了 a,b,c -> 正交盒子
            box9 = np.array([box[0], 0, 0, 0, box[1], 0, 0, 0, box[2]])
        elif box.size == 6:                     # a,b,c,alpha,beta,gamma
            a, b, c, al, be, ga = box
            al, be, ga = np.radians([al, be, ga])
            cx = c * np.cos(be)
            cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
            cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
            box9 = np.array([a, 0, 0,
                             b * np.cos(ga), b * np.sin(ga), 0,
                             cx, cy, cz])
        else:
            box9 = box
        write_xtc_frame(self.fh, np.asarray(xyz_ang, dtype=np.float64) / 10.0,
                        box9 / 10.0, step, time, self.precision)
        self.n_frames += 1

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()
            self.fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
