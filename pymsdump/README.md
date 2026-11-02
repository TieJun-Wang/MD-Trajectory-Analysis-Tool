# pymsdump —— MS_Trajdump 的纯 Python 版本

不用编译。`.trj` 解析器和 **XTC 写入器**都是纯 Python（numpy），
把 Materials Studio 的 `.trj` 轨迹导出成 XTC / XYZ / GRO / 能量表。

上游项目：[liuyujie714/MS_Trajdump](https://github.com/liuyujie714/MS_Trajdump)（C++，依赖 Eigen + xdrfile）。
它自带的 `PyMSDump` 只是 C++ 扩展（`PyMSDump_`）的壳，**仍然需要编译**；
这个目录是独立重写，不依赖任何 C 扩展。

---

## 1. 用法

```powershell
cd trj2xtc\pymsdump

# .trj -> .xtc（最常用）
python msdump.py -f ..\122C12-450K.trj -s ..\..\122C12-450K.pdb -o out.xtc

# 输出格式由扩展名决定
python msdump.py -f ..\122C12-450K.trj -s ..\..\122C12-450K.pdb -o traj.xyz   # 扩展 xyz，Ovito 直接读
python msdump.py -f ..\122C12-450K.trj -s ..\..\122C12-450K.pdb -o traj.gro
python msdump.py -f ..\122C12-450K.trj -o energy.txt                          # 逐帧能量
python msdump.py -f ..\122C12-450K.trj -o energy.xvg                          # gromacs xvg

# 只转前 100 帧 / 隔 10 帧取一帧
python msdump.py -f ..\122C12-450K.trj -s ..\..\122C12-450K.pdb -o out.xtc -n 100 --stride 10

# 帧源也可以是 dcd/xtc（.trj 不在时）
python msdump.py -f ..\..\122C12-450K.dcd -s ..\..\122C12-450K.pdb -o out.xtc
```

| 选项 | 说明 |
|---|---|
| `-f/--file` | 轨迹：`.trj`（自己解析）/ `.dcd` / `.xtc`（走 MDAnalysis） |
| `-s/--struct` | 结构：`.pdb`（纯 Python 读）/ `.xsd` / `.xtd`；不给则原子名一律 `C` |
| `-o/--out` | 输出文件，**扩展名决定格式**：`xtc` / `xyz` / `gro` / `txt` / `xvg` |
| `-n/--nframes` | 最多转多少帧 |
| `--stride` | 每隔多少帧取一帧 |
| `--precision` | XTC 精度，默认 1000（= 0.001 nm = 0.01 Å） |

## 2. 实测结果（本体系 122C12-450K，5596 原子）

| 项目 | 结果 |
|---|---|
| 写入速度 | **0.025 秒/帧** → 全 6001 帧约 **2.5 分钟** |
| XTC 体积 | 21.4 KB/帧 → 全轨迹约 **128 MB**（MDAnalysis 生成的是 124.8 MB） |
| 坐标精度 | 与原始 `.trj` 最大偏差 **0.005 Å** = precision=1000 量化步长（0.01 Å）的一半 |
| 盒子和时间 | 完全一致 |
| 能量表 | Epot 224301.58 kJ/mol = 53609.364 kcal/mol × 4.184 ✓，P −7823.96 bar = −0.782 GPa ✓（与 `122C12-450K.txt` 一致）|

## 3. 文件

| 文件 | 说明 |
|---|---|
| `xtcwriter.py` | **纯 Python XTC 写入器**（从 `xdrfile.c` 的 `xdrfile_compress_coord_float` 逐行移植）|
| `msdump.py` | 命令行工具（MS_dump 的对应物）|
| `test_xtcwriter.py` | 自检：写 XTC → 用 MDAnalysis 读回 → 与原始帧比对 |

依赖：`numpy`；读 `.dcd/.xtc` 时需要 `MDAnalysis`（可选）。

## 4. XTC 压缩算法要点

```
magic(1995) natoms step time(ps)
box[9]                        # nm，行主序 3x3
size(=natoms)                 # 以下仅当 natoms > 9
precision                     # 默认 1000
minint[3] maxint[3] smallidx  # ← smallidx 必须是「初始值」，循环里它会变
nbytes + 压缩数据(4 字节对齐)
```

1. 坐标先乘 `precision` 四舍五入成整数，平移 `minint` 到非负区间；
2. 三个整数用 `sizeofints` 算出的位宽「三合一」打包（`encodeints`）；
3. 相邻原子若三个方向差都小于 `smallnum`，合并成一个 run 做差分编码
   （`smallidx` 位宽）——XTC 压缩率主要来自这里；
4. 位流用 `encodebits` 写进缓冲区，**写入语义是「覆盖当前位置」而不是追加**。

移植时踩到的两个坑（都已修正，自检覆盖）：

* `buf1` 里存的是**原始**整数值，平移 `minint` 是写每个原子时临时做的；
  提前平移会导致 `encodeints` 收到越界值。
* 头部要写**初始** `smallidx`，而循环会修改它——写晚了读回来就是坏文件
  （症状：MDAnalysis 读的时候整数除零异常 `0xC0000094`）。

## 5. 与 C++ 版的差异

* **没做 `.trr`**：TRR 不压缩、实现简单，但需要逐帧写 6 个数组，这里没排进去；
  需要的话用 `MDAnalysis.coordinates.TRR.TRRWriter`。
* **没做 PDB 的 PBC 重排**：C++ 版会为对齐 MS 导出的 PDB 做盒子变换（三斜体系
  差别明显）。本工具的 `.trj` 里存的就是正交坐标，且已验证与 MS 的 `.pdb`
  首/末帧一致到 6e-14 Å，所以不需要变换。
* **`.xvg` 的交互式选组**没做，直接输出全部能量项。

## 6. 自检

```powershell
python test_xtcwriter.py --n 20
```
会用 `.dcd`（或指定的帧源）写一份 XTC，再用 MDAnalysis 读回来比对，
坐标偏差应 ≤ 0.005 Å。
