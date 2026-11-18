# test/my_test —— 本工具自己的功能脚本

这里放的是**用本工具的引擎把每一项功能跑一遍**的脚本：一个分析一个文件，
每个都能单独运行，也能一次跑完全部。它回答的是"**这个功能怎么用、参数怎么给、
输出长什么样、导出的文件叫什么名**"。

> 和 `test/mda_test/` 的分工：
> - `test/mda_test/` = **第三方对拍**（MDAnalysis 的 `InterRDF` / freud 的 `Steinhardt` /
>   作者自己的 `allfast-3-2.py` 口径…）→ 回答"**算得对不对**"；
> - `test/my_test/`（本目录）= **功能脚本**（直接调 `mdta.analysis.*`）→ 回答"**怎么用**"。
>
> 两者互补：先在这里看清输出，再去 mda_test 核对数值。

---

## 一、入口

| 命令（在**仓库根目录**执行） | 做什么 | 用时（20PE30，20 帧） |
|---|---|---|
| `python test/my_test/run_all.py` | 加载一次体系，跑**全部**模块并打印 ✅/⏭/✗ 汇总表 | 约 20 s |
| `cd test; python -m my_test` | 同上（等价入口） | 约 20 s |
| `python test/my_test/<模块>.py` | **单个模块**：不带参数 = 交互菜单，带参数 = 命令行 | 每个 0.1–6 s |
| `python test/my_test/run_all.py --pipeline` | 跑完 16 项后再跑一遍「一键分析」（`Analyzer.run_all` + 导出全部） | +10–30 s |
| `python test/my_test/<模块>.py --help` | 看这个模块的全部命令行开关 | — |

```powershell
# 典型用法
python test/my_test/run_all.py --frames 20                  # 默认体系 20PE30
python test/my_test/run_all.py --system mix --frames 10      # 多组分体系（界面/质心距能跑出来）
python test/my_test/contact.py --frames 50 --cutoff 4.5 -y   # 只跑接触分析
python test/my_test/orientation.py --mode backbone --weighting molecule --frames 50 -y
```

---

## 二、模块清单

| 脚本 | 对应的工具功能（引擎函数） | 关键参数 | 导出文件（= 界面「下载 CSV」同名） |
|---|---|---|---|
| `system.py` | 体系信息 / 组分识别 / 能力检测 / 可行性预判（`systeminfo` / `selection` / `capabilities` / `feasible`） | — | `system__components.csv`、`system__feasible.csv` |
| `frames.py` | 帧选择（`preprocess.select_frames`） | `start_ps` / `stop_ps` / `interval_ps` / `equil_ps` / `max_frames` | `frames__selection.csv` |
| `rg.py` | 回转半径 Rg（`analyze_rg`） | `object`（分子/整体）、`mass_weighted` | `rg__time.csv`、`rg__dist.csv`、`rg__permol.csv` |
| `ree.py` | 端到端距离 R_ee（`analyze_end_to_end`） | `ends`（bond_graph / residue / custom）、`per_molecule` | `ree__time.csv`、`ree__dist.csv` |
| `dihedral.py` | 二面角分布 + trans/gauche 比例（`analyze_dihedrals`） | `mode`、`gauche_edge`、`bin_width` | `dihedral__dist.csv`、`dihedral__trans.csv` |
| `density.py` | 密度分布 ρ(z)（`analyze_density`） | `axis`、`mode`（质量/数密度）、`nbins` | `density__profile.csv`、`density__norm.csv` |
| `rdf.py` | 径向分布函数 g(r)（`analyze_rdf`） | `rmax`、`nbins`、`mode`（total/inter/intra/all） | `rdf__rdf.csv`（+`rdf__halves.csv`） |
| `rdf2d.py` | 面内径向分布 g₂(r)（`analyze_rdf_inplane`） | `plane_axis`、`rmax`、`nbins`、`slab_lo/hi` | `rdf2d__inplane.csv` |
| `comdist.py` | 两组分质心距（`analyze_com_distance`） | `axis`、`pbc`、`pair_a/pair_b` | `comdist__dist.csv`、`comdist__axis.csv` |
| `contact.py` | 接触分析（`analyze_contacts`） | `cutoff`、`mode`、`pair_a/pair_b`、`top_n` | `contact__count.csv`、`contact__mindist.csv`、`contact__matrix.csv`、`contact__pairs.csv` |
| `interface.py` | 界面宽度 10–90（`analyze_interface_width`） | `axis`、`nbins`、`method`、`smooth`、`pair_a/pair_b` | `interface__profile.csv`、`interface__norm.csv` |
| `orientation.py` | 链段取向 S / P₂ / ⟨cos²θ⟩（`analyze_orientation`） | `mode`、`stride`、`p2_axis`、`weighting` | `orientation__s.csv`、`orientation__p2.csv`、`orientation__angle.csv`、`orientation__dist.csv` |
| `order.py` | 结构有序度（`analyze_structural_order`） | `gauche_edge`、`g_ref` | `order__components.csv`、`order__index.csv` |
| `boo.py` | 键取向序 BOO（`analyze_boo`） | `cutoff`、`averaged` | `boo__qbar6.csv`、`boo__qbar6_dist.csv`、`boo__q4q6.csv` |
| `msd.py` | 均方位移 MSD + D（`analyze_msd`） | `object`（原子/质心）、`remove_drift`、`max_particles` | `msd__tau.csv`、`msd__fit.csv`、`msd__axes.csv`、`msd__power.csv` |
| `vacf.py` | 速度自相关函数（`analyze_vacf`） | `max_lag`、`normalize` | `vacf__vacf.csv`、`vacf__fit.csv` |
| `pipeline.py` | **一键分析**：`Analyzer.run_all` + `export_all` | `which`、`estimate`、`excel`、`panel_pngs` | 每项的 CSV/PNG + `summary.xlsx` |

两个公共文件：

| 文件 | 作用 |
|---|---|
| `_common.py` | `Engine` = 加载体系（`mdta.io.load_trajectory`）→ 识别组分（`Analyzer.auto_setup`）→ 选帧 → 跑分析 → 打印 + 导出 |
| `_menu.py` | 交互菜单 + 命令行（**复用** `test/mda_test/_menu.py`，只把加载器换成工具的） |

---

## 三、两种用法

```powershell
# ① 交互菜单（不带参数）：拓扑 → 轨迹 → 分析对象 → 帧 → 本模块参数 → 输出 → 确认
python test/my_test/contact.py

# ② 命令行（带参数）：适合批处理 / 复现
python test/my_test/contact.py --frames 50 --cutoff 4.5 --mode inter -y
python test/my_test/contact.py --help          # 本模块全部开关
```

菜单里「分析对象」会**列出体系里有哪些组分**（各多少原子/分子），
可以按序号选 `resname XXX`，也可以直接敲选择串（如 `"element O"`）；
「帧选择」支持全部 / 均匀取 N 帧 / 指定区间。
常用开关（每个模块都有）：`--top` `--trj` `--select` `--frames`
`--outdir` `--prefix` `--png` `-y`。

```python
# ③ 当库用（自己的脚本里调）
import sys; sys.path.insert(0, r"C:\temp\MDT\test")
from my_test import Engine, test_rg, test_contact, test_orientation

eng = Engine(max_frames=20)                  # 默认 20PE30；也可以 Engine(top, trj)
test_rg(eng)                                 # 逐分子 Rg
test_contact(eng, cutoff=4.5)                # 接触分析
test_orientation(eng, mode="backbone", weighting="molecule")
```

---

## 四、默认体系与换体系

默认 **20PE30**（`C:\temp\dataset\20PE30`）：20 条 PE30 链 / 3640 原子 / 301 帧 /
立方盒 33.6 Å / **单组分**（`polymer` ← 正构烷烃 C60H122）。
换多组分体系可看界面宽度 / 质心距 / 面内 RDF：

```powershell
python test/my_test/run_all.py --system mix --frames 10
#   → 5 条 PEO20 + 2000 水 + 48 离子 / 6758 原子 / 1001 帧
#     自动识别 polymer / water / ion 三个组分
```

| 方式 | 写法 |
|---|---|
| 命令行 | `--top 拓扑 --trj 轨迹`（不给即用预置体系） |
| 预置体系 | `--system pe`（20PE30）/ `--system mix`（5PE20…） |
| 环境变量 | `MDTA_MYTEST_SYSTEM` / `MDTA_MYTEST_TOP` / `MDTA_MYTEST_TRJ` / `MDTA_MYTEST_SELECT` / `MDTA_MYTEST_FRAMES` / `MDTA_MYTEST_OUT` |
| 只读单帧结构 | 菜单里把轨迹留空（命令行用 `--no-traj`） |

输出目录默认 `test/my_test/_out/`（**不入库**）；用 `--outdir` 或 `MDTA_MYTEST_OUT` 改。

---

## 五、几处容易误解的地方

**1. 单组分体系里「界面宽度」「两组分质心距」会跳过 —— 这是对的。**
它们都是"两组分之间"的量：一个组分时没有界面可谈。脚本会打印
`⏭ 跳过：需要至少勾选 2 个组分`，`system.py` 的可行性预判里也会提前标出来
（界面「分析功能」里置灰的就是这一类）。要看它们跑出结果请用 `--system mix`。

**2. 接触分析在单组分体系是 A–A 同组自接触。**
引擎按"A = 最大组分，B = 与 A 不同分子的最大组分"选组，凑不出第二组时退回
同组无序对（不含 `i==i`）。20PE30 实测 5 Å / 20 帧：平均 43,503 对/帧。

**3. VACF 需要**带速度**的轨迹。**
本目录对 `.trj` **默认保留速度**（转 TRR，带坐标+速度），所以
`python test/my_test/vacf.py` 开箱可用；`.xtc` / `.dcd` 不存速度，VACF 会
`⏭ 跳过`并说明原因。想省磁盘/时间（只要坐标）就设 `MDTA_MYTEST_VELOCITIES=0`。

**4. 「取向 S」与「结构有序度 S」是同一个数。**
有序度里的取向分量通过 `Analyzer._shared_params` 继承「链段取向」的设置
（链段定义 / stride / 统计权重），所以两处报的 S 必须一致；不一致就是 bug。

**5. `dihedral` 的 `trans/gauche` 阈值是**闭区间**：`|φ| ≤ gauche_edge` 记 `gauche±`。**
与 Forcite 的 torsion 统计同约定（边值归 gauche）。

**6. 平滑会加宽窄界面。**
`interface --smooth 5` 对 σ≈2 Å 的界面会偏宽约 9%；核对绝对宽度用 `--smooth 1`。

---

## 六、`run_all.py` 的判断规则

| 标记 | 含义 |
|---|---|
| ✅ | 这一项跑出了结果（已在 `_out/` 落盘 CSV） |
| ⏭ | 体系不满足前提，**引擎主动跳过**（例如单组分做界面宽度、无速度做 VACF） |
| ✗ | 真的报错了（脚本会打印 traceback 前 3 层，退出码 1） |

单组分体系预期是 **14 ✅ / 2 ⏭ / 0 ✗**；多组分（`--system mix`）预期全部 ✅
（VACF 仍取决于轨迹有没有速度）。

---

## 七、跑之前准备什么

1. **依赖**：`python -m pip install -r requirements.txt`（仓库根那份）。
2. **样例数据**（可选）：上面两个预置体系在 `C:\temp\dataset\` 下；
   缺了就用 `--top/--trj` 指到自己的体系，或设环境变量。
3. **产物**：CSV/PNG 落在 `test/my_test/_out/`（已 gitignore），随时可删。

口径的完整说明（每一项怎么定义、为什么这么定义、跟 Forcite 怎么对齐）
见仓库根的 `使用手册.md`；数值对拍见 `test/mda_test/` 与 `test/README.md`。
