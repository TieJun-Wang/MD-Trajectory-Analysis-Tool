# test —— 测试与自检

本目录只保留**两个**入口脚本（其余历史诊断/探针脚本已按用户要求清理，
旧脚本归档在仓库外层的 `_test_archive/`，不入库、需要时可翻）。

> 目录整体被 `.gitignore` 排除，**不会**随 `git add .` 上传；这里只有
> `README.md` 本身入库，方便别人知道"测试怎么跑"。

---

## 一、两个入口

| 命令（在**仓库根目录**执行） | 检查什么 | 规模 / 用时 |
|---|---|---|
| `python test/selftest.py` | **数值正确性主回归**：Rg / RDF / MSD / 二面角 / 界面 / 取向 / 接触 / BOO / 晶体 / QC / 格式后缀表 / `.trj` 转换 / MS 文档 / LAMMPS 后缀 / 取消响应 … | **97 项**，约 2–4 min |
| `python test/webapp/_api_check.py` | **后端 API 全流程**：健康检查 → 列目录 → 开会话 → 选对象 → 跑分析 → 导出 → **刷新恢复/增量追加/清空** → 错误处理 → 关闭 | 全通过，约 30 s（用 `TestClient`，**不用起服务**） |

两者都不需要浏览器；前端渲染类自检在 `webapp/frontend/_ui_render.mjs`
（需在 `webapp/frontend` 目录执行 `node _ui_render.mjs`，754 项）。

---

## 一·五、`test/mda_test/` —— 用 MDAnalysis 独立复现（对拍用）

按分析模块拆成一个个小脚本，每个都能单独跑，也支持 `python test/mda_test/run_all.py`
一次跑完：

| 脚本 | 复现什么 |
|---|---|
| `_common.py` | 加载体系（`.xsd/.xtd/.pdb/.gro/.tpr` + `.trj/.trr/.xtc`）、补质量、帧窗口、分子编号、最小镜像距离、导出 CSV |
| `_menu.py` | **交互菜单 + 参数规格**（一步步问：拓扑 → 轨迹 → 对象 → 帧 → 模块参数 → 输出） |
| `rg.py` / `ree.py` / `msd.py` / `rdf.py` / `comdis.py` | Rg、末端距、MSD、RDF、两组分质心距 |
| `rdf_mdat.py` | **RDF 的 util_rdf.py 口径**（MDTransport：逐帧全原子对 + 当场最小镜像 + 同种粒子数−1 归一化），含配位数曲线；与 `InterRDF`/引擎/原实现三方对拍 |
| `msd_mdat.py` | **MSD/扩散系数的 util_msd.py + util_diffusion.py 口径**（增量最小镜像解包 + 去质心漂移 + FFT 全起点平均；D = β≈1 窗口的 OLS 斜率/(2·dim)）；与 `EinsteinMSD`/引擎对拍 |
| `density.py` | **1D 密度分布 + 归一化曲线**：与程序「导出板块」同口径导出 `density.csv` / `density__norm.csv`；教科书实现 + 与 `LinearDensity` 对拍 + 质量守恒自检 |
| `contact.py` | **接触分析 + 接触矩阵**（与引擎逐位一致）+ **独立复现 & 与引擎逐帧对拍**（原 `examples/contact_reproduce.py` 的包内版本：`reproduce_contacts` / `compare_engine` / `test_contact_reproduce`） |
| `orientation.py` | **链段取向**（`allfast-3-2.py` 口径：键中点矢量 → ⟨cos²θ⟩ / 取向参数 / ⟨θ⟩(t)）；同时给**逐链等权**与**链段加权**（= 程序引擎）两套数，自动跳过石墨/环/支化碎片 |
| `interface.py` | **界面宽度 10–90 + erf**：参考实现（教科书剖面 + 局部 10–90 + erf 拟合）、**合成已知宽度界面**（真实 10–90 = 2.5631σ）、与引擎逐位对拍、`LinearDensity` 剖面交叉验证 |

> 与第三方程序（**MDTransport**：`util_rdf.py` / `util_msd.py` + `util_diffusion.py`）
> 能逐值互证的就是 **RDF** 与 **MSD/扩散系数** 两项，都已并入（`rdf_mdat.py` /
> `msd_mdat.py`）。它的其余子命令是离子输运/寿命类（Nernst–Einstein、Einstein–Helfand、
> Onsager、Stefan–Maxwell、vehicular、cage/cage_p/pcl、corr/spda），我们本来就不涉，
> 无法互证；唯一还能顺带做的是 **COM–COM RDF**（把分子质心当粒子）。

### 两种用法（每个模块都一样）

```powershell
# ① 交互菜单：不带参数，一步步问
python test/mda_test/contact.py
#    ① 拓扑文件 ② 轨迹文件 ③ 分析对象（列组分让你选）④ 帧选择 ⑤ 模块参数 ⑥ 输出 → 确认

# ② 命令行：带参数，适合批处理
python test/mda_test/contact.py --top a.xtd --trj a.trj --frames 100 ^
       --cutoff 5 --mode inter --select-b "" --prefix contact_a -y
python test/mda_test/contact.py --help        # 看本模块的全部开关
```

菜单里的「分析对象」会**列出体系里有哪些组分**（各多少原子/分子），可以按序号选、
也可以直接敲 MDAnalysis 选择串（如 `"element O"`）；「帧选择」支持全部 / 均匀取 N 帧 /
指定 `start/stop/step`。挑出来的帧保证是**等差**的，这样 `InterRDF` / `EinsteinMSD`
这类只认 `start/stop/step` 的分析类跑的帧和菜单里选的完全一致。

**默认体系 = `C:\temp\dataset\20PE30`**（20 条 PE30 链 / 3640 原子 / 301 帧 / 立方盒
33.6 Å，拓扑 `.xsd` + Materials Studio `.trj`）。换体系不用改代码：环境变量
`MDTA_MDA_XSD` / `MDTA_MDA_TRR` / `MDTA_MDA_SELECT` / `MDTA_MDA_FRAMES` /
`MDTA_MDA_OUT`，或命令行 `--top / --trj / --select`；不给 `--top/--trj` 就用默认体系，
要只读单帧结构加 `--no-traj`。口径说明见仓库根 `接触分析复现指南.md`。

### 界面宽度（`interface.py`）怎么对照

界面宽度**没有**第三方的"一键函数"，所以 `interface.py` 做了三层对照，每层验的东西不同：

1. **解析轮廓**（`check_definition()`）：直接给一条 `f(x)=½[1−erf((x−x₀)/(√2σ))]` 的
   理想曲线，10–90 宽度必须等于 `2.5631σ`、erf 拟合必须还原 σ。这验证的是**宽度判据**
   （不掺采样噪声）。
2. **合成已知宽度体系**（`synth_slab_universe` / `test_synthetic`）：原子按
   `½[1−erf((d−hw)/(√2σ))]` 的概率判给 A/B，构成两层板（两个界面），真实 10–90 宽度
   解析等于 `2.5631σ`。单份构型 0.5 Å 分箱的统计噪声约 ±5%，多份取平均后可到 1–2%。
3. **与引擎逐位对拍**（`compare_engine`）＋ **剖面换 `LinearDensity`**
   （`compare_mdanalysis_interface`）：同一批帧、同一 bin 下，我们参考实现的界面位置 /
   宽度与程序引擎的差异应在 1e-9 Å 量级。

两个必须知道的细节：

- **平滑会加宽窄界面**：`smooth=5`、0.5 Å 分箱时，对 σ=2 Å（真实 10–90 ≈ 5.13 Å）的界面
  实测会到 5.6 Å 左右（**+9%**）。核对绝对宽度时用 `--smooth 1`（模块默认就是 1），
  真实数据为压噪声才用 5。
- **硬壁不报宽度**：石墨/晶体壁这类"一个 bin 就走完 0→1"的界面，10–90 在采样范围内
  量不出来，程序会如实说"宽度不可直接测量"（位置照给）。这不是 bug。

### 链段取向（`orientation.py`）怎么对照

`orientation.py` 里每个量都输出**两套权重**，弄清楚该看哪一列很重要：

| 列 | 含义 | 与谁对应 |
|---|---|---|
| `cos2_axis` / `P2` | **逐链等权**：每条链先算 ⟨cos²θ⟩，再在链之间平均 | 程序「统计权重 = 每条链等权」 |
| `cos2_pooled` / `P2_pooled` | **链段加权**：所有链的所有链段进一个池子再平均，长链权重大 | 程序「统计权重 = 链段加权」（**默认**，逐值一致，实测差 ~1e-17） |
| `theta_deg` / `theta_pooled` | 同上两种权重的 ⟨θ⟩ | 引擎的角 θ(t) 面板 |

**单分散体系（122×C12、20×PE30 这类）两套权重数值相同**，所以只有多分散体系
才会被发现"对不上"。实测 310K（C6–C50 混油 + 石墨片，96 个分子）：

- 程序默认（链段加权）P₂ 第 0 帧 −0.132838；改成「每条链等权」→ −0.085727；
- 参考脚本的 `P2_pooled` 与前者、`P2` 与后者分别逐位一致；
- 旧的参考脚本按**众数链长**只留下 25 条 C6（其余 71 条链全丢）→ P₂ 到 +0.16。

所以参考脚本默认行为已改成：**不按链长过滤**、只跳过**非线性**碎片（石墨/环/支化：
重原子子图有 3 度节点或成环），并在开头打印链长分布与"多分散"提醒。
要按长度筛就用 `--n-atoms <整数>`（或 `chain_coordinates(..., n_atoms="mode")`
复现旧的众数口径）。

坐标系也用了解包坐标（`make_whole`）：跨周期边界的分子不会被算成
"横穿盒子的链段"。

---

## 二、跑之前准备什么

1. **依赖**：`python -m pip install -r requirements.txt`（仓库根那份，含 MDAnalysis / FastAPI 等）。
2. **样例数据**（可选）：`selftest.py` 默认用仓库根的 `adk_oplsaa.tpr` + `adk_oplsaa.xtc`；
   这两个是**轨迹数据、不入库**。缺了不影响主流程——相关用例会用合成体系或跳过。
   也可以显式指定：`python test/selftest.py sys.tpr sys.xtc`，
   或用环境变量 `MDTA_TEST_TOP` / `MDTA_TEST_XTC`。
3. **夹具**：`selftest.py` 会在 `test/` 下自建 `_fmt_probe/`、`_lammps_fixture/`
   等临时目录（都是自动生成，删掉无妨）。

---

## 三、`.trj`（Materials Studio）相关

`.trj` 转换所需的两个纯 Python 模块**已随仓库提供**（`pymsdump/ms_trj.py` 解析器 +
`pymsdump/xtcwriter.py` 写入器），开箱即用。自检一行：

```powershell
python -m mdta.ms_trj
```

想用别的副本时再指定：`set MDTA_MS_CONVERT=D:\path\md_convert;D:\path\pymsdump`。
