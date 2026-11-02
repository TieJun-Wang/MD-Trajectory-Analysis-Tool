# test —— 测试与诊断脚本

本目录存放**所有测试、回归、诊断、一次性探针脚本**。仓库只提交核心代码
（`mdta/`、`webapp/`、`mdta_cli.py`、`mdta_gui.py`、`examples/`、`pymsdump/`），
本目录整体被 `.gitignore` 排除，**不会**随 `git add .` 上传。

> 目录约定：脚本里若出现绝对路径 `C:\temp\MDT\...`，那是当时排查问题时的原样记录，
> 换个机器跑需要改路径；标了「当前入口」的四个才是日常要跑的。

---

## 一、当前入口（每次改动后必跑）

| 命令（在仓库根目录执行） | 检查什么 | 参考规模 |
|---|---|---|
| `python test/selftest.py` | **数值正确性主回归**：Rg / RDF / MSD / 二面角 / 界面 / 取向 / BOO / 晶体 / QC / 格式表 / `.trj` 转换 / MS 文档 / **LAMMPS 后缀**等 | 69 项，约 2–4 min |
| `python test/webapp/_api_check.py` | **后端 API 全流程**：健康检查 → 列目录 → 开会话 → 选对象 → 跑分析 → 导出 → 错误处理 → 关闭 | 全通过，约 30 s |
| `python test/webapp/_validate_supporting_info.py` | 与**作者原始输出**逐项对照（Rg/GROMACS `gyrate`、R_ee/PLUMED、密度/`gmx density`…） | 7 项，约 1 min |
| `python test/webapp/_validate_cross_system.py` | **跨体系回归**：45 个 CG 体系里能跑的 35 个，逐项比作者留下的参考值 | 约 3–4 min |

前端两项（必须在 `webapp/frontend` 目录内跑，因为要解析 `node_modules`）：

```powershell
cd webapp/frontend
node _build.mjs          # 构建 dist/（改了前端源码后必做）
node _ui_render.mjs      # React 组件渲染自检：356 项
cd ../..                 # 再回根目录
python test/webapp/_css_check.py   # 需要 8000 端口有服务在跑
```

> `_ui_render.mjs` / `_test_layout.mjs` / `_e2e*.mjs` / `_ssr_entry.js` **留在
> `webapp/frontend/`**：它们 `import` 的是 `vite` / `react`，Node 的模块解析必须从
> `webapp/frontend` 往上找 `node_modules`，挪出来会 `MODULE_NOT_FOUND`。
> 它们同样被 `.gitignore` 排除。

---

## 二、目录结构

```
test/
├── selftest.py                 ← 主回归（上面那张表第一行）
├── webapp/                     ← 后端/前端相关的检查脚本（Python）
│   ├── _api_check.py           ← 当前入口
│   ├── _css_check.py           ← 构建产物 CSS 几何校验（需 8000 端口）
│   ├── _validate_supporting_info.py
│   ├── _validate_cross_system.py
│   ├── _test_formats.py        ← 跨格式一致性（.tpr/.gro/.pdb）
│   ├── _test_export.py         ← 导出链路
│   ├── _test_live*.py          ← 真实 uvicorn + HTTP 的端到端
│   ├── _test_abeta.py          ← Aβ 肽板层体系
│   ├── _font_check_all.py      ← 字体回退检查
│   └── _probe46.py             ← 早期 46 体系体检
├── comparisons/                ← 与另一套工具（mdtransport）的逐项对照实验
│   ├── 1_msd_water_vs_theirs.py
│   ├── 2_units_and_refit.py
│   ├── 3_pbc_control.py
│   └── 4_factor_is_species_dependent.py
├── fixtures_fmt/               ← 从 .tpr 导出的同名 .gro / .pdb（跨格式测试用）
├── _diag_*.py                  ← 一次性诊断脚本（报告是同名 _diag_*.txt）
├── _probe_*.py                 ← 一次性探针（查某个数为什么不对）
└── _*.txt                      ← 上述脚本留下的报告
```

### `_diag_*.py` 与 `_probe_*.py` 是什么

排查具体问题时写的**一次性脚本**，保留下来是为了让"当时为什么这么改"有据可查：
每个都对应 `modify.md` 里的一条修复记录（例如 `_diag_rg_243.py` ↔ Rg 差一倍那条、
`_diag_lammps_*.py` ↔ LAMMPS 后缀补充那条）。它们**不参与**回归，写法也不统一
（多数把 `C:\temp\MDT\...` 写死了）。要看结论直接读同名 `.txt`。

---

## 三、夹具（运行中自动生成的目录）

| 目录 | 谁生成 | 内容 |
|---|---|---|
| `test/_fmt_probe/` | `selftest.py` | 格式错槽位/`.xtd`+`.trj` 配对用的临时文件 |
| `test/_lammps_fixture/` | `selftest.py` | LAMMPS data / dump / `in.lammps` / `log.lammps` 夹具 |
| `test/_selftest_out/` | `selftest.py` | 导出链路测试的产物 |
| `test/webapp/…` 旁的临时目录 | 各自脚本 | 例如 `_export_probe/`、`_fontcheck_out/` |

这些都随 `test/` 一起被忽略，删掉不影响仓库。

---

## 四、跑测试前的准备

1. **依赖**：`pip install -r webapp/requirements.txt`（含 MDAnalysis / FastAPI / numpy…）
2. **样例数据**：`selftest.py` 默认用仓库根的 `adk_oplsaa.tpr` + `adk_oplsaa.xtc`
   （这两个是**轨迹数据，不入库**；缺失时相关用例会跳过或报"文件缺失"）。
   也可以显式指定：`python test/selftest.py sys.tpr sys.xtc`，
   或用环境变量 `MDTA_TEST_TOP` / `MDTA_TEST_XTC`。
3. **`_api_check.py` 不需要起服务**（用 FastAPI `TestClient`）；
   `_css_check.py` / `_test_live_http.py` 需要 8000 端口上有服务：
   `python -m webapp.backend.run`（另开一个终端，检查完关掉）。
4. **前端**：`cd webapp/frontend && node _build.mjs` 产出 `dist/`，
   后端从 `dist/` 托管页面（不构建也能跑 API，只是页面拿不到前端产物）。

---

## 五、`.trj`（Materials Studio）相关

MS 的 `.trj` 转换依赖两个外部目录（`md_convert/ms_trj.py` + `pymsdump/xtcwriter.py`），
缺了会给出可操作的报错。位置可用环境变量指定：

```powershell
set MDTA_MS_CONVERT=D:\md_convert;D:\pymsdump
```

`webapp/frontend/_build.mjs` 是**构建脚本**（保留在版本库的排除之外，见下），
与测试无关；只是它和前端测试脚本放在同一个目录，容易混淆。
