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

按分析模块拆成一个个小脚本，每个都能单独跑（`python test/mda_test/contact.py`），
也支持 `python test/mda_test/run_all.py` 一次跑完：

| 脚本 | 复现什么 |
|---|---|
| `_common.py` | 加载体系（`.xsd`/`.xtr`…）、补质量、帧窗口、分子编号、最小镜像距离、导出 CSV |
| `rg.py` / `ree.py` / `msd.py` / `rdf.py` / `comdis.py` | Rg、末端距、MSD、RDF、两组分质心距 |
| `rdf_mdat.py` | **RDF 的 util_rdf.py 口径**（MDTransport：逐帧全原子对 + 当场最小镜像 + 同种粒子数−1 归一化），含配位数曲线；与 `InterRDF`/引擎/原实现三方对拍 |
| `contact.py` | **接触分析 + 接触矩阵**（与引擎逐位一致） |
| `orientation.py` | **链段取向**（`allfast-3-2.py` 口径：键中点矢量 → ⟨cos²θ⟩ / 取向参数 / ⟨θ⟩(t)） |

数据与输出全部用环境变量控制：`MDTA_MDA_XSD` / `MDTA_MDA_TRR` / `MDTA_MDA_SELECT`
/ `MDTA_MDA_FRAMES` / `MDTA_MDA_OUT`。定位说明见仓库根 `接触分析复现指南.md`。

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
