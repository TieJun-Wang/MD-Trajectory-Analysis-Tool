# test —— 测试与自检

本目录只保留**两个**入口脚本（其余历史诊断/探针脚本已按用户要求清理，
旧脚本归档在仓库外层的 `_test_archive/`，不入库、需要时可翻）。

> 目录整体被 `.gitignore` 排除，**不会**随 `git add .` 上传；这里只有
> `README.md` 本身入库，方便别人知道"测试怎么跑"。

---

## 一、两个入口

| 命令（在**仓库根目录**执行） | 检查什么 | 规模 / 用时 |
|---|---|---|
| `python test/selftest.py` | **数值正确性主回归**：Rg / RDF / MSD / 二面角 / 界面 / 取向 / BOO / 晶体 / QC / 格式后缀表 / `.trj` 转换 / MS 文档 / LAMMPS 后缀 … | **70 项**，约 2–4 min |
| `python test/webapp/_api_check.py` | **后端 API 全流程**：健康检查 → 列目录 → 开会话 → 选对象 → 跑分析 → 导出 → **刷新恢复/增量追加/清空** → 错误处理 → 关闭 | 全通过，约 30 s（用 `TestClient`，**不用起服务**） |

两者都不需要浏览器；前端渲染类自检在 `webapp/frontend/_ui_render.mjs`
（需在 `webapp/frontend` 目录执行 `node _ui_render.mjs`，406 项）。

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
