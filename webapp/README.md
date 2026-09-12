# MD Trajectory Analysis Tool —— Web 版

FastAPI 后端 + React 前端。**分析引擎完全复用现有的 `mdta` Python 包**，
后端只负责算和给数据，图表由前端用 ECharts 渲染。

```text
浏览器 (React + ECharts)
   │  fetch /api/*
   ▼
FastAPI 后端  ──►  mdta 包（MDAnalysis 分析引擎，与 CLI / 桌面版完全相同）
   │
   └── 静态托管 前端 dist/  → 单一 URL
```

---

## 快速开始

```powershell
# 1) 装后端依赖（FastAPI）
python -m pip install -r webapp\requirements.txt

# 2) 装前端依赖并构建（只需一次）
cd webapp\frontend
npm install
npm run build
cd ..\..

# 3) 启动服务
python -m webapp.backend.run
```

然后打开 **<http://127.0.0.1:8000>**。
接口文档在 <http://127.0.0.1:8000/docs>（FastAPI 自动生成，可直接点着试）。

### 开发模式（改前端代码即时生效）

```powershell
# 终端 1：后端
python -m webapp.backend.run --reload

# 终端 2：Vite 开发服务器（已配好 /api 代理到 8000）
cd webapp\frontend
npm run dev            # http://127.0.0.1:5173
```

---

## 界面说明

整页**一屏显示，不出现页面级滚动**；左侧操作台是三块**互斥折叠**面板，
一次只展开一块。

```text
┌───────────────────────────────────────────────────────────────────────┐
│ MD  MD Trajectory Analysis Tool          ● 后端在线  sid xxxx  10 项结果│
├──────────────────┬────────────────────────────────────────────────────┤
│ ▼ 1. 文件读取     │ [图表] [数据表] [统计量/导出] [运行日志]            │
│   ┌────────────┐ │ ┌──────────────┬───────────────────────────────┐   │
│   │目录浏览/选择│ │ │ 图表导航      │  回转半径 Rg —— 最大链          │   │
│   │[读取体系]  │ │ │  ▾ 回转半径Rg │  第 1 / 2 张  [上一张][下一张]  │   │
│   └────────────┘ │ │    1.Rg-时间  │  ┌─────────────────────────┐  │   │
│ ▶ 2. 体系信息     │ │    2.Rg 分布  │  │      ECharts 图          │  │   │
│ ▶ 3. 分析设置     │ │  ▸ 二面角分析 │  │  缩放/平移/hover/图例    │  │   │
│                  │ │  ▸ 接触分析   │  └─────────────────────────┘  │   │
│                  │ │               │  说明 + 关键统计量（固定小块）│   │
│                  │ └──────────────┴───────────────────────────────┘   │
└──────────────────┴────────────────────────────────────────────────────┘
```

### 顶栏

整条顶栏是**深靛紫底纹**（`linear-gradient(120deg, #312e81 → #3f3d9e → #4c1d95)`，
再加一道内高光/暗边的 `box-shadow`，看起来是有纹理的一条而不是纯色块）。

- **左对齐**：第一行加粗黑体 `MD Trajectory Analysis Tool`（白色）；第二行
  `Molecular Dynamics Simulation Analysis Platform`（淡靛紫 `#c7d2fe`，
  比第一行小 2px）；
- **右对齐**：`● status: Online`（`#86efac` 亮绿 / 离线 `#fca5a5` 亮红）
  与 `● version: 1.0.0`；跑分析时在它们前面插入实时进度条。
  会话 id 与 Python 版本收进这两个元素的 `title` 提示里，不占版面。

> 深色底上一个容易踩的坑：只改底色、不改文字色，就会出现「深底深字」看不清。
> 所以顶栏相关的色全部重新调过，并且 `_css_check.py` 里加了一道**对比度断言**：
> 直接算相对亮度，要求底色够深、标题/副标题/状态色与底色对比度 > 4:1。
> 当前实测：底色 `#312e81`（亮度 0.042），标题 **11.4:1**、副标题 **7.7:1**
> —— 都超过 WCAG AAA（7:1）。

### 左侧：三块互斥折叠面板

| 面板 | 内容 | 说明 |
|---|---|---|
| **1. 文件读取** | 目录浏览 + 选拓扑/轨迹 + 「读取体系」 | 选中 `.tpr` 后自动匹配同名 `.xtc` |
| **2. 体系信息** | 原子数/帧数/模拟盒/组分等 + 「展开完整报告」 | 读取成功后**自动切到这一块**；底部有「下一步：分析设置 →」 |
| **3. 分析设置** | **分析对象 + 参数设置 + 分析功能 + 运行**（合成一块） | 见下 |

文件选择器同时列出**拓扑**与**轨迹**两类文件，支持 `.tpr` `.gro` `.pdb` `.psf`
`.prmtop`（拓扑）与 `.xtc` `.trr` `.dcd` `.nc` `.gro` `.pdb` 等（轨迹）。
`.gro` / 无 `CONECT` 的 `.pdb` 没有化学键，工具会**自动猜键**（见根目录 README 的
格式支持表）；`.top`（GROMACS 力场拓扑）会在读取时被**明确拒绝**并给出提示，
而不是抛一个看不懂的底层错误。

**互斥**：点某块的标题栏展开时，其它两块自动收起，**一次只展开一块**；
标题栏右侧还会显示该块的摘要（文件名 / 原子数 / 已选分析项数）。
未读取体系时，「体系信息」「分析设置」两块是禁用的。

#### 左栏可整体折叠（竖条永远在最左侧）

整条操作栏可以收起，把宽度全部让给图表：

- **最左侧**常驻一条 **26px 的竖条**（竖排文字）：展开时显示
  「**收起操作台**」，收起时显示「**展开操作台**」——
  位置始终不变，鼠标不用跑到屏幕另一头去找按钮；
- 收起后整栏只剩这条竖条，其余宽度全部让给图表；
- 折叠/展开是 **240ms 的宽度过渡**，图表导航与图表区随之**平滑扩展开**：
  - 面板**内容保持固定宽度**（只是被裁掉），所以动画期间文字不会反复重排、不会抖；
  - `ChartView` 用 `ResizeObserver` 捕获容器变化，并把回调**合并到每帧一次**
    （`requestAnimationFrame`）—— 动画期间 ECharts 每帧只 resize 一次，
    大曲线也不会卡；
  - 折叠时内容同时置 `visibility: hidden`，否则 Tab 键还会聚焦到看不见的输入框。

#### 左右两栏互斥 + 跑完自动让位给图表

屏幕宽度有限，左右两栏**不会同时展开**：

- 展开操作台 → 自动收起右侧说明栏；展开说明栏 → 自动收起操作台；
- **一点「开始分析」就立刻**「收起操作台 + 展开说明栏」，把屏幕让给图表，
  不用等分析跑完。进度不会因此看不见：
  - 顶栏常驻一个紧凑的实时指示器（`● 实时更新中` + 进度条 + `4/10 · 12s`）；
  - 图表区的占位提示也会显示「正在计算第一项：…（0/10）」；
  - 想回看详细进度（当前项、排队中列表、取消按钮）就点最左侧竖条把操作台展开。

> 说明栏的折叠载体是 `.notes-clip`（`flex-basis` 从 `--notes-width` 动画到 `0`），
> 固定宽度的 `.notes-body` 在它**里面**。这一点很关键：
> 早先把固定宽度的内容直接放在竖条前面，折叠时内容占满整栏、把竖条挤到
> 可视区之外，于是「折叠之后就再也点不开了」。现在竖条一定落在可见范围内。

第 3 块的内部排版（都为紧凑设计，保证整块能塞进一屏）：

- **分析对象**：主链下拉框（自动=最大链 / 按组分 / 按 segid / 自定义选择语句）；
  参与分析的组分做成**可点切换的胶囊**，比一排复选框省地方；
- **参数设置**：**三列网格**（起始、结束、平衡段 / 抽帧间隔、最多帧数、cutoff /
  密度方向、bin 数、RDF rmax / 二面角模式）；
- **分析功能**：四行紧凑勾选（链构象 / 界面 / 结晶 / 辅助），带已选计数，
  以及一排快捷按钮：**全选 / 仅链构象 / 仅界面 / 仅结晶 / 仅辅助**。
  - 「仅 XX」按钮由分组表自动生成（不会和勾选项不同步），
    当当前勾选**正好等于**某一类时会**高亮**，一眼看出自己在哪个预设上；
  - 只有 5 个按钮，`flex-wrap: nowrap` **强制单行**
    （构建产物检查里用真实 CSS 数值估算：占用 251px / 可用 348px，余量 97px）；
  - **没有「全不选」按钮** —— 要全部取消就逐个点掉勾选框，
    "什么都不选"本身不是一个需要按钮的状态；
  - 一个分析项都没勾就点「开始分析」会直接提示「请至少选择一个分析项」，
    而不是空跑一轮。
- **运行**：「开始分析」+ 进度条 + 最近一行日志摘要，以及
  「查看完整运行日志 →」（跳到右侧日志页）。
  运行时这一块会变成**实时面板**：
  - 标题旁出现 `● 实时更新中` 徽标（带脉冲圆点）；
  - 按钮文字显示 `分析中… 4/10`；
  - 进度条按后端真实进度走，下面一行是后端的当前消息（`已完成 4/10：…`）
    与已用时间；
  - 「排队中」列出还没算的分析（中文标题）；
  - 「取消（保留已算完的）」会真的让后端停下，已算完的图不会丢。

### 实时更新：算好一项就出一项

以前是「等全部 10 项跑完才一次性返回」，Abeta 体系要干等 18 秒才看到第一张图。
现在后台**每算完一项就立刻写进结果集**，前端每 500 ms 增量拉一次：

- 图表导航里的分析**一个一个长出来**（`还有 N 项在算` 徽标告诉你还剩几项）；
- 第一项算完就自动切到图表页，**0.4 秒**就能开始看图；
- 数据表页 / 统计量页同样跟着长——不用等；
- 第一项还没算完时，图表区显示「正在计算第一项：回转半径 Rg…」，
  而不是误导性的「还没有结果」；
- 跑完 / 取消后徽标消失，一切回到静态展示。

### 右侧：结果区

五个标签页：**图表 / 数据表 / 统计量 / 导出 / 运行日志**。
每个标签页**内部**按需滚动（图表导航列表、数据表、统计量列表各自滚），
**页面本身不滚动**。

#### 体系信息的「完整报告」是表格，不是文本

「2. 体系信息」里展开「完整报告」看到的是**一串表格**（9 个小节）：

| 小节 | 列 |
|---|---|
| 文件与规模 / 时间轴 / 模拟盒与质量 / 拓扑与属性 | 项目 · 值 |
| 组分构成 | 组分 · residue 数 · 占比 |
| 元素组成 / 常见 residue 名 / 常见原子名 | 名称 · 数量 · 占比 |
| 分子 / 链信息（按组成归类） | 分组 · 条数 · 每条原子数 · 每条 residue 数 · 尺寸分布 |

- 数据来自后端的 `info.sections`（`mdta.systeminfo.info_tables()` 生成，
  形状统一为「列 + 行」），前端不再去解析定宽文本报告；
- **只纵向滚动，横向锁死**：`.inforeport` 用 `overflow: hidden auto`
  （= x 锁死 / y 滚动），表格 `table-layout: fixed` 按比例分配列宽，
  单元格 `overflow-wrap: anywhere` 让长文件名、长分组名**自动换行**，
  不会把表格撑宽逼你左右拖；旧版文本报告也改成 `pre-wrap` 换行；
- 列名如实标注「**每条**原子数 / residue 数」——`ChainType.n_atoms`
  是单条链的规模（取该类最主要的尺寸），不是合计，尺寸不一致时
  另用「尺寸分布」列列出（例如 `11084 条×3 原子 / 11084 条×1 原子`）。

> 运行日志从左侧移到了右侧标签页 —— 否则左侧会被日志撑长、破坏"一屏"。

### 图表页

左侧「图表导航」按**大纲模块**分两层，**点标题即切换右侧的图**；一次只显示一张图。

```text
▾ 链构象                3 项 / 6 张
    ▾ 回转半径 Rg —— 最大链     2
        1. Rg 随时间变化
        2. Rg 分布
    ▾ 端到端距离 R_ee           2
▾ 界面                  4 项 / 9 张
    ▾ 密度分布 …
▾ 结晶                  2 项 / 4 张
▾ 辅助                  1 项 / 2 张
```

- 模块名与分组来自后端 `mdta.pipeline.ANALYSIS_GROUPS`（经 `/api/analyses`
  的 `groups` 下发），前端 `src/analysisGroups.js` 只留一份同内容兜底 ——
  「分析功能」的预设按钮与「图表 / 数据表导航」的分层用的是**同一份定义**，
  不会各写一份而对不上；
- 每个模块可折叠，标题右侧显示「N 项 / M 张」；
- **只显示已经有结果的模块**，所以实时运行时模块会一项项长出来；
- 不在任何模块里的分析项归入「其他」，保证不会漏掉。

> **数据表页的导航是同一套两层结构**（`DataPanel`），模块标题同样可折叠。

图表交互（基础四项）：

| 操作 | 效果 |
|---|---|
| 滚轮 / 拖动底部滑块 | 缩放、平移 |
| 鼠标悬停 | 显示该点的精确数值（十字准星 + 数值框） |
| 点击图例 | 显示 / 隐藏对应曲线 |
| 右上角工具箱 | 保存图片（PNG）、框选缩放、重置 |

图表占满剩余高度。版面用 ECharts 的 `grid.containLabel: true`，
由图表库按标题/坐标轴标签的实际占位自动收缩绘图区，因此**标签不会被容器
裁掉**；双对数图（MSD）会自动切到对数轴并剔除 ≤0 的点。

#### 图表说明移到右侧独立栏（可折叠）

说明文字与关键统计量原来压在图表下方，把图挤得很矮；现在改成整页
**三栏布局**：`[左：操作台] [中：结果区] [右：图表说明]`。

- 右栏顶部是当前图的标题与「第 N / M 张」，中间是**算法说明**与**关键统计量表格**，
  底部提示完整统计量在「统计量」页；
- 内容**跟随当前正在看的那张图**：点导航切换图，右栏同步刷新
  （`current` 由 `App` 统一算好，避免图表区与说明栏各算一套导致不一致）；
- 右栏同样可折叠，**竖条在最右侧**（展开时「收起说明」、收起时「展开说明」），
  折叠后图表变宽；
- 只在「图表」标签页显示 —— 数据表 / 统计量 / 日志页不需要它，直接全宽。
- 图表区因此拿到了整块高度，比原来高出一小截。

**排版（窄栏里最要紧的一条）**：长键名 / 长说明必须**换行**，
不能把栏撑宽去做左右滑动。

- 算法说明用 `.noterow`：`display: flex` + `.notebullet` / `.notetext` 分列，
  形成**悬挂缩进** —— 换行后的文字仍与首行左缘对齐，不会缩到项目符号底下
  （这就是之前看起来「文字错位」的原因）；
  字号 **9.5px**、行高 1.7；项目符号是一个 5px 的实心圆点
  （`.notebullet`，accent 色半透明），按首行行高对齐；
- 关键统计量用 `table.notes-table`：`table-layout: fixed` 固定列宽（键列 62%）、
  单元格 `overflow-wrap: anywhere` 自动换行；覆盖默认 `.stats` 那套
  「每格 10px 内边距 + 键列 55%」的宽屏样式 —— 那套在 264px 窄栏里会把
  长键名挤成参差不齐的多行。字号 11.5 / 12px；
- **横向锁死**：`.notes-scroll` 显式写 `overflow-x: hidden`。
  ⚠️ 只写 `overflow-y: auto` 是不够的：另一轴会由 `visible` 计算成 `auto`，
  内容一超宽就冒出横向滚动条。纵向照常可滚；
- 顺带修掉一个类名撞车：`.notes` 原本既是右侧面板容器、又是说明文字样式，
  现在说明文字用独立类名（右侧面板 `.noterow`、统计量页 `.statnotes`）。

> ⚠️ **高度链必须一路「可压缩」**，而且**选择器一定要和真实 DOM 对上**。
> - 中间那层 `.notes-clip` 必须是 flex 纵向容器并带 `min-height: 0`；
> - `.notes-body` 要用 **`.notes-clip > .notes-body`**（不是 `.notes > .notes-body`），
>   并带 `flex: 1 1 auto; min-height: 0`，**不要用 `height: 100%`**
>   （百分比高度对"由 stretch 决定高度"的父元素解析不出来）；
> - 这个 bug **连续漏过两次**，原因都是只检查了"CSS 里有没有这条规则"：
>   `.notes > .notes-body` 一直好好地写在样式表里，但自从加了 `.notes-clip`，
>   `.notes-body` 就不再是 `.notes` 的直接子元素，**选择器匹配不上**，
>   `display:flex / min-height:0` 一条都没生效 —— 内容被 `overflow:hidden` 裁掉，
>   既看不到也滚不动。
> - 所以现在有一道**真实浏览器**的验证：`webapp/_test_layout.mjs`
>   （Headless Chrome + CDP，量 `getComputedStyle` 与
>   `scrollHeight/clientHeight`、并真的把 `scrollTop` 推到最大确认能滚）。

#### 全屏显示

图表标题栏右侧有「**⤢ 全屏**」按钮，点它把图表区铺满整个屏幕，便于仔细看数据；
再点一次或按 `Esc` 退出。

- 优先用浏览器 **Fullscreen API**（`requestFullscreen`），全屏时标题栏与
  「上一张 / 下一张」仍然可用，可以直接在全屏下翻图；
- 某些环境会拒绝原生全屏（iframe 权限、非用户手势等），此时**自动退化成
  整页覆盖**（`position: fixed; inset: 0`），同样能铺满并支持 `Esc` 退出；
- 两种模式都靠已有的 `ResizeObserver` 让 ECharts 重新适配，不用额外处理。

### 数据表页

左侧是**与图表导航相同的两层目录树**（大纲模块 → 分析项 → 各张表），
右侧把曲线的 x/y 列成表格，可「复制」（TSV，直接粘进 Excel）或「下载 CSV」。

表格本身的取舍：

- **横向滚动保留**：数值表列多了以后单元格换行反而没法读
  （一列数字折成两行比滑动更难对）。所以这里和说明栏**故意不同**，允许左右滑动；
- 但首列（横坐标）**冻结在左边**（`position: sticky; left: 0`），
  往右滚时始终能看到横坐标，不会"滑丢了"；
- 表头 `position: sticky; top: 0` 冻结，纵向滚动时列名一直在；
- 字号 11.5px → 12px、行距略放宽，与说明栏的字号调整保持一致。

### 统计量页 与 导出页（两个同级标签页）

结果区有**五个同级标签页**：`图表 | 数据表 | 统计量 | 导出 | 运行日志`。
统计量与导出**各自独立成一页，职责不混**：

| 标签页 | 只负责 | 不包含 |
|---|---|---|
| **统计量** | 显示统计量指标 | 任何导出控件（目标目录 / 模块 / 格式 / 导出按钮） |
| **导出** | 导出功能 | 任何统计量明细 |

**统计量页**：内部再按大纲模块拆成单一模块标签
（`链构象 | 界面 | 结晶 | 辅助`，默认停在第一个有结果的模块）。

- 一次只显示**一个模块**，不必在长页面里翻；点标签即切换；
- 没有结果的模块会明确提示「这个模块下没有结果（可能没勾选对应分析项）」；
- 每项分析是一张**带表头的表格**（`统计量 | 数值`）；
- 模块分组与图表导航、数据表导航用的是**同一份** `analysisGroups`。

**导出页**：只有导出相关的东西 —— 目标目录、导出模块、文件形式、
开始导出、打开目标目录、最近导出。详见下一节。

> 拆分的意义：以前这两件事挤在同一页，导出设置在上一半、统计量在下一半，
> 想「看一眼指标」也得先路过一堆导出控件。现在各自一页，互不干扰。
> 测试里对这条做了**双向断言**：统计量页不得出现「开始导出/目标目录/文件形式」，
> 导出页不得出现统计量明细，防止以后再粘回去。

### 所有表格的单元格：水平居中 + 垂直居中

`.kv` / `.stats` / `.inforows` / `.notes-table` / `.datatable`
五类表格的单元格统一 `text-align: center; vertical-align: middle`
（含数据表冻结的首列）。这条由**真实浏览器**逐格验证
（`_test_layout.mjs` 会数出每张表的格子数并检查计算后的样式），
不是只看 CSS 里写没写。

### 运行日志页

记录读取体系、运行分析、导出的完整过程（最多保留 300 行），可一键清空。

### 导出区（目标目录 / 模块 / 文件形式 / 打开目录）

```
目标目录   C:\...\outputs\adk_xxx          [选择…] [用默认]
导出模块   [✓链构象] [✓界面] [✓结晶] [✓辅助]  [全选]
文件形式   [✓CSV] [✓PNG] [✓Excel]  dpi [200]  [□每个面板单独出图]
[开始导出]  [📂 打开目标目录]
✓ 最近导出  2026-09-12 19:57:13  C:\...\outputs\adk_xxx（CSV 6 / PNG 3 / summary.xlsx）
```

- **目标目录**：默认是会话输出目录；点「选择…」打开内置目录浏览器
  （复用 `/api/files`，可切盘符、进子目录、回上级）。目录不存在会自动创建。
  「用默认」恢复默认目录。
- **导出模块**：按大纲模块勾选（分组与图表/数据表导航同源）。
  只导出勾选模块下的分析项，标题旁显示「已选 N / M 项」。
- **文件形式**：CSV / PNG / Excel 任选（可多选），另有 dpi 与
  「每个面板单独出图」。一个都不选会直接报错，不会悄悄用默认值。
- **打开目标目录**：排在「开始导出」右边，调 `POST /open-dir` 让**服务器**用
  系统文件管理器打开该目录（Windows 用 `explorer`，macOS `open`，Linux `xdg-open`）。
  没传目录时打开最近一次导出的目录。
- **导出完成后只显示「目录 + 时间 + 计数」**，不再罗列每个文件
  （文件名/大小/下载那张表已去掉；要下载单个文件仍可走
  `GET /api/session/{sid}/files/{name}`）。

> 安全：`/open-dir` 只打开**确实存在的目录**；下载接口只取
> `os.path.basename(name)`，`../` 逃逸会被挡住（实测返回 404）。

---

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 + 能力清单 |
| GET | `/api/analyses` | 可用分析项、中文标题、**大纲模块分组** |
| GET | `/api/files?dir=` | 文件选择器（列目录 / 盘符，按 MD 扩展名过滤） |
| POST | `/api/session` | 打开体系 → `{sid, info, chains, components}` |
| GET | `/api/session/{sid}/info` | 体系信息 |
| GET | `/api/session/{sid}/chains` | 分子/链清单 |
| GET | `/api/session/{sid}/components` | 组分清单 |
| POST | `/api/session/{sid}/select` | 只设置选择（主链 / 组分） |
| POST | `/api/session/{sid}/run` | 跑分析，**同步**等全部跑完 → 全部曲线数据（JSON） |
| POST | `/api/session/{sid}/run/start` | **实时**跑分析：立即返回，后台逐项算 |
| GET | `/api/session/{sid}/run/progress?after=N` | 运行状态 + 客户端**还没有**的结果 |
| POST | `/api/session/{sid}/run/cancel` | 请求取消（当前这项算完后停，已完成的结果保留） |
| POST | `/api/session/{sid}/export` | 导出（可指定目标目录 / 模块 / 文件形式） |
| POST | `/api/session/{sid}/open-dir` | 用系统文件管理器打开目标目录 |
| GET | `/api/session/{sid}/outputs` | 列出已导出的文件 |
| GET | `/api/session/{sid}/files/{name}` | 下载某个导出文件 |
| DELETE | `/api/session/{sid}` | 关闭会话（释放并清理导出目录） |

### 实时更新（边算边出）

Web 界面用的是 `/run/start` + `/run/progress` 这一对，而不是同步的 `/run`：

```
POST /run/start              → 立即返回 {status:"running", n_total:10, pending:[...]}
GET  /run/progress?after=0   → 500ms 后：拿到 rg、ree 两项 + frac/message
GET  /run/progress?after=2   → 又拿到 dihedral、density……（不重传前两项）
GET  /run/progress?after=10  → {status:"done"}
```

要点：

- **逐项推送**：`mdta.pipeline.Analyzer.run_all` 新增 `on_result(name, res, ...)`
  回调，每算完**一项**就立刻序列化写进会话结果集，页面马上就能画那张图；
- **增量拉取**：`after` 是客户端已收到的条数（按完成顺序），后端只回传新增的那几项，
  所以轮询的数据量与"已完成多少"无关，不会重复传几百 KB 的曲线；
- **真实进度**：进度条来自后端的 `frac`（已完成项数 / 总项数）与 `message`，
  不是前端定时器假装的；`pending` 会列出还没算的分析（用中文标题）；
- **取消是真取消**：`/run/cancel` 置位一个 `threading.Event`，分析循环在**每一项开始前**
  检查它并停止；**已完成的结果全部保留**，不会白算；
- **运行期间锁住 Universe**：`select` / 再次 `start` / `export` 在运行中会返回 400
  （`分析正在运行中…`），因为 MDAnalysis 的 Universe 不能被两个线程同时用。

> 实测（Abeta_4_16_Cu，40 896 原子，10 项分析，总 18.3 s）：
> 第一项结果在 **0.4 s** 就到达，10 项分散在整个 18.3 s 里陆续出现
> （`rg/ree 0.4s → dihedral 0.8s → density 1.3s → rdf 6.8s → contact 16.2s
> → interface 16.6s → orientation 17.0s → order 17.5s → msd 18.3s`）。
> 以前要干等 18 s 才看到第一张图，现在 0.4 s 就能开始看图。

`/run/start` 的请求体与同步的 `/run` **完全相同**：

```jsonc
{
  "which": ["rg", "ree", "dihedral", "density", "rdf",
            "contact", "interface", "orientation", "order", "msd"],
  "frames":  { "start_ps": null, "stop_ps": null, "equil_ps": null,
               "interval_ps": 100, "max_frames": null },
  "params":  { "density":   { "axis": 2, "nbins": 100 },
               "interface": { "axis": 2, "nbins": 120 },
               "rdf":       { "rmax": 12, "nbins": 120 },
               "contact":   { "cutoff": 5 },
               "dihedral":  { "mode": "auto" } },
  "selection": {
    "primary":    { "mode": "auto" },          // auto | component | chain | custom
    "components": ["protein", "water"]         // 不传则用自动识别的全部组分
  }
}
```

`/run/progress` 的返回（`results` 里只有 `after` 之后完成的项）：

```jsonc
{
  "status": "running",            // running | done | error | cancelled | idle
  "frac": 0.4, "message": "已完成 4/10：径向分布函数 g(r)",
  "n_total": 10, "n_done": 4,
  "pending": ["contact", "interface", "orientation", "order", "msd"],
  "completion": ["rg", "ree", "dihedral", "density"],   // 完成顺序
  "results": { "rdf": { /* 与 /run 里同样的结果对象 */ } },
  "summary_rows": [ /* 汇总表，始终完整 */ ],
  "timings": { "rg": 0.9, "ree": 0.1, ... },
  "elapsed_sec": 6.8, "error": ""
}
```

返回的每个分析结果形如：

```jsonc
{
  "name": "rg",
  "title": "回转半径 Rg —— 最大链 (seg_0_AKeco, 3341 原子)",
  "n_panels": 2,
  "panels": [ { "index": 0, "title": "Rg 随时间变化",
                "xlabel": "时间 Time (ps)", "ylabel": "回转半径 Rg (Å)",
                "xscale": "linear", "yscale": "linear", "legend": true,
                "curves": ["最大链 …", "平均 Rg"] } ],
  "curves": [ { "label": "最大链 …", "kind": "line", "panel": 0,
                "x": [...], "y": [...], "n": 10, "downsampled": false } ],
  "summary": { "Rg mean": 19.6658, "Rg std": 0.1986, ... },
  "notes":   ["质量加权: True；PBC 展开: True", ...]
}
```

> 曲线超过 4000 点时后端会等间隔抽稀并置 `downsampled: true`（不影响导出，
> 导出走的是完整数据）。

---

## 目录结构

```text
webapp/
├── backend/
│   ├── app.py          FastAPI 应用：路由 + 静态托管
│   ├── session.py      会话管理（已打开体系、结果、导出目录）
│   ├── serialize.py    AnalysisResult → JSON
│   └── run.py          启动入口（uvicorn）
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 主界面：三块互斥折叠面板 + 四个结果标签页
│   │   ├── api.js                  接口封装
│   │   ├── chartOption.js          面板+曲线 → ECharts option
│   │   ├── styles.css              一屏布局与折叠面板样式
│   │   └── components/
│   │       ├── OpenPanel.jsx       ① 文件读取（目录浏览 + 选文件）
│   │       ├── InfoPanel.jsx       ② 体系信息
│   │       ├── SetupPanel.jsx      ③ 分析设置：
│   │       │                          SelectionBlock 分析对象
│   │       │                          ParamBlock     参数设置
│   │       │                          FunctionBlock  分析功能
│   │       │                          RunBlock       运行
│   │       ├── ChartPanel.jsx      图表导航 + 单图
│   │       ├── ChartView.jsx       ECharts 容器
│   │       ├── DataPanel.jsx       数据表视图
│   │       ├── SummaryView.jsx     统计量（卡片 / 表格）
│   │       ├── StatsPanel.jsx      「统计量」页（只显示指标）
│   │       ├── ExportPanel.jsx     「导出」页（只管导出）
│   │       └── LogPanel.jsx        运行日志
│   ├── _ssr_entry.js / _ui_render.mjs   渲染自检（85 项）
│   ├── _build.mjs                   生产构建（不经 vite.config.js 加载器）
│   └── dist/                       构建产物（FastAPI 托管）
├── _e2e.mjs            端到端验证（真实 HTTP）—— AdK 体系
├── _e2e_polymer.mjs    端到端验证 —— 糖蛋白 + DOL 体系
├── _api_check.py       后端 API 冒烟测试（TestClient）
├── _test_abeta.py      板层界面体系 + 拓扑格式（gro/pdb/top）路径验证
├── _test_formats.py    .tpr / .gro / .pdb 跨格式一致性
├── _test_live.py       实时运行接口（TestClient）
├── _test_live_http.py  实时运行接口（真实 HTTP + 板层体系）
├── _css_check.py       构建产物的布局 / 折叠面板样式检查
├── _font_check_all.py  导出图缺字 / 后端检查
├── _getwheel.mjs       离线拉取 Python wheel 的小工具
├── requirements.txt
└── README.md
```

---

## 自检

```powershell
# 后端 API（不用起服务，直接 TestClient）
python webapp\_api_check.py

# 端到端（需先启动服务）
python -m webapp.backend.run            # 终端 1
node webapp\_e2e.mjs                    # 终端 2

# 前端组件渲染（React SSR，无需浏览器）
cd webapp\frontend
node _ui_render.mjs                     # 203 项（含导航分层、统计量板块、说明排版、全屏、信息表格、快捷预设）
node _build.mjs                         # 生产构建（不经 vite.config.js 加载器）

# 构建产物里折叠面板 / 一屏布局的样式检查（需先启动服务）
python webapp\_css_check.py             # 62 项

# 真实两组分体系（糖蛋白 + DOL，301 帧）
node webapp\_e2e_polymer.mjs            # 29 项

# 导出图字体 / 后端检查
python webapp\_font_check_all.py

# 板层界面体系（Aβ4-16+Cu / 水 / DCE）+ 拓扑格式（gro/pdb/top）路径
python webapp\_test_abeta.py

# 拓扑格式一致性（.tpr / .gro / .pdb 的键表、组分识别必须一致）
python webapp\_test_formats.py          # 28 项

# 导出功能（目标目录 / 模块 / 文件形式 / 打开目录 / 只报目录+时间）
python webapp\_test_export.py            # 20 项（真实 HTTP）

# 实时运行接口（边算边出、增量拉取、运行期加锁、取消保留结果）
python webapp\_test_live.py             # 22 项（TestClient，不用起服务）
python webapp\_test_live_http.py        # 11 项（真实 HTTP，跑 Abeta 板层体系）

# 真实浏览器布局验证（Headless Chrome + CDP，量 computed style）
node webapp\_test_layout.mjs            # 22 项
```

> **`_test_layout.mjs` 需要先起一个带调试端口的 Chrome。**
> ⚠️ 受限沙箱里 Chrome **起不来**：它要用 Mojo 创建命名管道，
> 会直接崩在 `platform_channel.cc: Check failed: 拒绝访问 (0x5)`。
> 所以这条启动命令必须用 `danger-full-access` 运行：
>
> ```powershell
> Start-Process 'C:\Program Files\Google\Chrome\Application\chrome.exe' -ArgumentList `
>   '--headless=new','--remote-debugging-port=9222',
>   '--user-data-dir=C:\temp\MDT\webapp\_probe\chrome-profile','about:blank'
> ```
>
> 连不上浏览器时该测试会**打印跳过信息并返回 0**，不会把正常套件搞挂。


> `_ui_render.mjs` 与 `_build.mjs` 都用**编程式** Vite API 并设置 `configFile: false`。
> 原因：`vite build` / `vite` CLI 会先 `loadConfigFromFile`，其中
> `windowsSafeRealPathSync` 会 `exec` 子进程解析真实路径，在受限沙箱里该 spawn
> 会以 `EPERM` 失败。`_build.mjs` 把 `vite.config.js` 的内容原样内联，
> 产物与 `npm run build` 一致。
>
> `_e2e*.mjs` 结尾用 `process.exitCode = ...` 而**不用** `process.exit()`：
> `fetch`(undici) 的 keep-alive 套接字可能正在关闭，强制退出会触发 libuv 断言
> （`STATUS_STACK_BUFFER_OVERRUN`，退出码 `0xC0000409`），出现"明明 29 项全通过
> 却报失败"的假阴性。改用 `exitCode` 让事件循环自然排空即可（已连续多次验证退出码为 0）。

实测结果（三套真实轨迹）：

**A. `adk_oplsaa`（AdK 蛋白水溶液，47 681 原子 / 10 帧）**

| 检查 | 结果 |
|---|---|
| `_api_check.py` | 全部通过（10 项分析、48 条曲线、导出 23 CSV + 33 PNG） |
| `_e2e.mjs` | **35 / 35 通过**（含静态前端、下载 PNG/CSV、错误处理、会话生命周期） |
| `_ui_render.mjs` | **232 / 232 通过**（含五个标签页与顺序、统计量页不含导出控件、导出页不含统计量、导出区四项、顶栏、全屏、null 值显示） |
| `_test_export.py` | **20 / 20 通过**（自定义目标目录、模块与格式选择、参数校验、打开目录、路径逃逸防护、只回报目录+时间） |
| `_css_check.py` | **62 项通过**（含顶栏深靛紫底与**对比度断言**、标签栏、导出区样式；并反向检查旧样式已移除） |
| `_test_live.py` | **22 / 22 通过**（逐项到达、增量拉取、运行期加锁、取消保留结果、同步接口兼容） |
| `_font_check_all.py` | 缺字 0、警告 0 |

**B. `md_biopolymer_nowater`（糖蛋白 + 648 个 DOL + 5 Cl⁻，24 616 原子 / 301 帧 / 300 ns）**

| 检查 | 结果 |
|---|---|
| `_e2e_polymer.mjs` | **29 / 29 通过**（61 帧全 10 项分析，约 3 分钟） |

关键输出：

- 组分识别：`protein(3035)` / `sugar(192)` / `polymer(21384)` / `ion(5)`
  —— 8 个糖残基（`BGLCNA`/`AFUC`/`AMAN`/`BMAN`）被正确拆成独立的 `sugar` 组分，
  648 个 DOL 才是 `polymer`；自动主链选到整条糖蛋白（3227 原子，含糖链）
- Rg = 18.83 ± 0.15 Å（块平均标准误 0.05）
- 密度：polymer(DOL) 1.016、protein 0.157 g/cm³
- 蛋白–聚合物接触：平均 263 对，接触概率 1.0
- **MSD：DOL 的 MSD 曲线完全平坦（300 ns 内位移仅 ~1.8 Å），
  工具返回 `D = nan` 并说明"粒子基本没有扩散"，而不是给一个无意义的负值；
  蛋白 D = 4.8×10⁻¹² m²/s（拟合良好）**

**C. `Abeta_4_16_Cu`（Aβ4-16+Cu 肽 / 水 / DCE 板层体系，40 896 原子 / 501 帧 / 100 ns）**

用 `.gro` 作拓扑（**无键表 → 自动猜键**），10 项分析全部产出，约 22 秒：

- 组分识别：`protein(220)`（13 残基，含 CHARMM 的 `MARG`/`MHSE`）、
  `polymer(2055)`（DCE 等）、`water(7819)`、`ion(56)`，`other` 为空
- 沿 c 方向的密度分布确认为**分层板层**，且**肽吸附在水/DCE 界面上**
- 界面分析给出**两个界面**，各自局部测量：

  | | 界面 1（有肽） | 界面 2 |
  |---|---|---|
  | 位置 | 109.32 Å | 198.86 Å |
  | 宽度 10–90 | 9.02 Å | 8.45 Å |
  | erf 拟合宽度 | 9.59 Å | 9.19 Å |
  | R² | 0.9999 | 0.9998 |

- MSD 因**帧间隔 2 ns** 过大，最小镜像失效比例达 57–68%，
  工具把水/DCE/Cu²⁺ 的 D 全部标注为「**不建议引用**」并建议缩小帧间隔
  ——见注意事项第 2 条

**D. 拓扑格式一致性（`_test_formats.py`，28 / 28 通过）**

`_fmt_test/` 里是从 `.tpr` 转换出的同名 `.gro` 与 `.pdb`，用来验证跨格式一致性：

| 检查 | AdK | 糖蛋白 + DOL |
|---|---|---|
| `.tpr` 键数 / 片段数 | 25 533 / 22 173 | 24 020 / 654 |
| `.gro`（**自动猜键**） | 25 533 / 22 173 ✓ | 24 020 / 654 ✓ |
| `.pdb`（**CONECT 记录**） | 25 533 / 22 173 ✓ | 24 020 / 654 ✓ |
| 组分识别是否一致 | ✓ | ✓（含糖链） |

> 这个测试抓出了两个真问题：
> 1. **`.gro` 把 residue 名截断到 5 字符**，`BGLCNA` 变成 `BGLCN`，导致糖残基
>    掉进 `polymer` 兜底类别（sugar 192→83 原子）。已在 `is_sugar_resname` 里
>    增加"截断兼容"匹配，现在两种格式的组分识别完全一致。
> 2. 测试最初误判 `.pdb` "没有猜键信息"——实际是 `CONECT` 记录在**文件末尾**，
>    只扫前若干字节会漏掉。`.pdb` 自带键表时本就不该猜键。

**E. 实时运行（`_test_live.py` / `_test_live_http.py`）**

| 检查 | 结果 |
|---|---|
| `POST /run/start` 是否立即返回 | 0.10–1.06 s（不等分析跑完）✓ |
| 结果是否分散到达 | 首尾相差 14.5–17.9 s，不是一次性 ✓ |
| 第一项结果何时可用（Abeta，总 18.3 s） | **0.4 s** ✓ |
| `after=N` 是否增量 | `after=10` 回传 0 条；`after=7` 只回传最后 3 条 ✓ |
| 运行中改选择 / 再次 start | HTTP 400 `分析正在运行中…` ✓ |
| 取消 | 状态转 `cancelled`，已完成结果保留 ✓ |
| 同步 `/run` 向后兼容 | 仍返回全部 10 项且字段完整 ✓ |

> 这个测试抓出一个**死锁**：`cancel_run()` 在持有 `threading.Lock` 的情况下调用了
> `run_progress()`，而后者会再次获取同一把锁 —— 普通 `Lock` 不可重入，请求永久挂起
> （实测卡死，10 分钟超时）。修法是改成 `RLock`，并把取锁区间收紧。
> 这类 bug 在单线程测试里看不出来，必须用"发起请求 → 立刻轮询"的顺序才能撞到。


---

## 接触分析的同组 / 异组（重要）

| 情形 | 配对来源 | 配对含义 |
|---|---|---|
| 异组 A ≠ B | `capped_distance(A, B)` | A×B 笛卡尔积，每对 (i∈A, j∈B) 一次 |
| 同组 A = B | `self_capped_distance(A)` | 无序对，**不含 i==i 自配对**，每对 {i,j} 只记一次 |

早先不分同组异组、一律 `capped_distance(A, A)`，有三个后果：

1. 混进 N 个**距离为 0 的 i==i 自配对** → `最小原子间距` **恒为 0**；
2. 每对无序对 {i,j} 被记两次（(i,j) 与 (j,i)）→ **接触对数翻倍**；
3. **慢得多** —— 实测 21384 原子时 `capped_distance(A,A)` 1.01 s/帧
   vs `self_capped_distance(A)` 0.186 s/帧。

> 实测（Abeta，water 23457 原子，同组）：
> 接触对数 1,210,900 → **593,721**、最小间距 0.0000 → **0.9439 Å**、
> 耗时 2.578 → **0.134 s/帧（19×）**。
> 结果里会明确写出「配对方式：同组 A–A / 异组 A–B」，并在说明里注明
> 同组的统计约定，避免把两种口径的数字混着比。

另外，`最小原子间距` 不再**重复扫描**：`capped_distance(cutoff)` 已经拿到
cutoff 内所有原子对，只要非空，其中最小距离就是全局最小（cutoff 外的只会更大），
无需再调 `min_distance` 放大搜索一遍。只有"这一帧完全没有接触"时才需要它。

## 展开坐标缓存（order / dihedral 的主要开销）

`positions_for()` 里的 `ag.unwrap()` 占了 order / dihedral 一半以上的时间，
而这两个分析在一次遍历里会对**同一帧、同一个组**反复取展开坐标（各 4 次）。
现在按 `(组指纹, 帧号, compound, 坐标哈希)` 缓存：

- **必须把坐标本身放进键里**，不能只认帧号 —— 调用方可能不换帧直接改
  `ag.positions`（合成体系的测试就是这么做的），只认帧号会命中过期结果；
  坐标哈希的代价比 unwrap 低三个数量级，换来实现上的绝对安全。

> 实测（md_biopolymer_nowater，主链 3227 原子，301 帧）：
> dihedral 97.3 s → **3.8 s（25×）**；order 124 s → **74 s（1.7×）**。

## 与 CLI / 桌面版的关系

三者共用同一套分析代码（`mdta` 包），结果数值完全一致：

| 入口 | 图表 | 适合场景 |
|---|---|---|
| `mdta_cli.py run` | 导出 PNG（可 `--panel-pngs` 一图一文件） | 批量、脚本化、服务器上跑 |
| `mdta_gui.py` | 界面内不画图，数据表 + 导出 PNG | 本机操作台，看数值、导数据 |
| **Web 版** | 浏览器内交互图（缩放/平移/hover/图例） | 看图、演示、给导师看、远程访问 |

---

## 注意事项

1. **界面宽度分析只对"分层/界面"体系有物理意义**。工具会自己判断并给出
   「界面判据可靠性」：
   - **高**：密度曲线是台阶状，erf 拟合通过 → 界面位置与宽度可用；
   - **低**：组分是**分散/混合**的（例如蛋白溶解在 DOL 里），此时没有平界面，
     给出的"界面宽度"只是密度涨落的特征尺度。
   > 实测的 `md_biopolymer_nowater` 与 `adk_oplsaa` 都属于第二类（蛋白分散在
   > DOL / 水中），工具如实标注为"低"；而 `Abeta_4_16_Cu` 属于第一类，
   > 标注为"高（板层体系，检测到 2 个界面，均已单独测量，erf 拟合通过）"。
   >
   > **板层体系一定有两个界面**（周期性），工具会把两个都按 z 顺序列出并**各自
   > 局部测量宽度**，不会把两个界面混在一起测出一个大得多的宽度。
2. **扩散系数不会给负值**。MSD 平坦（粒子不动、玻璃态、被约束）时返回 `nan`
   并在「拟合可靠性」里说明原因；α 明显偏离 1 时也会标注"仅作定性参考"。
   **此外还会看「最小镜像失效比例」**：该误差是随机的，会让 MSD 依旧平滑线性、
   R² 接近 1，所以只看 R²/α 会误判。失效比例 > 5% 时降级提示，> 20% 时标注
   「不建议引用」。
   > 实测 `Abeta_4_16_Cu` 帧间隔 2 ns → 失效 57–68%，水与 DCE 的 D 几乎相同
   > （3.56 vs 3.56 ×10⁻¹⁰ m²/s，典型的回绕噪声主导征兆），工具如实标注不可引用。
   > `adk_oplsaa` 帧间隔 100 ps → 失效 1.0%，D = 3.5×10⁻⁹ m²/s 可信。
3. **统计量里的"不可测"值显示为「—」**：后端把 `NaN` 序列化成 JSON `null`，
   前端统一用 `formatCell` 渲染，不会出现字面量 `null` / `NaN`。
4. **只监听 127.0.0.1**：默认仅本机可访问。改成 `--host 0.0.0.0` 可局域网访问，
   但**没有任何鉴权**，而且 `/api/files` 能浏览服务器文件系统，请只在可信网络里用。
5. **会话有上限**：进程内最多保留 8 个会话、闲置 6 小时自动回收（导出目录一并清理）。
   重启服务会丢掉所有会话，需要重新打开体系。
6. **导出写在服务器上**：路径是 `webapp/outputs/<体系名>_<sid>/`；界面会把文件
   列出来供下载，也可以直接去那个目录拿。
7. **图表是前端渲染的**：后端不生成图片，所以导出 PNG 走的是 matplotlib
   （服务器端），界面里看到的是 ECharts，两者数值相同、外观略有差别。
8. **组分类型会自动识别**：`protein` / `nucleic` / `sugar` / `polymer` / `water` /
   `ion`。其中 `sugar` 覆盖 GLYCAM 风格命名（`AFUC`、`BMAN`、`AMAN` 等带 α/β
   前缀的糖），`protein` 额外覆盖 CHARMM 变体（`MARG`、`MHSE`、`MHSD` 等），
   `polymer` 是兜底类别（未被识别为前几类的待研究组分，
   高分子链和不认识的有机小分子都会落进来）。糖蛋白的糖链会单独成为
   `sugar` 组分；若要把糖链算进蛋白一起分析，直接用 `segid` 选择即可。
9. **离线装依赖**：如果 pip 源不通（本机就遇到过），可以用
   `node webapp\_getwheel.mjs webapp\_wheels fastapi` 从 PyPI 下 wheel，
   再 `pip install --no-deps --find-links webapp\_wheels fastapi`。
   仓库里已经带了一份 `_wheels/fastapi-0.141.1-py3-none-any.whl`。
   npm 依赖用 `webapp/frontend/.npmrc` 里的 `cache=.npm-cache` 避开权限问题。
