/**
 * 中间显示区的「三维结构」页：整块区域画分子（Three.js），左侧操作台只放控制项。
 *
 * 为什么放这里而不是塞进左侧小面板：三维结构需要大画布才有意义（旋转、观察、
 * 点选原子）。所以左侧「3. 结构选择」只放工具栏 / 层级树 / 选择信息 / 分析集合，
 * 真正的画面铺在这个标签页里，和「图表」同级。
 *
 * 与左侧共享状态：`st` 来自 `useStructureState()`（App 里只创建一次）。
 */
import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { STRUCT_MODES } from '../structureState'

//: CPK 配色（常见元素；未知元素用粉色，与 PyMOL 习惯一致）
const CPK = {
  H: '#ffffff', C: '#909090', N: '#3050f8', O: '#ff0d0d', F: '#90e050',
  P: '#ff8000', S: '#e6c83c', CL: '#1ff01f', BR: '#a62929', I: '#940094',
  NA: '#ab5cf2', K: '#8f40d4', LI: '#cc80ff', MG: '#8aff00', CA: '#3dff00',
  ZN: '#7d80b0', FE: '#e06633', SI: '#f0c8a0', B: '#ffb5b5',
}
const HIGHLIGHT = '#ff2d55'

/* ---------------------------------------------------------- 球棍模型（参考 Material Studio）
 * 材质习惯：**球**按共价半径缩一圈当"原子球"，**棍**沿键方向画，且两端各取原子配色
 * （半根一色），这样一根 C–H 键一半灰一半白，和 MS 里看起来一致。
 *   球半径 = 共价半径 × RADIUS_SCALE      （0.38 → C≈0.29 Å、H≈0.12 Å）
 *   棍半径 = min(STICK_RADIUS, 0.55 × 该端球半径)
 *
 * 屏幕空间下限是必要的：几万个原子按真实半径画只有不到 1 个像素，看上去"什么都没有"。
 *
 * ⚠️ 棍半径**必须由该端的球半径封顶**，这是修过的一个 bug：原来棍半径只按
 *    "0.9 像素"设下限，缩到全景时 H 的球半径被下限顶到 0.23 Å，而棍半径也有
 *    0.21 Å —— 两者几乎一样粗，于是氢原子看上去不是一颗球，而是"一根白棍"
 *    （用户实测："氢原子并没有以球状显示而是一根不同颜色的棍子"）。
 *    现在棍永远只有球的 55%，球一定露在棍外面。
 */
const RADIUS_SCALE = 0.38          // 球半径 / 共价半径
const STICK_RADIUS = 0.13          // 棍半径上限（Å）
const STICK_BALL_RATIO = 0.55      // 棍半径 ≤ 该端球半径 × 0.55（保证球不被棍吞掉）
const BALL_MIN_PX = 1.3            // 球半径的屏幕下限（像素）
const ROT_PER_PX = 0.004           // 旋转灵敏度（rad / 像素）：用户要求调低（原 0.008）
const PAN_PER_PX = 0.0009          // 平移灵敏度（视野半径 / 像素）：原 0.0016
const ZOOM_PER_DELTA = 0.0016      // 滚轮缩放灵敏度（rad 的指数）：原 0.0011，太慢
//: 最近能放大到什么程度：**最大那颗原子直径 ≈ 视野高的 BALL_FILL 倍**。
//: 1.8 倍 = 相机能钻进原子内部（用户要求"可以穿透不同的分子、一直放大到单个分子"）。
const BALL_FILL = 1.8
//: 最远能缩小到什么程度：**整个模拟盒占视野的 1/3**（用户要求，防止无限缩小）
const BOX_FILL = 1 / 3
//: 没有模拟盒、或算不出来时的兜底上下限（Å）
const MIN_RADIUS_A = 0.01
const MAX_RADIUS_A = 20000
//: 双击原子后停下来的相机距离（Å）：约在"一条键"尺度上，正好看清一个分子内部
const FOCUS_RADIUS_A = 2.2

/* 模拟盒：白色细柱（用户要求"白色条状"）。
 * 用圆柱而不是 LineSegments：白色细线在浅色背景上几乎看不见，圆柱有明暗边。
 *
 * ⚠️ 粗细**只按屏幕像素**定，绝不能再用"Å 绝对值下限"：
 *    原来写的是 max(0.2 Å, 1.8 像素)，放大后 0.2 Å 那一条会占主导 ——
 *    盒棱变成 0.2 Å 粗的白柱，一旦离相机近些就糊满整屏，看起来就像
 *    "缩放被盒子卡住了、放不大"（用户实测）。现在它永远是 1.8 像素的细线。 */
const BOX_COLOR = '#ffffff'
const BOX_PX = 1.8                 // 盒棱半径（屏幕像素，不随缩放变粗）

/**
 * 按**当前相机距离**重算"每像素多少 Å"和球/盒棱的屏幕尺度。
 *
 * 为什么不能只在建模时算一次：屏幕下限一旦固定在初始（全景）视角的值，
 * 用户放大后就还是那副"球大得把棍吞掉"的样子 —— 球棍模型永远看不出来。
 * 所以缩放时一起重算：**拉远时靠下限保住可见性，靠近时让真实比例接管**。
 */
export function recalcScales(st) {
  const viewH = Math.max(1, (st.el && st.el.clientHeight) || 420)
  const radius = (st.sph && st.sph.radius) || 100
  const pxToA = (2 * radius * Math.tan(Math.PI / 8)) / viewH   // 45°/2
  st.pxToA = pxToA
  st.minRadA = BALL_MIN_PX * pxToA
  // 盒棱：纯屏幕尺寸 → 任何缩放级别都是同一条细线，不会挡住原子
  st.boxRadA = BOX_PX * pxToA
  return pxToA
}

/**
 * 缩放的上下限：**按体系自身尺寸算**，不是硬编码常数。
 *
 * · 最近（`radiusMin`）：最大的那颗原子直径 ≈ 视野高的 3/4 —— 再近就只是在
 *   看一个原子的内部，没有意义；用户要求"单个原子可以铺满区域四分之三"。
 * · 最远（`radiusMax`）：整个模拟盒 ≈ 视野的 1/3 —— 防止无限缩小（用户要求）。
 *   若原子铺得比盒子还开（坐标没按周期折回），则至少保留"能回到初始视角"，
 *   否则一加载就被自己的限制卡住。
 */
function computeZoomLimits(st, data) {
  const tanHalf = Math.tan(Math.PI / 8)
  // ---- 最远：盒子在画面里占 1/3 → 2·boxR = BOX_FILL·viewH = BOX_FILL·2r·tanHalf
  let boxR = 0
  const vecs = cellVectors(data.box)
  if (vecs) {
    const [A, B, C] = vecs
    boxR = Math.hypot(A[0] + B[0] + C[0], A[1] + B[1] + C[1],
      A[2] + B[2] + C[2]) / 2          // 体对角线的一半
  }
  const homeR = st.home ? st.home.radius : 100
  st.boxRadius = boxR
  st.radiusMax = boxR > 0
    ? boxR / (BOX_FILL * tanHalf)
    : homeR * 1.6
  if (boxR > 0) st.radiusMax = Math.max(st.radiusMax, homeR)   // 别比初始视角还近
  st.radiusMax = Math.min(st.radiusMax, MAX_RADIUS_A)

  // ---- 最近：最大球直径 = BALL_FILL·viewH
  let maxBall = 0.3
  for (let i = 0; i < data.n_displayed; i += 1) {
    const r = (data.radii[i] || 1.4) * RADIUS_SCALE
    if (r > maxBall) maxBall = r
  }
  st.maxBallA = maxBall
  st.radiusMin = maxBall / (BALL_FILL * tanHalf)
  st.radiusMin = Math.min(st.radiusMin, homeR * 0.5)          // 别比初始视角还远
  st.radiusMin = Math.max(st.radiusMin, MIN_RADIUS_A)
  return st.radiusMin
}

/** 球半径（Å）：共价半径 × 系数，并受屏幕下限保护 */
function ballRad(st, data, i) {
  return Math.max(0.03, Math.min(2.5,
    Math.max((data.radii[i] || 1.4) * RADIUS_SCALE, st.minRadA || 0)))
}

/** 棍半径（Å）：**由该端球半径封顶** —— 否则小原子（H）会被棍吞掉 */
function stickRad(st, data, i) {
  return Math.min(STICK_RADIUS, STICK_BALL_RATIO * ballRad(st, data, i))
}

function colorOf(el) {
  const key = String(el || '').toUpperCase()
  return CPK[key] || CPK[key[0]] || '#ff69b4'
}

export default function StructureView({ st }) {
  const [fs, setFs] = useState(false)
  const { data, loading, selected, hover, err, nAtoms } = st

  // 第一次切到这个标签页且还没载入过 → 自动取一次（只读拓扑/首帧坐标）
  useEffect(() => {
    if (st.sid && !data && !loading) st.load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [st.sid])

  // 悬停的原子变了：DOM 里多/少一个原子信息框，画布必须**同一帧**重画，
  // 否则会出现"白画布配新提示框"的一闪（用户实测的闪白屏）。
  useEffect(() => {
    st.viewerRef?.current?.kick?.()
  }, [hover])

  // Esc 退出全屏（与图表页一致）
  useEffect(() => {
    if (!fs) return undefined
    const onKey = (e) => { if (e.key === 'Escape') setFs(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [fs])

  return (
    <div className={`structview${fs ? ' fs' : ''}`}>
      <div className="structview-head">
        <div>
          <div className="charttitle">三维结构</div>
          {/* 小字只留两个数：体系原子数 / 当前已选原子数（其余细节放 tooltip） */}
          <div className="chartsub"
               title={data ? [
                 `显示 ${data.n_displayed.toLocaleString()} 个原子`
                 + (data.truncated
                   ? `（按整残基截断：${data.n_residues_shown.toLocaleString()} / `
                     + `${data.n_residues.toLocaleString()} 残基）` : ''),
                 data.n_bonds ? `球棍模型（${data.n_bonds.toLocaleString()} 根键）`
                   : '无键表，仅画球（勾选“按坐标猜键”后重开体系可显示球棍）',
                 data.box && data.box.length >= 3
                   ? `模拟盒 ${data.box.slice(0, 3)
                     .map((x) => Number(x).toFixed(1)).join('×')} Å`
                     + (data.box_wrapped === false && data.box_frac_inside !== null
                       ? `（只有 ${(data.box_frac_inside * 100).toFixed(0)}% 的原子在盒内：`
                         + '坐标未按周期折回）' : '')
                   : '无模拟盒',
               ].join('\n') : ''}>
            {nAtoms.toLocaleString()} 原子 · 已选 {selected.size.toLocaleString()} 原子
          </div>
        </div>
        <div className="charttools">
          {hover !== null && (
            <span className="dim" style={{ fontSize: 10.5 }}>悬停：#{hover}</span>
          )}
          <button className="mini" onClick={() => st.load()} disabled={loading}>
            {loading ? '读取中…' : '重新载入'}
          </button>
          <button className="mini" onClick={st.resetView}>重置视角</button>
          {(data?.rings?.length || 0) > 0 && (
            <label className="mini"
                   style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
                   title="芳香六元环用环心虚线圆标出（苯环判定：6 元碳环 + 环内键长 ≤1.50 Å）">
              <input type="checkbox" defaultChecked
                     onChange={(e) => {
                       st.showRings = e.target.checked
                       if (st.ringGroup) st.ringGroup.visible = st.showRings
                     }} />
              芳香环 {data.rings.length}
            </label>
          )}
          <button className="mini fsbtn" onClick={() => setFs((v) => !v)}
                  title={fs ? '退出全屏（Esc）' : '全屏查看结构'}>
            {fs ? '⤡ 退出全屏' : '⤢ 全屏'}
          </button>
        </div>
      </div>

      {err && <div className="error">{err}</div>}

      <div className="structview-body">
        {!data && !loading && (
          <div className="placeholder">
            <p>还没有载入结构。</p>
            <p className="dim">切到「3. 结构选择」点「载入结构」，或直接点「重新载入」。</p>
          </div>
        )}
        {data && (
          <Viewer data={data} selected={selected}
                  onlySelected={!!st.onlySelected}
                  onPick={st.pick} onHover={st.setHover} binding={st.viewerRef}
                  onError={st.setErr} />
        )}
        {data && hover !== null && (
          <AtomTip data={data} atom={hover} />
        )}
        {data && (
          <div className="viewhint">
            球棍模型 · 左键拖动旋转 · 滚轮缩放 · **双击原子钻进分子** · 右键拖动平移 · 单击原子高亮（Ctrl 单击加减）
          </div>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ 悬停信息 */
function AtomTip({ data, atom }) {
  const i = data.index.indexOf(atom)
  if (i < 0) return null
  const xyz = data.xyz[i].map((v) => v.toFixed(2)).join(', ')
  return (
    <div className="atomtip">
      <b>#{atom}</b> {data.elements[i]} {data.names[i]}
      <br />残基 {data.resname[i]}{data.resid[i]} · 链 {data.segid[i]}
      <br />分子 #{data.molid[i]} · ({xyz}) Å
    </div>
  )
}

export function buildGeometry(st, data) {
    const { scene } = st
    for (const m of st.meshes) {
      scene.remove(m)
      m.geometry.dispose()
      m.material.dispose()
    }
    st.meshes = []
    if (st.bondMesh) {
      scene.remove(st.bondMesh)
      st.bondMesh.geometry.dispose()
      st.bondMesh.material.dispose()
      st.bondMesh = null
    }
    if (st.bondLines) {
      scene.remove(st.bondLines); st.bondLines.geometry.dispose(); st.bondLines = null
    }
    if (st.boxMesh) {
      scene.remove(st.boxMesh)
      st.boxMesh.geometry.dispose()
      st.boxMesh.material.dispose()
      st.boxMesh = null
    }
    st.boxInfo = null
    if (st.labelGroup) { scene.remove(st.labelGroup); st.labelGroup = null }
    if (st.ringGroup) {
      scene.remove(st.ringGroup)
      st.ringGroup.geometry.dispose()
      st.ringGroup.material.dispose()
      st.ringGroup = null
    }

    const n = data.n_displayed
    const xyz = data.xyz
    let cx = 0, cy = 0, cz = 0
    for (let i = 0; i < n; i += 1) { cx += xyz[i][0]; cy += xyz[i][1]; cz += xyz[i][2] }
    cx /= Math.max(1, n); cy /= Math.max(1, n); cz /= Math.max(1, n)
    let rmax = 1
    for (let i = 0; i < n; i += 1) {
      rmax = Math.max(rmax, Math.hypot(xyz[i][0] - cx, xyz[i][1] - cy, xyz[i][2] - cz))
    }
    st.center = new THREE.Vector3(cx, cy, cz)
    st.offset = [cx, cy, cz]
    st.home = { theta: 0.6, phi: 1.1, radius: rmax * 2.6, target: st.center.clone() }
    st.sph = { theta: st.home.theta, phi: st.home.phi, radius: st.home.radius }
    st.target = st.home.target.clone()
    st.posOf = new Map()
    for (let i = 0; i < n; i += 1) st.posOf.set(data.index[i], i)
    // 屏幕空间最小半径：大体系按真实共价半径画只有不到一个像素，看起来像"什么都没有"。
    // 球和棍各有一个下限；缩放时会重算（见 recalcScales）。
    recalcScales(st)
    // 缩放上下限按体系自身尺寸定（单原子占满 3/4 / 盒子占 1/3）
    computeZoomLimits(st, data)

    // 按元素分组 → 每种元素一个 InstancedMesh（draw call 少、半径可不同）
    const byEl = new Map()
    for (let i = 0; i < n; i += 1) {
      const el = data.elements[i] || 'X'
      if (!byEl.has(el)) byEl.set(el, [])
      byEl.get(el).push(i)
    }
    const baseGeo = new THREE.SphereGeometry(1, 12, 10)
    const baseMat = new THREE.MeshPhongMaterial({ shininess: 26, specular: 0x262626 })
    const m4 = new THREE.Matrix4()
    for (const [el, list] of byEl) {
      const mesh = new THREE.InstancedMesh(baseGeo.clone(), baseMat.clone(), list.length)
      mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage)
      const col = new THREE.Color(colorOf(el))
      list.forEach((i, k) => {
        const rad = ballRad(st, data, i)
        m4.makeScale(rad, rad, rad)
        m4.setPosition(xyz[i][0] - cx, xyz[i][1] - cy, xyz[i][2] - cz)
        mesh.setMatrixAt(k, m4)
        mesh.setColorAt(k, col)
      })
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
      mesh.userData = { element: el, displayIndex: data.index, slots: list }
      mesh.frustumCulled = false
      scene.add(mesh)
      st.meshes.push(mesh)
    }
    // 基础几何/材质只用于 clone，本身没进场景 —— 现在释放，否则每换一次漏一份
    baseGeo.dispose()
    baseMat.dispose()

    buildBonds(st, data)
    buildRings(st, data)
    buildBox(st, data)
    // 先用"没有选择"的状态把棍/盒的矩阵写一遍：buildGeometry 单独被调用时
    // （自检、或 React 两个 effect 之间）也必须是一个画得出来的有效状态。
    applyBondState(st, data, null, false)
    applyBoxState(st, data)
}

/* ------------------------------------------------------- 模拟盒（白色细柱） */
/**
 * 由 `[a, b, c, alpha, beta, gamma]` 还原三条棱向量（三斜的标准取向：
 * a 沿 x，b 在 xy 面内，c 由三个夹角定出）。
 */
export function cellVectors(box) {
  if (!Array.isArray(box) || box.length < 3) return null
  const [a, b, c] = box.slice(0, 3).map(Number)
  if (!(a > 0 && b > 0 && c > 0)) return null
  const deg = (v, dflt) => {
    const x = v === null || v === undefined ? NaN : Number(v)
    return Number.isFinite(x) && x > 0 && x < 180 ? x : dflt
  }
  const al = deg(box[3], 90) * Math.PI / 180
  const be = deg(box[4], 90) * Math.PI / 180
  const ga = deg(box[5], 90) * Math.PI / 180
  const ca = Math.cos(al); const cb = Math.cos(be); const cg = Math.cos(ga)
  const sg = Math.sin(ga)
  const A = [a, 0, 0]
  const B = [b * cg, b * sg, 0]
  // c 的 z 分量：负值会被夹到 0（数值上极端情况），避免 NaN 让整帧消失
  const cz = Math.sqrt(Math.max(0, c * c - (c * cb) ** 2
    - (c * (ca - cb * cg) / (sg || 1)) ** 2))
  const C = [c * cb, c * (ca - cb * cg) / (sg || 1), cz]
  return [A, B, C]
}

function buildBox(st, data) {
  const vecs = cellVectors(data.box)
  st.boxInfo = null
  if (!vecs) return
  const [A, B, C] = vecs
  // 盒子里放一堆"原始坐标围成的盒子"，但坐标可能没按周期折回（MS 的 .xsd 常见），
  // 于是盒子会离原子很远。这里画**离原子质心最近的那一个镜像盒子**，
  // 让盒子套在看到的分子外面 —— 这是纯显示选择，不改任何坐标。
  const g = st.offset || [0, 0, 0]
  const det = A[0] * (B[1] * C[2] - B[2] * C[1])
    - A[1] * (B[0] * C[2] - B[2] * C[0])
    + A[2] * (B[0] * C[1] - B[1] * C[0])
  const shift = [0, 0, 0]
  if (Math.abs(det) > 1e-9) {
    // f = M⁻¹·g（M 的三列是 A/B/C），行向量分别是 B×C、C×A、A×B
    const cross = (u, v) => [u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    const f = [dot(cross(B, C), g) / det, dot(cross(C, A), g) / det,
      dot(cross(A, B), g) / det]
    const n = f.map((x) => (Number.isFinite(x) ? Math.floor(x) : 0))
    for (let k = 0; k < 3; k += 1) {
      shift[k] = n[0] * A[k] + n[1] * B[k] + n[2] * C[k]
    }
  }
  // 8 个角点（已减去显示偏移）
  const corner = (i, j, k) => [
    shift[0] + i * A[0] + j * B[0] + k * C[0] - g[0],
    shift[1] + i * A[1] + j * B[1] + k * C[1] - g[1],
    shift[2] + i * A[2] + j * B[2] + k * C[2] - g[2],
  ]
  const P = (i, j, k) => corner(i, j, k)
  const edges = [
    [P(0, 0, 0), P(0, 0, 1)], [P(0, 1, 0), P(0, 1, 1)],
    [P(1, 0, 0), P(1, 0, 1)], [P(1, 1, 0), P(1, 1, 1)],   // 沿 c
    [P(0, 0, 0), P(0, 1, 0)], [P(0, 0, 1), P(0, 1, 1)],
    [P(1, 0, 0), P(1, 1, 0)], [P(1, 0, 1), P(1, 1, 1)],   // 沿 b
    [P(0, 0, 0), P(1, 0, 0)], [P(0, 1, 0), P(1, 1, 0)],
    [P(0, 0, 1), P(1, 0, 1)], [P(0, 1, 1), P(1, 1, 1)],   // 沿 a
  ]
  const nE = edges.length
  const center = new Float32Array(nE * 3)
  const quat = new Float32Array(nE * 4)
  const len = new Float32Array(nE)
  const UP = new THREE.Vector3(0, 1, 0)
  const dir = new THREE.Vector3()
  const q = new THREE.Quaternion()
  for (let e = 0; e < nE; e += 1) {
    const [p, r] = edges[e]
    dir.set(r[0] - p[0], r[1] - p[1], r[2] - p[2])
    const L = dir.length()
    if (L < 1e-9) { len[e] = 0; continue }
    dir.multiplyScalar(1 / L)
    q.setFromUnitVectors(UP, dir)
    center[3 * e] = (p[0] + r[0]) / 2
    center[3 * e + 1] = (p[1] + r[1]) / 2
    center[3 * e + 2] = (p[2] + r[2]) / 2
    quat[4 * e] = q.x; quat[4 * e + 1] = q.y
    quat[4 * e + 2] = q.z; quat[4 * e + 3] = q.w
    len[e] = L
  }
  const geo = new THREE.CylinderGeometry(1, 1, 1, 8, 1, false)
  // 白色 + 一点自发光：浅色背景上纯白细线几乎看不见，圆柱的明暗边才认得出是"盒子"
  const mat = new THREE.MeshPhongMaterial({
    color: new THREE.Color(BOX_COLOR), emissive: new THREE.Color(0x9aa4b2),
    shininess: 70, specular: 0xffffff,
  })
  const mesh = new THREE.InstancedMesh(geo, mat, nE)
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  mesh.userData = { n: nE, center, quat, len }
  mesh.frustumCulled = false
  mesh.renderOrder = -1          // 先画盒子，原子盖在上面
  st.scene.add(mesh)
  st.boxMesh = mesh
  st.boxInfo = {
    box: data.box.slice(0, 3).map((x) => Number(x)),
    angles: data.box.slice(3, 6).map((x) => Number(x)),
    shift,
  }
}

/** 盒子随缩放改粗细（不随选择变化 —— 盒子是参照物，一直显示） */
function applyBoxState(st) {
  const bm = st.boxMesh
  if (!bm || !bm.userData || !bm.userData.n) return
  const u = bm.userData
  const r = st.boxRadA || 0.4
  const p = new THREE.Vector3()
  const q = new THREE.Quaternion()
  const s = new THREE.Vector3()
  const m4 = new THREE.Matrix4()
  for (let e = 0; e < u.n; e += 1) {
    p.set(u.center[3 * e], u.center[3 * e + 1], u.center[3 * e + 2])
    q.set(u.quat[4 * e], u.quat[4 * e + 1], u.quat[4 * e + 2], u.quat[4 * e + 3])
    s.set(r, u.len[e] || 1e-6, r)
    m4.compose(p, q, s)
    bm.setMatrixAt(e, m4)
  }
  bm.instanceMatrix.needsUpdate = true
}

/* --------------------------------------------------------------- 棍（化学键） */
/* -------------------------------------------------- 芳香环（虚线圆，MS 风格） */
/**
 * 把后端给的**芳香六元环**（`data.rings`，原始原子编号）画成环心的虚线圈，
 * 与 Materials Studio 里苯环中间那个虚线圆一致。
 *
 * 全部环合成**一个** LineSegments（一个 draw call）：每个环在其所在平面内画 48 段
 * 圆周（平面由环上三个原子决定）；虚线由 LineDashedMaterial + computeLineDistances
 * 实现。只画**当前显示**的原子都齐的环（抽稀/截断时自动跳过）。
 */
function buildRings(st, data) {
  const rings = data.rings || []
  if (!rings.length) return
  const [ox, oy, oz] = st.offset
  const xyz = data.xyz
  const posOf = st.posOf
  const SEG = 48
  const verts = []
  const c = new THREE.Vector3()
  const uA = new THREE.Vector3()
  const vA = new THREE.Vector3()
  const nA = new THREE.Vector3()
  const tmp = new THREE.Vector3()
  for (const ring of rings) {
    if (ring.length < 3) continue
    const pts = []
    let ok = true
    for (const orig of ring) {
      const i = posOf.get(orig)
      if (i === undefined) { ok = false; break }
      pts.push(new THREE.Vector3(xyz[i][0] - ox, xyz[i][1] - oy, xyz[i][2] - oz))
    }
    if (!ok) continue
    c.set(0, 0, 0)
    pts.forEach((p) => c.add(p))
    c.multiplyScalar(1 / pts.length)
    uA.copy(pts[0]).sub(c)
    nA.copy(pts[1]).sub(pts[0]).cross(tmp.copy(pts[2]).sub(pts[0]))
    if (nA.lengthSq() < 1e-10 || uA.lengthSq() < 1e-10) continue
    nA.normalize()
    uA.addScaledVector(nA, -uA.dot(nA)).normalize()
    vA.crossVectors(nA, uA).normalize()
    const r = pts.reduce((acc, p) => acc + p.distanceTo(c), 0) / pts.length
    let prev = null
    for (let k = 0; k <= SEG; k += 1) {
      const th = (k / SEG) * Math.PI * 2
      const p = c.clone()
        .addScaledVector(uA, Math.cos(th) * r)
        .addScaledVector(vA, Math.sin(th) * r)
      if (prev) verts.push(prev.x, prev.y, prev.z, p.x, p.y, p.z)
      prev = p
    }
  }
  if (!verts.length) return
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
  const mat = new THREE.LineDashedMaterial({
    color: 0x1f2937, dashSize: 0.22, gapSize: 0.16,
    transparent: true, opacity: 0.9, depthWrite: false,
  })
  const seg = new THREE.LineSegments(geo, mat)
  seg.computeLineDistances()
  seg.renderOrder = 2
  seg.visible = st.showRings !== false
  seg.frustumCulled = false
  st.ringGroup = seg
  st.scene.add(seg)
}

/**
 * 把 `data.bonds` 画成圆柱。后端给的是**显示位置**成对数组（`[a0,b0,a1,b1,…]`），
 * 直接用即可 —— 前端不做任何编号换算（上一版就因为把原始编号当显示下标用而白屏）。
 *
 * 每根键拆成**两根半棍**（各取一端原子的颜色），这正是 Material Studio 的样子；
 * 一个 InstancedMesh 装下全部半棍，只多一个 draw call。
 */
function buildBonds(st, data) {
  const flat = data.bonds || []
  const n = data.n_displayed
  const xyz = data.xyz
  const [ox, oy, oz] = st.offset
  const pairs = []
  for (let b = 0; b + 1 < flat.length; b += 2) {
    const ia = flat[b] | 0
    const ib = flat[b + 1] | 0
    if (!(ia >= 0 && ia < n) || !(ib >= 0 && ib < n)) continue
    const L = Math.hypot(xyz[ib][0] - xyz[ia][0], xyz[ib][1] - xyz[ia][1],
      xyz[ib][2] - xyz[ia][2])
    if (L < 1e-4) continue          // 重合原子（同一位置）画不出方向
    pairs.push(ia, ib, L)
  }
  const nB = pairs.length / 3
  if (!nB) { st.nSticks = 0; return }
  const nHalf = nB * 2

  const geo = new THREE.CylinderGeometry(1, 1, 1, 6, 1, false)   // 单位半径/高度，沿 +Y
  const mat = new THREE.MeshPhongMaterial({ shininess: 18, specular: 0x1f1f1f })
  const mesh = new THREE.InstancedMesh(geo, mat, nHalf)
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

  const center = new Float32Array(nHalf * 3)
  const quat = new Float32Array(nHalf * 4)
  const half = new Float32Array(nHalf)        // 半根棍的长度
  const atoms = new Int32Array(nHalf)         // 该半根棍属于哪个**显示位置**
  const baseCol = new Float32Array(nHalf * 3) // 基础色（已转线性空间，与球一致）

  const UP = new THREE.Vector3(0, 1, 0)
  const vA = new THREE.Vector3(); const vB = new THREE.Vector3()
  const dir = new THREE.Vector3(); const mid = new THREE.Vector3()
  const q = new THREE.Quaternion()
  const c = new THREE.Color()
  for (let b = 0; b < nB; b += 1) {
    const ia = pairs[3 * b]; const ib = pairs[3 * b + 1]; const L = pairs[3 * b + 2]
    vA.set(xyz[ia][0] - ox, xyz[ia][1] - oy, xyz[ia][2] - oz)
    vB.set(xyz[ib][0] - ox, xyz[ib][1] - oy, xyz[ib][2] - oz)
    dir.copy(vB).sub(vA).normalize()
    q.setFromUnitVectors(UP, dir)
    mid.copy(vA).add(vB).multiplyScalar(0.5)
    for (let s = 0; s < 2; s += 1) {
      const dp = s === 0 ? ia : ib
      const k = 2 * b + s
      // 半棍中点：一端原子与键中点的中点
      const src = s === 0 ? vA : vB
      center[3 * k] = (src.x + mid.x) / 2
      center[3 * k + 1] = (src.y + mid.y) / 2
      center[3 * k + 2] = (src.z + mid.z) / 2
      quat[4 * k] = q.x; quat[4 * k + 1] = q.y
      quat[4 * k + 2] = q.z; quat[4 * k + 3] = q.w
      half[k] = L / 2
      atoms[k] = dp
      c.set(colorOf(data.elements[dp]))
      baseCol[3 * k] = c.r; baseCol[3 * k + 1] = c.g; baseCol[3 * k + 2] = c.b
    }
  }
  mesh.userData = { n: nHalf, center, quat, half, atoms, baseCol }
  mesh.frustumCulled = false
  st.scene.add(mesh)
  st.bondMesh = mesh
  st.nSticks = nHalf
}

export function applyStateToScene(st, data, selected, onlySelected = false) {
    const hi = new THREE.Color(HIGHLIGHT)
    const m4 = new THREE.Matrix4()
    const [ox, oy, oz] = st.offset || [0, 0, 0]
    for (const mesh of st.meshes) {
      const base = new THREE.Color(colorOf(mesh.userData.element))
      mesh.userData.slots.forEach((i, k) => {
        const atom = data.index[i]
        const isSel = !!(selected && selected.has(atom))
        mesh.setColorAt(k, isSel ? hi : base)
        // 「只显示已选」：没选中的缩到不可见（**编号不变**，只是不画）
        const s = onlySelected && !isSel ? 1e-6 : ballRad(st, data, i)
        m4.makeScale(s, s, s)
        m4.setPosition(data.xyz[i][0] - ox, data.xyz[i][1] - oy, data.xyz[i][2] - oz)
        mesh.setMatrixAt(k, m4)
      })
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    }
    applyBondState(st, data, selected, onlySelected)
    applyBoxState(st)
    st.kick()
}

/** 棍跟着球一起变：被选中的半棍高亮，`只显示已选` 时缩到不可见 */
function applyBondState(st, data, selected, onlySelected) {
    const bm = st.bondMesh
    if (!bm || !bm.userData || !bm.userData.n) return
    const u = bm.userData
    const hi = new THREE.Color(HIGHLIGHT)
    const col = new THREE.Color()
    const p = new THREE.Vector3()
    const q = new THREE.Quaternion()
    const s = new THREE.Vector3()
    const m4 = new THREE.Matrix4()
    for (let k = 0; k < u.n; k += 1) {
      const dp = u.atoms[k]
      const isSel = !!(selected && selected.has(data.index[dp]))
      const f = onlySelected && !isSel ? 1e-6 : 1
      // 棍半径由**这一端**的球半径封顶：氢球小，棍就必须更细，
      // 否则氢会被画成"一根白棍"（用户实测的 bug）。
      const r0 = stickRad(st, data, dp)
      p.set(u.center[3 * k], u.center[3 * k + 1], u.center[3 * k + 2])
      q.set(u.quat[4 * k], u.quat[4 * k + 1], u.quat[4 * k + 2], u.quat[4 * k + 3])
      s.set(r0 * f, u.half[k] * f, r0 * f)
      m4.compose(p, q, s)
      bm.setMatrixAt(k, m4)
      if (isSel) col.copy(hi)
      else col.setRGB(u.baseCol[3 * k], u.baseCol[3 * k + 1], u.baseCol[3 * k + 2])
      bm.setColorAt(k, col)
    }
    bm.instanceMatrix.needsUpdate = true
    if (bm.instanceColor) bm.instanceColor.needsUpdate = true
}

/* ==================================================================== 画布 */
function Viewer({ data, selected, onlySelected, onPick, onHover,
                  binding, onError }) {
  const boxRef = useRef(null)
  const stRef = useRef(null)
  // 回调放 ref：切换"选择层级"会换掉 onPick 引用，放进依赖会让整个 WebGL 场景重建
  const pickRef = useRef(onPick)
  const hoverRef = useRef(onHover)
  const errRef = useRef(onError)
  pickRef.current = onPick
  hoverRef.current = onHover
  errRef.current = onError

  useEffect(() => {
    const box = boxRef.current
    if (!box) return undefined
    let renderer
    try {
      // ⚠️ preserveDrawingBuffer 必须开：关着的时候，浏览器在合成后可以把
      //    WebGL 的绘图缓冲清掉，而此时若 React 因为悬停换了覆盖层（原子信息框）
      //    重排 DOM，就会有一帧拿到"已被清空"的画布 —— 表现就是悬停切换原子时
      //    **短暂闪一下白屏**（用户实测）。
      renderer = new THREE.WebGLRenderer({
        antialias: true, alpha: false, preserveDrawingBuffer: true,
      })
    } catch (e) {
      box.innerHTML = '<div class="placeholder"><p>这个浏览器/环境不能创建 WebGL '
        + '上下文，三维视图不可用（分析功能不受影响）。</p></div>'
      return undefined
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
    renderer.setClearColor(0xf7f9fc)
    box.appendChild(renderer.domElement)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 40000)
    // near 会在每次渲染时按相机距离自适应（见 render()），far 留够最大缩放
    scene.add(new THREE.AmbientLight(0xffffff, 0.72))
    const l1 = new THREE.DirectionalLight(0xffffff, 0.75)
    l1.position.set(1, 1.2, 1.6)
    scene.add(l1)
    const l2 = new THREE.DirectionalLight(0xffffff, 0.3)
    l2.position.set(-1, -0.8, -1.2)
    scene.add(l2)

    const st = {
      renderer, scene, camera,
      meshes: [], bondMesh: null, boxMesh: null, bondLines: null, labelGroup: null,
      sph: { theta: 0.6, phi: 1.1, radius: 100 },
      target: new THREE.Vector3(),
      home: null,
      raycaster: new THREE.Raycaster(),
      hoverIdx: null,
      posOf: new Map(),
      center: new THREE.Vector3(),
    }
    st.el = box                 // 数据/状态 effect 里要量尺寸，只能通过 st 拿
    // 「双击聚焦」：把注视点挪到该原子上，并把相机拉到 FOCUS_RADIUS_A。
    // 这是"钻进分子里"的唯一入口 —— 绕质心旋转 + 滚轮只能靠近质心，
    // 想看清多分子体系里的某一个分子，必须先把它变成新的注视点。
    st.focusOn = (atomIndex) => {
      const data = st.lastData
      if (!data) return
      const dp = st.posOf ? st.posOf.get(atomIndex) : undefined
      if (dp === undefined || dp < 0) return
      const [ox, oy, oz] = st.offset || [0, 0, 0]
      const p = data.xyz[dp]
      st.target = new THREE.Vector3(p[0] - ox, p[1] - oy, p[2] - oz)
      const lo = st.radiusMin || MIN_RADIUS_A
      st.sph.radius = Math.max(lo, Math.min(st.sph.radius, FOCUS_RADIUS_A))
      st.refresh?.()
      kick()
    }
    stRef.current = st

    const render = () => {
      const { theta, phi, radius } = st.sph
      const t = st.target
      camera.position.set(
        t.x + radius * Math.sin(phi) * Math.cos(theta),
        t.y + radius * Math.cos(phi),
        t.z + radius * Math.sin(phi) * Math.sin(theta))
      camera.lookAt(t)
      // 近裁剪面跟着相机距离走：固定 0.1 时拉近到原子尺度会被裁掉（看起来像
      // "缩放有上限"），固定太小又会让球和棍 z-fighting。按半径的比例取最稳：
      // 全景 196 Å → near 0.78；贴脸 0.5 Å → near 0.002，任何缩放级别都不裁。
      const near = Math.max(1e-3, radius * 0.004)
      if (Math.abs(camera.near - near) > near * 0.05) {
        camera.near = near
        camera.updateProjectionMatrix()
      }
      renderer.render(scene, camera)
    }
    // 「需要时重画」：用 rAF 句柄去重。
    // ⚠️ 不能拿某个"已排队"布尔量去早退 —— 初值想错一次就永远画不出第一帧
    //    （这正是"看不到分子"的原因：kick() 第一次调用就 return 了）。
    let raf = 0
    const kick = () => {
      if (raf) return
      raf = requestAnimationFrame(() => {
        raf = 0
        render()
      })
    }
    st.kick = kick

    const resize = () => {
      // 容器还没排完版时 clientHeight 可能是 0 → drawing buffer 只有 1px 高，
      // 画面等于空白。用 640×420 兜底，ResizeObserver 随后会纠正。
      const w = box.clientWidth || 640
      const h = box.clientHeight || 420
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      st.refresh?.()          // 画布高度变了 → "每像素多少 Å" 也变了
      kick()
    }
    const ro = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(resize) : null
    if (ro) ro.observe(box)
    window.addEventListener('resize', resize)
    resize()

    // ---- 拾取：屏幕坐标 → 命中的**原始原子编号**
    const pickAt = (cx, cy) => {
      const rect = box.getBoundingClientRect()
      const ndc = new THREE.Vector2(
        ((cx - rect.left) / rect.width) * 2 - 1,
        -((cy - rect.top) / rect.height) * 2 + 1)
      st.raycaster.setFromCamera(ndc, camera)
      for (const m of st.meshes) {
        if (!m.visible) continue
        const hits = st.raycaster.intersectObject(m, false)
        if (hits.length) return m.userData.displayIndex[hits[0].instanceId]
      }
      return null
    }

    // dragging: 0 无 / 1 旋转 / 2 平移；moved 记录累计位移，用来区分"单击"与"拖动"
    let dragging = 0
    let moved = 0
    let last = { x: 0, y: 0 }
    const onDown = (ev) => {
      box.setPointerCapture?.(ev.pointerId)
      dragging = (ev.button === 2 || ev.shiftKey) ? 2 : 1
      moved = 0
      last = { x: ev.clientX, y: ev.clientY }
    }
    const onMove = (ev) => {
      if (!dragging) {
        st.pendingHover = { x: ev.clientX, y: ev.clientY }
        return
      }
      const dx = ev.clientX - last.x
      const dy = ev.clientY - last.y
      moved += Math.abs(dx) + Math.abs(dy)
      last = { x: ev.clientX, y: ev.clientY }
      if (dragging === 1) {
        // 灵敏度用常量（用户实测：原来一转就飞了）
        st.sph.theta -= dx * ROT_PER_PX
        st.sph.phi = Math.min(Math.PI - 0.02, Math.max(0.02, st.sph.phi - dy * ROT_PER_PX))
      } else {
        const scale = st.sph.radius * PAN_PER_PX
        const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0)
        const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1)
        st.target.addScaledVector(right, -dx * scale)
        st.target.addScaledVector(up, dy * scale)
      }
      kick()
    }
    const onUp = (ev) => {
      const wasDragging = dragging
      const dist = moved
      dragging = 0
      moved = 0
      box.releasePointerCapture?.(ev.pointerId)
      // ⚠️ 只有"真的位移了"才算拖动：原来按 pointerdown 的标志位判断，
      //    于是每次单击都被当作拖动直接 return，原子永远选不中（用户实测）。
      if (wasDragging && dist > 4) return
      const idx = pickAt(ev.clientX, ev.clientY)
      if (idx === null || idx === undefined) return
      // 双击某个原子 = **把镜头移到它身上并放大到分子内部**（用户要求：
      // "可以穿透不同的分子、一直放大到多分子里的单个分子" —— 光靠绕质心转
      // 是钻不进去的，得有一个"聚焦到这里"的手势）
      if (ev.detail >= 2) {
        st.focusOn?.(idx)
        return
      }
      pickRef.current?.(idx, ev.ctrlKey || ev.metaKey)
    }
    const onWheel = (ev) => {
      ev.preventDefault()
      const before = st.sph.radius
      // 上下限都按体系自身算（见 computeZoomLimits）：
      //   最近 = 最大那颗原子直径 ≈ 视野 3/4；最远 = 模拟盒 ≈ 视野 1/3。
      const lo = st.radiusMin || MIN_RADIUS_A
      const hi = st.radiusMax || MAX_RADIUS_A
      st.sph.radius = Math.min(hi, Math.max(lo,
        st.sph.radius * Math.exp(ev.deltaY * ZOOM_PER_DELTA)))
      // 缩放会改变"每像素多少 Å"→ 球/棍/盒棱的屏幕尺度跟着变（球棍模型靠这一步才成立：
      // 拉远靠下限保住可见，靠近让真实比例接管）。变化超过 8% 才重算矩阵，别每格都刷。
      if (Math.abs(st.sph.radius / before - 1) > 0.08) st.refresh?.()
      kick()
    }
    const onCtx = (ev) => ev.preventDefault()

    box.addEventListener('pointerdown', onDown)
    box.addEventListener('pointermove', onMove)
    box.addEventListener('pointerup', onUp)
    box.addEventListener('wheel', onWheel, { passive: false })
    box.addEventListener('contextmenu', onCtx)

    // 悬停拾取：给个 ~40ms 的节流。
    // 鼠标一动就拾取的话，每帧都要对几万个实例做射线求交，主线程被占满 ——
    // 表现出来就是"悬停切换原子时闪一下白屏"。
    let hoverRaf = 0
    let hoverLast = 0
    const hoverLoop = (now) => {
      if (st.pendingHover && now - hoverLast >= 40) {
        const p = st.pendingHover
        st.pendingHover = null
        hoverLast = now
        const idx = pickAt(p.x, p.y)
        if (idx !== st.hoverIdx) {
          st.hoverIdx = idx
          hoverRef.current?.(idx)
        }
      }
      hoverRaf = requestAnimationFrame(hoverLoop)
    }
    hoverRaf = requestAnimationFrame(hoverLoop)

    return () => {
      cancelAnimationFrame(hoverRaf)
      box.removeEventListener('pointerdown', onDown)
      box.removeEventListener('pointermove', onMove)
      box.removeEventListener('pointerup', onUp)
      box.removeEventListener('wheel', onWheel)
      box.removeEventListener('contextmenu', onCtx)
      window.removeEventListener('resize', resize)
      if (ro) ro.disconnect()
      // 释放 GPU 资源（几何 / 材质 / 纹理）
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose()
        const mats = Array.isArray(o.material) ? o.material : [o.material]
        for (const mm of mats) {
          if (!mm) continue
          for (const k of ['map', 'alphaMap']) if (mm[k]) mm[k].dispose()
          mm.dispose()
        }
      })
      renderer.dispose()
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement)
      }
      stRef.current = null
    }
  }, [])

  // ------------------------------------------------------- 数据变化：重建几何
  useEffect(() => {
    const st = stRef.current
    if (!st || !data) return
    try {
      buildGeometry(st, data)
      st.lastData = data          // 「双击聚焦」要用它换算原子坐标
      st.kick()
    } catch (e) {
      // 这一层兜底很关键：effect 里抛错会让 React 卸掉整棵树（表现为白屏）。
      // 这里改成把原因显示在视图上方，页面其余部分照常可用。
      console.error('[mdta] 三维几何构建失败：', e)
      errRef.current?.(`三维视图构建失败：${e.message || e}`)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data])

  // ------------------------------------------ 选择 / 隐藏 / 键 / 标签：只改颜色
  useEffect(() => {
    const st = stRef.current
    if (!st || !data) return
    try {
      applyStateToScene(st, data, selected, !!onlySelected)
      // 缩放时要能重算比例再重画一遍，这里把"当前该画成什么样"记在 st 上
      st.refresh = () => {
        recalcScales(st)
        applyStateToScene(st, data, selected, !!onlySelected)
      }
    } catch (e) {
      console.error('[mdta] 三维状态更新失败：', e)
      errRef.current?.(`三维视图更新失败：${e.message || e}`)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, selected, onlySelected])

  // 暴露"重置视角"给左侧工具栏
  useEffect(() => {
    if (!binding) return
    binding.current = {
      resetView: () => {
        const st = stRef.current
        if (!st || !st.home) return
        st.sph = { theta: st.home.theta, phi: st.home.phi, radius: st.home.radius }
        st.target = st.home.target.clone()
        st.refresh?.()            // 相机距离变了 → 球/棍的屏幕下限也要重算
        st.kick()
      },
      // 覆盖层（原子信息框）变了就让画布同帧重画一次，避免出现"白画布 + 新 DOM"
      kick: () => stRef.current?.kick?.(),
    }
  }, [binding])

  return <div ref={boxRef} className="viewer-canvas" />
}
