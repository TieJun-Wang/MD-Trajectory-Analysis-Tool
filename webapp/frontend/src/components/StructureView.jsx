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
import {
  AROMATIC_RING, ATOM_MATERIAL, BOND_MATERIAL, DEFAULT_COLOR, HIGHLIGHT_COLOR, HIGHLIGHT_SCALE,
  LIGHTING, RENDER, BOX as BOX_CFG,
  bondRadius, displayRadius, elementColor, qualityFor,
} from '../renderConfig'

//: 元素颜色（配置集中在 renderConfig.js —— 这里只是兼容旧名的薄封装）
function colorOf(el) {
  return elementColor(el)
}
//: 被选中的原子/棍的高亮色 —— **亮黄**（用户要求"要明显"；洋红 #ff2d55 在浅底
//: 上不够醒目，深黄 #c9a227 也不够亮，这里直接拉到纯黄系，#ffe600），
//: 并且 `applyStateToScene` 会把选中的球再放大 15%，一眼就能看出来。
const HIGHLIGHT = HIGHLIGHT_COLOR

/* ---------------------------------------------------------- 球棍模型（参考 Material Studio）
 * 视觉参数（元素颜色 / 显示半径 / 键半径 / 材质）全部集中在 renderConfig.js，
 * 这里只做几何拼装。
 *   球半径 = displayRadius(元素) × 全局缩放（见 renderConfig.ATOM_SCALE）
 *   棍半径 = bondRadius(两端元素)，并封顶到"两端球半径较小者 × 0.6"，
 *            保证球一定露在棍外面（否则 H 会被画成"一根白棍"）。
 *
 * 屏幕空间下限是必要的：几万个原子按真实半径画只有不到 1 个像素，看上去"什么都没有"。
 */
const BALL_MIN_PX = 1.3            // 球半径的屏幕下限（像素）
const ROT_PER_PX = 0.004           // 旋转灵敏度（rad / 像素）：用户要求调低（原 0.008）
const PAN_PER_PX = 0.0009          // 平移灵敏度（视野半径 / 像素）：原 0.0016
const ZOOM_PER_DELTA = 0.0009      // 滚轮缩放灵敏度（rad 的指数）：0.0016 → 0.0009
                                   // （用户实测"缩放太灵敏、一动就飞"，放慢一档）
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
const BOX_COLOR = BOX_CFG.color
const BOX_PX = BOX_CFG.px            // 盒棱半径（屏幕像素，不随缩放变粗）

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
    // 球半径来自**显示半径表**（renderConfig.ELEMENT_RADII × ATOM_SCALE）
    const r = Math.max(0.03, Math.min(2.5, displayRadius(data.elements[i])))
    if (r > maxBall) maxBall = r
  }
  st.maxBallA = maxBall
  // 最近缩放：最大球直径 = 视野高 × BALL_FILL（1.8 → 已经能钻进原子内部）。
  // 注意：能不能"贴近箱内某颗原子"靠的是**注视点**（滚轮会朝光标下的原子漂），
  // 不是这个下限 —— 以前只沿"体系中心"缩，看起来像"被盒子拦住"。
  st.radiusMin = maxBall / (BALL_FILL * tanHalf)
  st.radiusMin = Math.min(st.radiusMin, homeR * 0.5)          // 别比初始视角还远
  st.radiusMin = Math.max(st.radiusMin, MIN_RADIUS_A)
  return st.radiusMin
}

/** 球半径（Å）：**显示半径表**（renderConfig.ELEMENT_RADII × ATOM_SCALE），
 *  再受"屏幕下限"保护 —— 屏幕下限只在拉远时兜住可见性，放大后让真实比例接管。
 *  只影响画面，不改坐标、不改分析数据。 */
function ballRad(st, data, i) {
  const base = displayRadius(data.elements[i])
  return Math.max(0.03, Math.min(2.5, Math.max(base, st.minRadA || 0)))
}

/** 棍半径（Å）：按**两端元素**取键半径，并封顶到两端球半径较小者的 0.6 —— 
 *  否则小原子（H）会被棍吞掉，画出来像"一根白棍"而不是一颗球。 */
function stickRad(st, data, i) {
  const base = bondRadius(data.elements[i], data.elements[i])
  return Math.min(base, 0.6 * ballRad(st, data, i))
}

/** 一根键的半径：按两端元素查表，并保证不会被两端的球吞掉 */
function bondRad(st, data, ia, ib) {
  const base = bondRadius(data.elements[ia], data.elements[ib])
  const cap = 0.6 * Math.min(ballRad(st, data, ia), ballRad(st, data, ib))
  return Math.min(base, cap)
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
                  pendingView={st.pendingView} sid={st.sid}
                  onPick={st.pick} onHover={st.setHover} binding={st.viewerRef}
                  onPickMonomer={st.pickMonomer} onPickChain={st.pickChain}
                  onClearSelection={st.resetAll}
                  onError={st.setErr} />
        )}
        {data && hover !== null && (
          <AtomTip data={data} atom={hover} />
        )}
        {data && (
          <div className="viewhint">
            球棍模型 · 左键拖动旋转 · 滚轮缩放 · 单击原子高亮 · Ctrl 单击加/减选 ·
            双击原子选中单体 · 右键原子选中整链（再右击 = 同类全部链）·
            Ctrl+双击 聚焦 · 点空白处清空选择
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
    // 相机：优先"恢复存档"（F5 后接回来的），其次**保留当前视角**（同会话重载），
    // 都没有才回默认视角。用户要求：刷新/重载后还是原来一样。
    const savedView = st.pendingView
      || (st.loadSavedView ? st.loadSavedView() : null)
    st.pendingView = null
    if (savedView && savedView.sph && Number.isFinite(savedView.sph.radius)) {
      st.sph = { theta: savedView.sph.theta, phi: savedView.sph.phi,
        radius: savedView.sph.radius }
      st.target = savedView.target
        ? new THREE.Vector3(savedView.target.x, savedView.target.y, savedView.target.z)
        : st.home.target.clone()
    } else if (st.keepCam && st.sph && Number.isFinite(st.sph.radius)) {
      if (!st.target) st.target = st.home.target.clone()
    } else {
      st.sph = { theta: st.home.theta, phi: st.home.phi, radius: st.home.radius }
      st.target = st.home.target.clone()
    }
    st.keepCam = true
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
    // 球：质量档位决定细分（standard = 24×24）。材质用 MeshPhysicalMaterial：
    // roughness 0.28 + clearcoat 0.25 → 圆润柔和的高光，不金属、不塑料。
    // ⚠️ 实例颜色是乘在材质颜色上的，材质底色必须是白（否则会被压暗）。
    const Q = qualityFor(n)
    const baseGeo = new THREE.SphereGeometry(1, Q.sphereW, Q.sphereH)
    const baseMat = new THREE.MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: ATOM_MATERIAL.roughness,
      metalness: ATOM_MATERIAL.metalness,
      clearcoat: ATOM_MATERIAL.clearcoat,
      clearcoatRoughness: ATOM_MATERIAL.clearcoatRoughness,
      ior: ATOM_MATERIAL.ior,
      reflectivity: ATOM_MATERIAL.reflectivity,
      envMapIntensity: ATOM_MATERIAL.envMapIntensity,
    })
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
/* --------------------------------------- 芳香环（苯环内部半透明圆环） */
/**
 * 把后端给的**芳香环**（`data.rings`，原始原子编号）画成环心的一根圆环。
 *
 * 数据来源：后端 `/structure` 的 `rings` 已经**只含芳香环** ——
 * `mdta.rings.analyze_rings` 判 "6 元碳环 + 6 根环键均 ≤1.50 Å"（is_benzene），
 * 前端不按六元环形状自己猜芳香性。
 *
 * 几何：THREE.TorusGeometry（半径 = 苯环内切半径 × 0.78，管道 0.045 Å，
 * 沿圆周 96 段）。法向量用**协方差 + Jacobi 特征分解**取最小特征值对应的
 * 特征向量（PCA 平面拟合；环接近共线时退回 Newell 面积法向），
 * 再用四元数把 +Z 转到该法向 —— 不假设苯环在 XY 平面。
 *
 * 渲染：半透明 + depthWrite:false + renderOrder 抬后，圆环不遮挡原子与化学键；
 * 多个芳香环共用一个 InstancedMesh（一次 draw call）。
 */
function _ringNormal(pts) {
  // 1) 平面拟合：Newell 面积法向 —— 对任意朝向/倾斜的环都成立（旋转不变，
  //    对近似共面的环也稳）。这就是"最小二乘平面"的法向，等同于 PCA 在平面
  //    情况下的结果。
  const nrm = new THREE.Vector3()
  const n = pts.length
  for (let i = 0; i < n; i += 1) {
    const a = pts[i]
    const b = pts[(i + 1) % n]
    nrm.x += (a.y - b.y) * (a.z + b.z)
    nrm.y += (a.z - b.z) * (a.x + b.x)
    nrm.z += (a.x - b.x) * (a.y + b.y)
  }
  if (nrm.lengthSq() > 1e-16) return nrm.normalize()

  // 2) 退化（环几乎共线）→ PCA 兜底：3×3 协方差最小特征向量（Jacobi 特征分解）
  const c = new THREE.Vector3()
  for (const p of pts) c.add(p)
  c.multiplyScalar(1 / n)
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
  for (const p of pts) {
    const d = [p.x - c.x, p.y - c.y, p.z - c.z]
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) A[i][j] += d[i] * d[j]
    }
  }
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) A[i][j] /= Math.max(n, 1)
  }
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
  for (let sweep = 0; sweep < 32; sweep += 1) {
    let off = 0
    for (let i = 0; i < 3; i += 1) {
      for (let j = i + 1; j < 3; j += 1) off += A[i][j] * A[i][j]
    }
    if (off < 1e-24) break
    for (let p1 = 0; p1 < 3; p1 += 1) {
      for (let q1 = p1 + 1; q1 < 3; q1 += 1) {
        if (Math.abs(A[p1][q1]) < 1e-24) continue
        const theta = (A[q1][q1] - A[p1][p1]) / (2 * A[p1][q1])
        const tt = Math.sign(theta || 1)
          / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const cth = 1 / Math.sqrt(tt * tt + 1)
        const sth = tt * cth
        for (let k = 0; k < 3; k += 1) {
          const akp = A[k][p1]; const akq = A[k][q1]
          A[k][p1] = cth * akp - sth * akq
          A[k][q1] = sth * akp + cth * akq
        }
        for (let k = 0; k < 3; k += 1) {
          const apk = A[p1][k]; const aqk = A[q1][k]
          A[p1][k] = cth * apk - sth * aqk
          A[q1][k] = sth * apk + cth * aqk
        }
        for (let k = 0; k < 3; k += 1) {
          const vkp = V[k][p1]; const vkq = V[k][q1]
          V[k][p1] = cth * vkp - sth * vkq
          V[k][q1] = sth * vkp + cth * vkq
        }
      }
    }
  }
  const dg = [A[0][0], A[1][1], A[2][2]]
  const mi = dg.indexOf(Math.min(...dg))
  const out = new THREE.Vector3(V[0][mi], V[1][mi], V[2][mi])
  if (!Number.isFinite(out.x) || out.lengthSq() < 1e-16) return null
  return out.normalize()
}

function buildRings(st, data) {
  const rings = data.rings || []
  if (!rings.length) return
  const [ox, oy, oz] = st.offset
  const xyz = data.xyz
  const posOf = st.posOf
  const CFG = AROMATIC_RING
  const UP = new THREE.Vector3(0, 0, 1)      // TorusGeometry 默认在 XY 平面，法向 +Z
  const items = []                          // { center, quat, radius }
  const c = new THREE.Vector3()
  const q = new THREE.Quaternion()
  for (const ring of rings) {
    if (!ring || ring.length < 3) continue
    const pts = []
    let ok = true
    for (const orig of ring) {
      const i = posOf.get(orig)
      if (i === undefined) { ok = false; break }
      pts.push(new THREE.Vector3(xyz[i][0] - ox, xyz[i][1] - oy, xyz[i][2] - oz))
    }
    if (!ok) continue
    c.set(0, 0, 0)
    for (const p of pts) c.add(p)
    c.multiplyScalar(1 / pts.length)
    const nrm = _ringNormal(pts)
    if (!nrm) continue
    // 正六边形边长取平均 → 内切半径 = 边长 × √3/2；圆环半径 = 内切半径 × 0.78
    let side = 0
    for (let i = 0; i < pts.length; i += 1) {
      side += pts[i].distanceTo(pts[(i + 1) % pts.length])
    }
    side /= pts.length
    const inradius = side * Math.sqrt(3) / 2
    const radius = Math.max(0.08, inradius * CFG.radiusFrac)
    q.setFromUnitVectors(UP, nrm)
    items.push({ center: c.clone(), quat: q.clone(), radius })
  }
  if (!items.length) return

  // 单位圆环：环半径 1、管道 0.045（按"典型半径"折算成比例，实例再整体缩放）。
  // 这样不同半径的环共用一份几何，只用一个 draw call。
  const typical = items.reduce((a, it) => a + it.radius, 0) / items.length
  const tubeRatio = CFG.tube / Math.max(typical, 1e-6)
  const geo = new THREE.TorusGeometry(1, tubeRatio,
    CFG.radialSegments, CFG.tubularSegments)
  const mat = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(CFG.color),
    transparent: !!CFG.transparent,
    opacity: CFG.opacity,
    roughness: CFG.roughness,
    metalness: CFG.metalness,
    emissive: new THREE.Color(CFG.emissive),
    emissiveIntensity: CFG.emissiveIntensity,
    depthWrite: !!CFG.depthWrite,
    side: THREE.DoubleSide,
  })
  const mesh = new THREE.InstancedMesh(geo, mat, items.length)
  const m4 = new THREE.Matrix4()
  const scl = new THREE.Vector3()
  items.forEach((it, k) => {
    scl.set(it.radius, it.radius, it.radius)
    m4.compose(it.center, it.quat, scl)
    mesh.setMatrixAt(k, m4)
  })
  mesh.instanceMatrix.needsUpdate = true
  mesh.frustumCulled = false
  // 半透明对象在透明队列里画：放在原子/化学键之后（renderOrder 抬后），
  // 且 depthWrite=false 不写深度 —— 不会遮挡原子球和键，也不会和它们 z-fighting。
  mesh.renderOrder = 3
  st.ringGroup = mesh
  st.scene.add(mesh)
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
  //: 键级（可选）：后端给了 `bond_orders` 就按 1/2/3 画成平行多根圆柱，
  //  没给就一律按单键。双键/三键每根子圆柱更细、按垂直于键轴的平面错开。
  const orders = data.bond_orders || null
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
    let order = 1
    if (orders) {
      const o = orders[b >> 1]
      order = Number.isFinite(o) ? Math.max(1, Math.min(3, Math.round(o))) : 1
    }
    pairs.push(ia, ib, L, order)
  }
  const nB = pairs.length / 4
  if (!nB) { st.nSticks = 0; return }
  // 每根键拆成 order 根子圆柱、每根再拆成两端半棍 → 2×order 个实例
  let nHalf = 0
  for (let b = 0; b < nB; b += 1) nHalf += 2 * pairs[4 * b + 3]

  const Q = qualityFor(data.n_displayed)
  // 单位圆柱（半径 1 / 高 1，沿 +Y）：键粗细由实例矩阵的 x/z 缩放给出，
  // 因此**同一个 geometry 可以画出任意粗细的键**（不同元素对不同粗细）。
  const geo = new THREE.CylinderGeometry(1, 1, 1, Q.bondRadial, 1, false)
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: BOND_MATERIAL.roughness,
    metalness: BOND_MATERIAL.metalness,
    clearcoat: BOND_MATERIAL.clearcoat,
    clearcoatRoughness: BOND_MATERIAL.clearcoatRoughness,
    reflectivity: BOND_MATERIAL.reflectivity,
    envMapIntensity: BOND_MATERIAL.envMapIntensity,
  })
  const mesh = new THREE.InstancedMesh(geo, mat, nHalf)
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)

  const center = new Float32Array(nHalf * 3)
  const quat = new Float32Array(nHalf * 4)
  const half = new Float32Array(nHalf)        // 半根棍的长度
  const radii = new Float32Array(nHalf)       // 该半根棍的半径（按两端元素对）
  const atoms = new Int32Array(nHalf)         // 该半根棍属于哪个**显示位置**
  const baseCol = new Float32Array(nHalf * 3) // 基础色（已转线性空间，与球一致)

  const UP = new THREE.Vector3(0, 1, 0)
  const vA = new THREE.Vector3(); const vB = new THREE.Vector3()
  const dir = new THREE.Vector3(); const mid = new THREE.Vector3()
  const perp = new THREE.Vector3(); const tmpV = new THREE.Vector3()
  const q = new THREE.Quaternion()
  const c = new THREE.Color()
  let k = 0                                   // 实例序号（双/三键会占多个）
  for (let b = 0; b < nB; b += 1) {
    const ia = pairs[4 * b]; const ib = pairs[4 * b + 1]; const L = pairs[4 * b + 2]
    const order = pairs[4 * b + 3]
    vA.set(xyz[ia][0] - ox, xyz[ia][1] - oy, xyz[ia][2] - oz)
    vB.set(xyz[ib][0] - ox, xyz[ib][1] - oy, xyz[ib][2] - oz)
    dir.copy(vB).sub(vA).normalize()
    q.setFromUnitVectors(UP, dir)
    mid.copy(vA).add(vB).multiplyScalar(0.5)
    const rb = bondRad(st, data, ia, ib)
    // 单键 1 根；双键 2 根；三键 3 根。子圆柱半径按根数收细，
    // 错开方向取"垂直于键轴"的一个向量（键轴与 y 平行时换 x 兜底）。
    const subR = order === 1 ? rb : rb / 1.7
    const gap = order === 1 ? 0 : rb * 1.05
    perp.crossVectors(dir, Math.abs(dir.y) > 0.9 ? tmpV.set(1, 0, 0) : UP).normalize()
    const shifts = order === 1 ? [0]
      : order === 2 ? [-gap, gap] : [-gap, 0, gap]
    for (const shift of shifts) {
      const off = tmpV.copy(perp).multiplyScalar(shift)
      for (let s = 0; s < 2; s += 1) {
        const dp = s === 0 ? ia : ib
        // 半棍中点：一端原子与键中点的中点（再加上多键的错位）
        const src = s === 0 ? vA : vB
        center[3 * k] = (src.x + mid.x) / 2 + off.x
        center[3 * k + 1] = (src.y + mid.y) / 2 + off.y
        center[3 * k + 2] = (src.z + mid.z) / 2 + off.z
        quat[4 * k] = q.x; quat[4 * k + 1] = q.y
        quat[4 * k + 2] = q.z; quat[4 * k + 3] = q.w
        half[k] = L / 2
        radii[k] = subR
        atoms[k] = dp
        c.set(colorOf(data.elements[dp]))
        baseCol[3 * k] = c.r; baseCol[3 * k + 1] = c.g; baseCol[3 * k + 2] = c.b
        k += 1
      }
    }
  }
  mesh.userData = { n: nHalf, center, quat, half, radii, atoms, baseCol }
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
        // 「只显示已选」：没选中的缩到不可见（**编号不变**，只是不画）；
        // 选中的再放大 15% —— 亮黄色 + 略大的球，远看也能一眼认出选了什么
        // （用户实测：只换颜色不够明显）。
        const r = ballRad(st, data, i)
        const s = onlySelected && !isSel ? 1e-6 : r * (isSel ? HIGHLIGHT_SCALE : 1)
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
      // 棍半径：建几何时已按**两端元素对**算好（u.radii），这里直接用；
      // 老数据没有 radii 时再退回按这一端球半径封顶。
      const r0 = u.radii ? u.radii[k] : stickRad(st, data, dp)
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
function Viewer({ data, selected, onlySelected, pendingView = null,
                  sid = '', onPick, onHover, binding, onError,
                  onPickMonomer = null, onPickChain = null,
                  onClearSelection = null }) {
  const boxRef = useRef(null)
  const stRef = useRef(null)
  // 回调放 ref：切换"选择层级"会换掉 onPick 引用，放进依赖会让整个 WebGL 场景重建
  const pickRef = useRef(onPick)
  const hoverRef = useRef(onHover)
  const errRef = useRef(onError)
  //: 双击 / 右击 / 点空白 三个新手势也放进 ref（同 onPick，避免重建场景）
  const gestureRef = useRef({ monomer: onPickMonomer, chain: onPickChain,
    clear: onClearSelection })
  pickRef.current = onPick
  hoverRef.current = onHover
  errRef.current = onError
  gestureRef.current = { monomer: onPickMonomer, chain: onPickChain,
    clear: onClearSelection }

  useEffect(() => {
    const box = boxRef.current
    if (!box) return undefined
    let renderer
    let stEnv = null                     // 环境贴图（卸载时释放）
    try {
      // ⚠️ preserveDrawingBuffer 必须开：关着的时候，浏览器在合成后可以把
      //    WebGL 的绘图缓冲清掉，而此时若 React 因为悬停换了覆盖层（原子信息框）
      //    重排 DOM，就会有一帧拿到"已被清空"的画布 —— 表现就是悬停切换原子时
      //    **短暂闪一下白屏**（用户实测）。
      renderer = new THREE.WebGLRenderer({
        antialias: RENDER.antialias, alpha: false, preserveDrawingBuffer: true,
      })
    } catch (e) {
      box.innerHTML = '<div class="placeholder"><p>这个浏览器/环境不能创建 WebGL '
        + '上下文，三维视图不可用（分析功能不受影响）。</p></div>'
      return undefined
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, RENDER.maxPixelRatio))
    // 色彩管理 / 色调映射：sRGB 输出 + ACES 曲线，高光不会过曝成死白
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = RENDER.exposure
    // 画布底色：深色（renderConfig.RENDER.background，默认 #101722）
    renderer.setClearColor(new THREE.Color(RENDER.background))
    box.appendChild(renderer.domElement)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(RENDER.background)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.5, 40000)
    // near 会在每次渲染时按相机距离自适应（见 render()），far 留够最大缩放
    // ---- 光照：柔和多光源（强度/方位都来自 renderConfig.LIGHTING）
    scene.add(new THREE.AmbientLight(0xffffff, LIGHTING.ambient))
    const hemi = new THREE.HemisphereLight(0xdfe9ff, 0x2b2f38, LIGHTING.hemisphere)
    scene.add(hemi)
    const lMain = new THREE.DirectionalLight(0xffffff, LIGHTING.main)
    lMain.position.set(...LIGHTING.mainPos)
    scene.add(lMain)
    const lFill = new THREE.DirectionalLight(0xffffff, LIGHTING.fill)
    lFill.position.set(...LIGHTING.fillPos)
    scene.add(lFill)
    const lRim = new THREE.DirectionalLight(0xbfd4ff, LIGHTING.rim)
    lRim.position.set(...LIGHTING.rimPos)
    scene.add(lRim)
    // ---- 环境贴图：给 MeshPhysicalMaterial 一点环境反射（粗糙度高，不会镜面反光）。
    // 用一张**程序生成的等距柱状贴图**（canvas 渐变）走 PMREM，不建任何 Mesh ——
    // 原来自检里专门盯着"不要为单个原子建 THREE.Mesh"，这里也不能破例。
    if (RENDER.useEnvironment) {
      try {
        const cv = document.createElement('canvas')
        cv.width = 64
        cv.height = 32
        const ctx = cv.getContext('2d')
        if (ctx) {
          const g = ctx.createLinearGradient(0, 0, 0, 32)
          g.addColorStop(0, '#4a5a78')
          g.addColorStop(0.45, '#1b2230')
          g.addColorStop(1, '#12161f')
          ctx.fillStyle = g
          ctx.fillRect(0, 0, 64, 32)
          // 一块柔光斑（模拟主光方向的天光，给球体一点边缘反射）
          const rg = ctx.createRadialGradient(14, 8, 1, 14, 8, 12)
          rg.addColorStop(0, 'rgba(223,233,255,0.85)')
          rg.addColorStop(1, 'rgba(223,233,255,0)')
          ctx.fillStyle = rg
          ctx.fillRect(0, 0, 64, 32)
        }
        const tex = new THREE.CanvasTexture(cv)
        tex.mapping = THREE.EquirectangularReflectionMapping
        tex.colorSpace = THREE.SRGBColorSpace
        const pmrem = new THREE.PMREMGenerator(renderer)
        const envTex = pmrem.fromEquirectangular(tex).texture
        scene.environment = envTex
        tex.dispose()
        pmrem.dispose()
        stEnv = envTex
      } catch { /* 环境贴图失败就用纯光照，不影响显示 */ }
    }

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
    // ---- 相机存档：视角/注视点直接写 localStorage（按体系 sid 区分）。
    // 为什么放在 Viewer 里而不是状态层：刷新（F5）后状态层的时序不可靠
    // （data 还没回来时 viewerRef 可能为空），这里在渲染循环里自给自足。
    st.viewKey = `mdta.structview:${sid || 'default'}`
    let lastViewSave = 0
    st.saveView = () => {
      const now = typeof performance !== 'undefined' ? performance.now() : Date.now()
      if (now - lastViewSave < 400) return
      lastViewSave = now
      try {
        if (!st.sph) return
        window.localStorage.setItem(st.viewKey, JSON.stringify({
          sph: { theta: st.sph.theta, phi: st.sph.phi, radius: st.sph.radius },
          target: st.target
            ? { x: st.target.x, y: st.target.y, z: st.target.z } : null,
        }))
      } catch { /* 配额满/隐私模式：忽略 */ }
    }
    st.loadSavedView = () => {
      try {
        const raw = window.localStorage.getItem(st.viewKey)
        return raw ? JSON.parse(raw) : null
      } catch { return null }
    }
    stRef.current = st

    const render = () => {
      // 缩放引起的"屏幕尺度重算"集中在每帧做一次（滚轮事件每帧可能有多个），
      // 避免同一帧里反复重写几万个实例矩阵 → 掉帧/抖动。
      if (st._needRescale) {
        st._needRescale = false
        try { st.refresh?.() } catch { /* 缩放后重算失败不应打断渲染 */ }
      }
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
        st.saveView?.()          // 视角一变就把"上次看到的位置"写进存档
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
    // ⚠️ 两个坑（用户实测"点的是别的链却识别成别的链"）：
    //   1. 实例号 ≠ 显示序号：每个元素一个 InstancedMesh，第 k 个实例对应该元素
    //      原子列表里的第 k 个（`userData.slots[k]` 才是**显示序号**）。
    //      以前直接拿 `instanceId` 当显示序号，于是点到 H 会落到全局第 instanceId
    //      个原子上 —— 完全是另一条链的原子。
    //   2. 以前"第一个命中的网格"就算数：C 球和 H 球分属两个网格，先遍历到的
    //      那个不一定是更近的那个。现在取**所有网格里最近**的交点。
    const pickAt = (cx, cy) => {
      const rect = box.getBoundingClientRect()
      const ndc = new THREE.Vector2(
        ((cx - rect.left) / rect.width) * 2 - 1,
        -((cy - rect.top) / rect.height) * 2 + 1)
      st.raycaster.setFromCamera(ndc, camera)
      let bestSlot = null
      let bestDist = Infinity
      let bestMesh = null
      for (const m of st.meshes) {
        if (!m.visible) continue
        const hits = st.raycaster.intersectObject(m, false)
        if (hits.length && hits[0].distance < bestDist) {
          bestDist = hits[0].distance
          bestSlot = m.userData.slots[hits[0].instanceId]
          bestMesh = m
        }
      }
      if (bestSlot === null || bestMesh === null) return null
      return bestMesh.userData.displayIndex[bestSlot]
    }

    /** 兜底拾取：射线没打到任何球时（氢球很小、点在球的边缘外），
     *  在屏幕 14 px 内找**最近的原子**。找不到才算"点了空白处"。
     *  用户实测："我明明点的是别的链却识别成别的链" —— 一半是上面实例号映射
     *  的 bug，另一半是点在球边缘直接落空、被当成点空白而清空选择。 */
    const nearestScreenAtom = (cx, cy, maxPx = 14) => {
      const rect = box.getBoundingClientRect()
      const v = new THREE.Vector3()
      const proj = new THREE.Matrix4()
        .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
      const [ox, oy, oz] = st.offset || [0, 0, 0]
      let best = null
      let bestD = maxPx * maxPx
      const xyz = data.xyz
      for (let i = 0; i < xyz.length; i += 1) {
        v.set(xyz[i][0] - ox, xyz[i][1] - oy, xyz[i][2] - oz).applyMatrix4(proj)
        if (v.z > 1) continue                          // 在相机后方，忽略
        const sx = (v.x * 0.5 + 0.5) * rect.width + rect.left
        const sy = (-v.y * 0.5 + 0.5) * rect.height + rect.top
        const dx = sx - cx
        const dy = sy - cy
        const d = dx * dx + dy * dy
        if (d < bestD) { bestD = d; best = data.index[i] }
      }
      return best
    }

    // dragging: 0 无 / 1 旋转 / 2 平移；moved 记录累计位移，用来区分"单击"与"拖动"
    let dragging = 0
    let moved = 0
    let last = { x: 0, y: 0 }
    //: 滚轮"朝光标原子钻"的拾取缓存：同一次滚轮手势沿用**同一个点**，
    //: 否则画面一移动光标就落到别的原子上，注视点在两个原子之间来回跳 = 抖。
    let cursorPoint = null
    let cursorPickAt = { x: 0, y: 0, t: 0 }
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
      let idx = pickAt(ev.clientX, ev.clientY)
      // 点在球边缘外一点点：先按"屏幕 14 px 内最近的原子"兜底，避免把
      // 一次普通点选当成"点了空白处"而把整份选择清掉。
      if (idx === null || idx === undefined) {
        idx = nearestScreenAtom(ev.clientX, ev.clientY)
      }
      // 点在空白处（14 px 内也没有原子）= 撤销全部选择（用户要求）
      if (idx === null || idx === undefined) {
        gestureRef.current?.clear?.()
        return
      }
      // 右击：选中所在链；已经正好选中这条链时，再右击 → 同类全部链
      if (ev.button === 2) {
        gestureRef.current?.chain?.(idx)
        return
      }
      if (ev.detail >= 2) {
        // 双击 = 选中该原子所在的**单体（重复单元）**（用户要求）；
        // Ctrl/⌘+双击 仍保留原来的"镜头聚焦到该原子"的手势。
        if (ev.ctrlKey || ev.metaKey || ev.shiftKey) st.focusOn?.(idx)
        else gestureRef.current?.monomer?.(idx)
        return
      }
      pickRef.current?.(idx, ev.ctrlKey || ev.metaKey)
    }
    const onWheel = (ev) => {
      ev.preventDefault()
      const before = st.sph.radius
      // 上下限都按体系自身算（见 computeZoomLimits）：
      //   最近 = 最大球直径占视野 1.8 倍（能钻进原子内部）；最远 = 模拟盒 ≈ 视野 1/3
      const lo = st.radiusMin || MIN_RADIUS_A
      const hi = st.radiusMax || MAX_RADIUS_A
      const after = Math.min(hi, Math.max(lo,
        st.sph.radius * Math.exp(ev.deltaY * ZOOM_PER_DELTA)))
      st.sph.radius = after
      // 放大时**朝鼠标下的原子钻**（用户要求"能钻进分子的某一处"）。
      // ⚠️ 抖动根因在这里：以前每个滚轮事件都重新拾取，"光标下那颗原子"会随着
      // 画面移动而变化 → 注视点在两颗原子之间来回 lerp，看起来就是画面在抖。
      // 现在**一次手势沿用同一个缓存点**：只有光标移动 >8 px 或隔了 600 ms 才重新拾取。
      const moved = before - after > 1e-9
      if (moved) {
        const now = (typeof performance !== 'undefined' ? performance.now() : Date.now())
        const stale = !cursorPoint
          || Math.abs(ev.clientX - cursorPickAt.x) > 8
          || Math.abs(ev.clientY - cursorPickAt.y) > 8
          || now - cursorPickAt.t > 600
        if (stale) {
          cursorPickAt = { x: ev.clientX, y: ev.clientY, t: now }
          cursorPoint = null
          try {
            const idx = pickAt(ev.clientX, ev.clientY)
            const dp = idx === null || idx === undefined
              ? null : (st.posOf ? st.posOf.get(idx) : null)
            if (dp !== null && dp !== undefined && data.xyz[dp]) {
              const [ox, oy, oz] = st.offset || [0, 0, 0]
              cursorPoint = new THREE.Vector3(
                data.xyz[dp][0] - ox, data.xyz[dp][1] - oy, data.xyz[dp][2] - oz)
            }
          } catch { /* 拾取失败不影响缩放本身 */ }
        }
        if (cursorPoint) {
          const k = Math.min(0.5, Math.max(0.06, 1 - after / before))
          st.target.lerp(cursorPoint, k)
        }
      }
      // 尺度重算**推迟到渲染帧里做一次**（见 render()）：以前每个滚轮事件都同步
      // 重写几万个实例矩阵，连续滚轮时掉帧，也是"抖"的一部分。
      if (Math.abs(after / before - 1) > 0.005) st._needRescale = true
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
      try { stEnv?.dispose?.() } catch { /* 环境贴图释放失败可忽略 */ }
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
    // 恢复存档的视角（F5 后）：只在新数据建几何这次用一次，之后清空
    if (pendingView) st.pendingView = pendingView
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
      // 相机状态（视角 / 注视点）：structureState 定时存盘，F5 后原样恢复
      getView: () => {
        const st = stRef.current
        if (!st || !st.sph) return null
        return {
          sph: { theta: st.sph.theta, phi: st.sph.phi, radius: st.sph.radius },
          target: st.target
            ? { x: st.target.x, y: st.target.y, z: st.target.z } : null,
        }
      },
      setView: (v) => {
        const st = stRef.current
        if (!st || !v || !v.sph) return
        st.sph = { theta: v.sph.theta, phi: v.sph.phi, radius: v.sph.radius }
        if (v.target) st.target = new THREE.Vector3(v.target.x, v.target.y, v.target.z)
        st.refresh?.()
        st.kick()
      },
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
