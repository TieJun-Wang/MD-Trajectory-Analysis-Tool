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
const RADIUS_SCALE = 0.8           // 显示半径 = 共价半径 × 系数（Å）

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
          <div className="chartsub">
            {nAtoms.toLocaleString()} 原子
            {data ? ` · 显示 ${data.n_displayed.toLocaleString()} 个`
              + (data.truncated
                ? `（按整残基截断：${data.n_residues_shown.toLocaleString()} / `
                  + `${data.n_residues.toLocaleString()} 残基）` : '') : ' · 结构未载入'}
            {selected.size ? ` · 高亮 ${selected.size.toLocaleString()} 原子` : ''}
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
                  onPick={st.pick} onHover={st.setHover} binding={st.viewerRef}
                  onError={st.setErr} />
        )}
        {data && hover !== null && (
          <AtomTip data={data} atom={hover} />
        )}
        {data && (
          <div className="viewhint">
            左键拖动旋转 · 滚轮缩放 · 右键拖动平移 · 单击原子高亮（Ctrl 单击加减）
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
    if (st.bondLines) {
      scene.remove(st.bondLines); st.bondLines.geometry.dispose(); st.bondLines = null
    }
    if (st.labelGroup) { scene.remove(st.labelGroup); st.labelGroup = null }

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
    // 屏幕空间最小半径：大体系按真实共价半径画只有几个像素，看起来像"什么都没有"。
    // 按初始相机距离估出"每像素多少 Å"，保证初始视角下原子 ≥ ~7px 直径。
    const homeDist = st.home.radius
    const halfH = homeDist * Math.tan(Math.PI / 8)       // 45°/2
    const viewEl = st.el || null
    const viewH = Math.max(1, (viewEl && viewEl.clientHeight) || 420)
    st.minRadA = 3.5 * ((2 * halfH) / viewH)

    // 按元素分组 → 每种元素一个 InstancedMesh（draw call 少、半径可不同）
    const byEl = new Map()
    for (let i = 0; i < n; i += 1) {
      const el = data.elements[i] || 'X'
      if (!byEl.has(el)) byEl.set(el, [])
      byEl.get(el).push(i)
    }
    const baseGeo = new THREE.SphereGeometry(1, 10, 8)
    const baseMat = new THREE.MeshLambertMaterial()
    const m4 = new THREE.Matrix4()
    const radOf = (i) => Math.max(0.25, Math.min(2.5, Math.max(
      (data.radii[i] || 1.4) * RADIUS_SCALE, st.minRadA || 0)))
    for (const [el, list] of byEl) {
      const mesh = new THREE.InstancedMesh(baseGeo.clone(), baseMat.clone(), list.length)
      mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage)
      const col = new THREE.Color(colorOf(el))
      list.forEach((i, k) => {
        const rad = radOf(i)
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
}

export function applyStateToScene(st, data, selected, onlySelected = false) {
    const hi = new THREE.Color(HIGHLIGHT)
    const m4 = new THREE.Matrix4()
    const [ox, oy, oz] = st.offset || [0, 0, 0]
    const radOf = (i) => Math.max(0.25, Math.min(2.5, Math.max(
      (data.radii[i] || 1.4) * RADIUS_SCALE, st.minRadA || 0)))
    for (const mesh of st.meshes) {
      const base = new THREE.Color(colorOf(mesh.userData.element))
      mesh.userData.slots.forEach((i, k) => {
        const atom = data.index[i]
        const isSel = !!(selected && selected.has(atom))
        mesh.setColorAt(k, isSel ? hi : base)
        // 「只显示已选」：没选中的缩到不可见（**编号不变**，只是不画）
        const s = onlySelected && !isSel ? 1e-6 : radOf(i)
        m4.makeScale(s, s, s)
        m4.setPosition(data.xyz[i][0] - ox, data.xyz[i][1] - oy, data.xyz[i][2] - oz)
        mesh.setMatrixAt(k, m4)
      })
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    }
    st.kick()
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
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
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
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 20000)
    scene.add(new THREE.AmbientLight(0xffffff, 0.72))
    const l1 = new THREE.DirectionalLight(0xffffff, 0.75)
    l1.position.set(1, 1.2, 1.6)
    scene.add(l1)
    const l2 = new THREE.DirectionalLight(0xffffff, 0.3)
    l2.position.set(-1, -0.8, -1.2)
    scene.add(l2)

    const st = {
      renderer, scene, camera,
      meshes: [], bondLines: null, labelGroup: null,
      sph: { theta: 0.6, phi: 1.1, radius: 100 },
      target: new THREE.Vector3(),
      home: null,
      raycaster: new THREE.Raycaster(),
      hoverIdx: null,
      posOf: new Map(),
      center: new THREE.Vector3(),
    }
    st.el = box                 // 数据/状态 effect 里要量尺寸，只能通过 st 拿
    stRef.current = st

    const render = () => {
      const { theta, phi, radius } = st.sph
      const t = st.target
      camera.position.set(
        t.x + radius * Math.sin(phi) * Math.cos(theta),
        t.y + radius * Math.cos(phi),
        t.z + radius * Math.sin(phi) * Math.sin(theta))
      camera.lookAt(t)
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
        st.sph.theta -= dx * 0.008
        st.sph.phi = Math.min(Math.PI - 0.02, Math.max(0.02, st.sph.phi - dy * 0.008))
      } else {
        const scale = st.sph.radius * 0.0016
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
      if (idx !== null && idx !== undefined) {
        pickRef.current?.(idx, ev.ctrlKey || ev.metaKey)
      }
    }
    const onWheel = (ev) => {
      ev.preventDefault()
      st.sph.radius = Math.min(20000, Math.max(0.5,
        st.sph.radius * Math.exp(ev.deltaY * 0.0011)))
      kick()
    }
    const onCtx = (ev) => ev.preventDefault()

    box.addEventListener('pointerdown', onDown)
    box.addEventListener('pointermove', onMove)
    box.addEventListener('pointerup', onUp)
    box.addEventListener('wheel', onWheel, { passive: false })
    box.addEventListener('contextmenu', onCtx)

    let hoverRaf = 0
    const hoverLoop = () => {
      if (st.pendingHover) {
        const p = st.pendingHover
        st.pendingHover = null
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
        st.kick()
      },
    }
  }, [binding])

  return <div ref={boxRef} className="viewer-canvas" />
}
