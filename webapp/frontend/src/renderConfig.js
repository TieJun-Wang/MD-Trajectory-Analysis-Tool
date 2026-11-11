/**
 * 三维球棍模型的**视觉配置中心**（用户要求：不要把视觉参数硬编码在渲染函数里）。
 *
 * 这里只放"怎么画"的参数，不含任何坐标/分析数据 —— 改这里的值不会影响任何
 * 计算结果，只影响三维画面的观感。结构 / 分析逻辑仍由后端与 structureState 负责。
 *
 * 结构：
 *   ElementColorConfig  元素颜色
 *   AtomRadiusConfig    原子显示半径（Å，**仅显示**）
 *   BondRadiusConfig    化学键半径（按成键元素对）
 *   MaterialConfig      球 / 棍材质参数
 *   LightingConfig      场景光照
 *   RenderQualityConfig 质量档位（球面细分、键径向分段）
 *   DisplayConfig       以上所有 + 背景色 / 高亮色 / 盒棱 统一出口
 */

/* ------------------------------------------------------------------ 元素颜色 */
/** 元素颜色（适度调整过的 CPK：不刺眼、不荧光，深浅都能区分元素） */
export const ELEMENT_COLORS = {
  C: '#50545C',     // 石墨灰
  H: '#E8E8E8',     // 珍珠白
  O: '#E54848',     // 珊瑚红
  N: '#4169D8',     // 深蓝
  S: '#E5BD36',     // 金黄
  P: '#E99A35',     // 橙
  CL: '#42A65A',    // 草绿
  F: '#65C9A5',     // 青绿
  NA: '#9670D5',    // 紫
  CA: '#62A6B8',    // 青灰
  // ---- 常用扩展（不在需求表里，但同族元素给个中性色能少一次"全是粉色"）
  BR: '#A6584B', SI: '#C8B08A', B: '#C77A7A', LI: '#8E7BC8',
  K: '#7E6BC0', MG: '#7DB46C', FE: '#B7643C', ZN: '#7D80B0', I: '#8A5A8A',
}
/** 未识别元素的中性灰（不抢眼，也不至于像"漏画的粉色"） */
export const DEFAULT_COLOR = '#7B8090'

export function elementColor(el) {
  const key = String(el || '').trim().toUpperCase()
  return ELEMENT_COLORS[key] || DEFAULT_COLOR
}

/* -------------------------------------------------------------- 原子显示半径 */
/** 元素显示半径（Å，**仅显示**；不是范德华/共价半径） */
export const ELEMENT_RADII = {
  C: 0.38, H: 0.25, O: 0.36, N: 0.36, S: 0.42, P: 0.40,
  CL: 0.42, F: 0.34, NA: 0.40, CA: 0.42,
}
export const DEFAULT_RADIUS = 0.36
/** 全局原子尺寸缩放系数（用户在配置里改这一个数即可整体放大/缩小） */
export const ATOM_SCALE = 1.0

export function displayRadius(el) {
  const key = String(el || '').trim().toUpperCase()
  const r = ELEMENT_RADII[key]
  return (Number.isFinite(r) ? r : DEFAULT_RADIUS) * ATOM_SCALE
}

/* ---------------------------------------------------------------- 化学键半径 */
/** 成键元素对 → 键半径（Å）。键按元素**排序**后查表，缺省 0.10 Å。 */
export const BOND_RADII = {
  'C-C': 0.12, 'C-H': 0.09, 'C-O': 0.11, 'H-N': 0.09,
}
export const BOND_RADIUS_DEFAULT = 0.10

export function bondRadius(elA, elB) {
  const a = String(elA || '').trim().toUpperCase()
  const b = String(elB || '').trim().toUpperCase()
  const key = a <= b ? `${a}-${b}` : `${b}-${a}`
  const r = BOND_RADII[key]
  return Number.isFinite(r) ? r : BOND_RADIUS_DEFAULT
}

/* ------------------------------------------------------------------ 材质参数 */
/** 原子球：圆润柔和的光泽，避免"塑料玩具"和金属感 */
export const ATOM_MATERIAL = {
  roughness: 0.28,
  metalness: 0.0,
  clearcoat: 0.25,
  clearcoatRoughness: 0.22,
  ior: 1.45,
  reflectivity: 0.35,
  envMapIntensity: 0.35,
}
/** 化学键：比球略哑光，颜色由两端元素决定（半棍）或中性灰 */
export const BOND_MATERIAL = {
  roughness: 0.35,
  metalness: 0.0,
  clearcoat: 0.10,
  clearcoatRoughness: 0.30,
  reflectivity: 0.25,
  envMapIntensity: 0.25,
}

/* -------------------------------------------------------------------- 光照 */
/** 柔和多光源：主光左上、补光右下、边缘光对侧提立体感 */
export const LIGHTING = {
  ambient: 0.35,                       // AmbientLight
  hemisphere: 0.8,                     // HemisphereLight（天空/地面）
  main: 2.0,                           // DirectionalLight 主光（左上方）
  fill: 0.6,                           // DirectionalLight 补光（减少死黑）
  rim: 1.0,                            // 边缘辅助光
  mainPos: [1, 1.2, 0.9],              // 左上方（相对注视点方向）
  fillPos: [-1, -0.6, -0.8],
  rimPos: [-0.2, 0.9, -1.0],
}

/* ------------------------------------------------------------ 渲染质量档位 */
/** 球面细分 / 键径向分段。默认 standard（24×24 / 12）。 */
export const QUALITY = {
  performance: { sphereW: 16, sphereH: 16, bondRadial: 8, ringRadial: 6 },
  standard: { sphereW: 24, sphereH: 24, bondRadial: 12, ringRadial: 8 },
  high: { sphereW: 32, sphereH: 32, bondRadial: 16, ringRadial: 10 },
  screenshot: { sphereW: 48, sphereH: 48, bondRadial: 24, ringRadial: 14 },
}
export const QUALITY_LEVEL = 'standard'
export const quality = () => QUALITY[QUALITY_LEVEL] || QUALITY.standard
/** 按体系原子数自动降档：>2 万原子用 performance 档（几十万个原子时 24×24 球会
 *  变成顶点瓶颈）。显式选了 high/screenshot 时**不**自动降档（用户说了算）。 */
export function qualityFor(atomCount) {
  const lvl = QUALITY_LEVEL
  if (lvl !== 'standard') return QUALITY[lvl] || QUALITY.standard
  return (Number(atomCount) || 0) > 20000 ? QUALITY.performance : QUALITY.standard
}

/* --------------------------------------------------- 渲染器 / 背景 / 高亮 */
export const RENDER = {
  antialias: true,
  background: '#101722',      // 深色背景（近黑，带一点蓝）
  exposure: 1.0,
  useEnvironment: true,       // 环境贴图（RoomEnvironment，给 PhysicalMaterial 反射）
  maxPixelRatio: 2,
}
/** 选中的原子/棍高亮色（用户要求"明显"）与放大系数 */
export const HIGHLIGHT_COLOR = '#ffe600'
export const HIGHLIGHT_SCALE = 1.15
/** 模拟盒棱（白色细柱，屏幕像素恒定） */
export const BOX = { color: '#ffffff', px: 1.8 }
/** 环的参考色（苯环细圈；旧版细线的颜色，保留给需要时用） */
export const RING_COLOR = '#ffffff'

/* ------------------------------------------------------- 芳香环（苯环）圆环 */
/** 芳香环内部的半透明圆环（Three.js TorusGeometry）。
 *  只对**已识别的芳香环**画：后端 `mdta.rings.analyze_rings` 判定的
 *  "6 元碳环 + 6 根环键均 ≤1.50 Å"（`is_benzene`）—— 前端不按几何形状自己猜。 */
export const AROMATIC_RING = {
  color: '#91AFC8',
  opacity: 0.48,
  transparent: true,
  roughness: 0.5,
  metalness: 0.0,
  emissive: '#263F55',
  emissiveIntensity: 0.15,
  depthWrite: false,
  /** 圆环半径 = 苯环内切半径 × 该系数 */
  radiusFrac: 0.78,
  /** 管道半径（Å） */
  tube: 0.045,
  /** 沿圆周的分段数（越大越平滑） */
  tubularSegments: 96,
  /** 截面分段（径向） */
  radialSegments: 12,
  /** 默认显示；界面上可单独开关（`st.showRings`） */
  visibleByDefault: true,
}

/** 统一出口：以后要加"显示风格"，在这里加一套参数即可 */
export const DisplayConfig = {
  colors: ELEMENT_COLORS, defaultColor: DEFAULT_COLOR,
  radii: ELEMENT_RADII, defaultRadius: DEFAULT_RADIUS, atomScale: ATOM_SCALE,
  bondRadii: BOND_RADII, bondDefault: BOND_RADIUS_DEFAULT,
  atomMaterial: ATOM_MATERIAL, bondMaterial: BOND_MATERIAL,
  lighting: LIGHTING, quality: QUALITY, qualityLevel: QUALITY_LEVEL,
  render: RENDER, highlight: HIGHLIGHT_COLOR, highlightScale: HIGHLIGHT_SCALE,
  box: BOX, ringColor: RING_COLOR, aromaticRing: AROMATIC_RING,
}
