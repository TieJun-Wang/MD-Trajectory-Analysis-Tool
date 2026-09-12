// 从 PyPI 下载指定包的 wheel 到本地目录（绕过 pip 的 SSL 问题）
// 用法: node webapp/_getwheel.mjs <outdir> <pkg> [pkg...]
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const [outdir, ...pkgs] = process.argv.slice(2);
if (!outdir || pkgs.length === 0) {
  console.error('用法: node _getwheel.mjs <outdir> <pkg> [pkg...]');
  process.exit(2);
}
mkdirSync(outdir, { recursive: true });

const PY_TAGS = ['py3-none-any', 'py2.py3-none-any', 'py3-none-win_amd64'];
const isCompat = (f) =>
  f.packagetype === 'bdist_wheel' &&
  (PY_TAGS.some((t) => f.filename.endsWith(t + '.whl')) ||
    /(cp313|cp312|cp311)-.*(win_amd64|any)\.whl$/.test(f.filename));

for (const pkg of pkgs) {
  const res = await fetch(`https://pypi.org/pypi/${pkg}/json`);
  if (!res.ok) {
    console.log(`FAIL ${pkg}: HTTP ${res.status}`);
    continue;
  }
  const j = await res.json();
  const files = (j.urls || []).filter(isCompat);
  if (!files.length) {
    console.log(`FAIL ${pkg}: 没有兼容的 wheel（可能需要源码编译）`);
    continue;
  }
  // 优先纯 python wheel
  files.sort((a, b) => {
    const score = (f) => (f.filename.includes('py3-none-any') ? 0 : 1);
    return score(a) - score(b);
  });
  const f = files[0];
  const buf = Buffer.from(await (await fetch(f.url)).arrayBuffer());
  const dest = join(outdir, f.filename);
  writeFileSync(dest, buf);
  console.log(`OK   ${pkg} ${j.info.version} -> ${f.filename} (${buf.length} bytes)`);
}
