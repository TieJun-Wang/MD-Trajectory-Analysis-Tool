# -*- coding: utf-8 -*-
"""启动 Web 服务。

用法::

    python -m webapp.backend.run                  # http://127.0.0.1:8000
    python -m webapp.backend.run --port 8080
    python -m webapp.backend.run --host 0.0.0.0   # 局域网可访问（注意安全）
    python -m webapp.backend.run --reload         # 开发时自动重载
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mdta-web",
        description="MD 轨迹分析工具 —— Web 后端（FastAPI）",
    )
    p.add_argument("--host", default=os.environ.get("MDTA_WEB_HOST", "127.0.0.1"),
                   help="监听地址，默认 127.0.0.1（仅本机）")
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("MDTA_WEB_PORT", "8000")),
                   help="监听端口，默认 8000")
    p.add_argument("--reload", action="store_true", help="代码变更自动重载（开发用）")
    p.add_argument("--log-level", default="info",
                   choices=("critical", "error", "warning", "info", "debug", "trace"))
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    args = build_parser().parse_args(argv)
    import uvicorn

    dist = _HERE.parent / "frontend" / "dist"
    print("=" * 68)
    print("  MD Trajectory Analysis Tool —— Web 服务")
    print("=" * 68)
    print(f"  地址        : http://{args.host}:{args.port}")
    print(f"  接口文档    : http://{args.host}:{args.port}/docs")
    print(f"  前端产物    : {'已就绪' if dist.is_dir() else '未构建'}"
          f"  ({dist})")
    if not dist.is_dir():
        print("  提示        : 开发时在前端目录执行 npm run dev；"
              "发布前执行 npm install && npm run build")
    print("=" * 68)

    if args.reload:
        uvicorn.run("webapp.backend.app:app", host=args.host, port=args.port,
                    reload=True, log_level=args.log_level)
    else:
        from .app import app as _app
        uvicorn.run(_app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
