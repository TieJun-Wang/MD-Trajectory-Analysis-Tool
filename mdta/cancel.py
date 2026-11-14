# -*- coding: utf-8 -*-
"""取消令牌：长时间的分析循环里定期检查，用户点「取消」能**立刻**停下。

为什么要分两种检查
------------------
"每一项开始前"检查一次是不够的（用户实测："点了取消没效果，计算进程还在，
界面一直卡在计算中"）。真正的粒度由**两次检查之间跑了多少计算**决定：

- 逐帧循环（``frame_iterator``）：每帧检查一次 —— 4.7 万原子体系上单帧就要
  几秒到几十秒，光靠这个，点取消还是要等一帧；
- 帧内的分块循环（邻居表、球谐、直方图、解包裹、MSD 的各个 lag）：
  用 :func:`tick` 每 ``every`` 次迭代问一次，开销可忽略（一次
  ``Event.is_set`` 约 50 ns）。

于是"取消"的实际延迟 ≈ 一个分块（几十毫秒），而不是一项分析、甚至一帧。
"""
import threading

_EVENT = None


class AnalysisCancelled(RuntimeError):
    """用户取消了本次分析（由 pipeline 捕获，记为"已取消"而不是失败）。"""


def bind(ev):
    global _EVENT
    _EVENT = ev


def clear():
    global _EVENT
    _EVENT = None


def is_set() -> bool:
    ev = _EVENT
    return bool(ev is not None and ev.is_set())


def check():
    """在长循环里调用：已请求取消就抛 :class:`AnalysisCancelled`。"""
    if is_set():
        raise AnalysisCancelled("用户取消了本次分析")


def tick(counter: int, every: int = 1) -> None:
    """**分块循环**里的便宜检查：每 ``every`` 次迭代才真的问一次事件。

    ``tick(k, 4096)`` 的额外开销只有一次取模；``every=1`` 等价于
    :func:`check`（用于每次迭代本身就几百毫秒的重活）。
    """
    ev = _EVENT
    if ev is None:
        return
    if every > 1 and (int(counter) % int(every)):
        return
    if ev.is_set():
        raise AnalysisCancelled("用户取消了本次分析")
