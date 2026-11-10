# -*- coding: utf-8 -*-
"""取消令牌：长时间的分析循环里定期检查，用户点「取消」能**立刻**停下。"""
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
