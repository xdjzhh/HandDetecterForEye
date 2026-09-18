# -*- coding: utf-8 -*-
"""
app_log.py —— 全局日志开关 + 终端/文件双写（Tee）
========================================================================
唯一安装点：server.py 启动最早期（读到 config.json 的 log_enabled 之后、
import 业务模块 hands_tasks/cv2 之前），保证启动日志也进文件。

开关语义（config.json "log_enabled"，默认 true）：
  true  = 所有写 stdout 的日志【同时】输出到终端 和
          项目根/logs/handdetecter_YYYY-MM-DD.log（按天切分、保留 7 天）；
          stderr（Flask werkzeug、未捕获异常堆栈等）同样双写。
  false = 静默模式：stdout 日志一律丢弃（控制台不刷屏、不产生日志文件）；
          stderr 原样直达终端——错误/崩溃堆栈始终可见，不被开关吞掉。

注意：
  · 这是"全量镜像"方案，项目里现有 print 无需逐个改造即自动落盘。
  · 线程安全（工作线程 + Flask 线程都会 print），文件写入加锁。
  · 重复安装幂等（只第一次生效）。
"""

import os
import sys
import glob
import time
import atexit
import datetime
import threading

_LOG_KEEP_DAYS = 7          # 日志文件保留天数
_LOG_PREFIX = "handdetecter_"
_installed = False
_lock = threading.Lock()


class _Tee:
    """同一个 write 同时落终端与日志文件；跨天自动换文件。"""

    def __init__(self, terminal, log_dir: str):
        self._terminal = terminal
        self._log_dir = log_dir
        self._day = datetime.date.today()
        self._fp = self._open_fp()

    def _log_path(self, day) -> str:
        return os.path.join(self._log_dir, f"{_LOG_PREFIX}{day.isoformat()}.log")

    def _open_fp(self):
        os.makedirs(self._log_dir, exist_ok=True)
        # 行缓冲文本模式：崩溃前最后一行也能落盘；utf-8 容纳中文/手势符号
        return open(self._log_path(self._day), "a", encoding="utf-8", buffering=1)

    def _maybe_rotate(self):
        today = datetime.date.today()
        if today != self._day:
            try:
                self._fp.close()
            except Exception:
                pass
            self._day = today
            self._fp = self._open_fp()

    def write(self, s):
        if not s:
            return 0
        with _lock:
            self._maybe_rotate()
            # 终端可能在无控制台环境为 None
            if self._terminal is not None:
                try:
                    self._terminal.write(s)
                except Exception:
                    pass
            try:
                self._fp.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        with _lock:
            if self._terminal is not None:
                try:
                    self._terminal.flush()
                except Exception:
                    pass
            try:
                self._fp.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._terminal.isatty()
        except Exception:
            return False

    def fileno(self):
        # 不能伪造文件描述符：有些库会 fileno() 探测后直接写终端
        raise OSError("tee stream has no fileno")

    @property
    def encoding(self):
        return getattr(self._terminal, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._terminal, "errors", None)

    def close(self):
        with _lock:
            try:
                self._fp.close()
            except Exception:
                pass


class _NullStream:
    """静默模式的 stdout：一切日志写进来即丢弃（零开销、不落盘）。"""

    def write(self, s):
        return len(s) if s else 0

    def flush(self):
        pass

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("null stream has no fileno")

    @property
    def encoding(self):
        return "utf-8"

    @property
    def errors(self):
        return None


def _purge_old_logs(log_dir: str) -> None:
    """删除超过保留期的旧日志（按文件 mtime；当天文件不动）。"""
    try:
        cutoff = time.time() - _LOG_KEEP_DAYS * 86400
        for p in glob.glob(os.path.join(log_dir, f"{_LOG_PREFIX}*.log")):
            try:
                if os.path.isfile(p) and os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def setup_logging(enabled: bool, project_root: str) -> bool:
    """安装全局 stdout/stderr 镜像。返回是否实际安装（重复调用不生效）。

    Args:
        enabled: True=终端+文件双写；False=静默 stdout，stderr 保持原样。
        project_root: 项目根目录，日志落到 <root>/logs/。
    """
    global _installed
    if _installed:
        return False
    _installed = True

    if not enabled:
        # 静默：只吞 stdout 日志；stderr 不动（异常堆栈照常直达终端）
        sys.stdout = _NullStream()
        return True

    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    _purge_old_logs(log_dir)

    out_tee = _Tee(sys.__stdout__, log_dir)
    err_tee = _Tee(sys.__stderr__, log_dir)
    sys.stdout = out_tee
    sys.stderr = err_tee
    atexit.register(out_tee.close)
    atexit.register(err_tee.close)

    # 这行会同时出现在终端和日志文件里（安装成功的自证标记）
    print(f"[app_log] 日志已开启：终端 + {out_tee._log_path(out_tee._day)}（保留 {_LOG_KEEP_DAYS} 天，跨天自动切分）")
    return True
