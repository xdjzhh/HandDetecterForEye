"""
server.py  —— 极简 Flask 壳，按用户「三步拆解」严格实现，零复杂线程逻辑
========================================================================

三步：
  Step 1：启动时自动加载模型，验证参数（不打开摄像头！）
  Step 2：POST /run 接口 → 启动后台线程调用 hands_tasks.run_loop（打开摄像头+识别手势）
  Step 3：Step 2 的全部逻辑主体已经在 hands_tasks.py 里（本文件只是薄壳）

  Step 4：服务就绪后自动以【无预览】方式调一次 /run（后台静默识别，不弹窗）；
          若要预览窗口，在首页先点「停止」再点「启动（带预览）」自行切换。

本文件刻意做得"简陋"：
  · 全局就 3 个变量：_run_thread / _stop_event / _status_store
  · 完全没有自定义类（没有什么 GestureServiceThread / PreviewThread）
  · 没有独立帧存储、没有复杂锁（status_store 是普通 dict，单写多读=GIL 安全）
  · Flask 就是普通 Flask，跑在主线程上（因为 imshow 会跑在另一个线程，两者互不影响）
"""

import os
import sys
import json
import time
import socket
import threading
import argparse
import urllib.request
import urllib.parse

# =========================================================================
# ⓪ 日志系统必须【最早】安装：先读 config.json 的 log_enabled，再 import 业务模块
#   （cv2/MediaPipe/Flask/hands_tasks 的启动日志也要被镜像到 logs/ 文件）
# =========================================================================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH_EARLY = os.path.join(_PROJECT_ROOT, "config.json")
_log_enabled = True
if os.path.exists(_CONFIG_PATH_EARLY):
    try:
        with open(_CONFIG_PATH_EARLY, "r", encoding="utf-8") as _f:
            _cfg_early = json.load(_f)
        _v = _cfg_early.get("log_enabled", True)
        _log_enabled = bool(_v) if isinstance(_v, bool) else (str(_v) not in ("0", "false", "False", "no", "None"))
    except Exception:
        _log_enabled = True
import app_log
app_log.setup_logging(enabled=_log_enabled, project_root=_PROJECT_ROOT)

# =========================================================================
# ⓪.5 服务级单例锁（跨平台文件锁：msvcrt 区域锁 / fcntl.flock）：锁在服务
#   身份上，与端口/参数无关
#   （换端口启动同样被拦）。必须在 hands_tasks/模型加载之前。进程存活期间
#   持有；正常退出或 taskkill 强杀后由 OS 自动释放——无残留死锁，无需
#   PID 活性检查与 atexit 清理。句柄引用保留在模块级（丢引用=锁释放）。
# =========================================================================
def _acquire_singleton_lock(lock_name: str):
    lock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"{lock_name}.lock")
    fp = open(lock_path, "a+", encoding="utf-8")
    fp.write(" ")          # 字节 0 = 锁区占位；PID 写在其后（冲突方可读）
    fp.flush()
    try:
        fp.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)   # Windows：区域锁
        else:
            import fcntl
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # Linux/macOS：整文件锁
    except OSError:
        old_pid = ""
        try:
            fp.seek(1)     # 跳过锁区（字节 0 被独占，读不到）
            old_pid = fp.read().strip()
        except Exception:
            pass
        print(f"[singleton] 已有实例在运行{f'（PID={old_pid}）' if old_pid else ''}，"
              f"锁文件={lock_path}，本次启动退出。", flush=True)
        sys.exit(0)
    try:
        fp.seek(0)
        fp.truncate()
        fp.write(" ")          # 字节 0 = 锁区占位
        fp.write(str(os.getpid()))  # PID 写在锁区外，冲突方可读
        fp.flush()
    except Exception:
        pass
    # fp 故意不关闭：进程存续期间持锁，退出后由 OS 释放
    return fp


_SINGLETON_LOCK_FP = _acquire_singleton_lock("handdetecter")

from flask import Flask, jsonify, request

import hands_tasks as HT

# =========================================================================
# ① Step 1：启动时加载模型（不打开摄像头！）
# =========================================================================
print("\n" + "=" * 70)
print("[server.py Step 1] 启动时加载模型、校验参数（不打开摄像头）")
print("=" * 70)

# =========================================================================
# 启动参数：命令行显式 > start_proporties.yaml > 内置默认（与 Open-AutoGLM
#   server.py 同一模式）。config.json 只管摄像头/预览等业务字段，host/port
#   迁入 YAML。
# =========================================================================
_START_PROPERTIES_NAME = "start_proporties.yaml"


def _load_start_properties(config_path=None):
    """读取启动参数 YAML，返回 dict。
    文件缺失/解析失败只告警并返回 {}（回退内置默认值），绝不阻断启动。
    """
    path = config_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), _START_PROPERTIES_NAME)
    if not os.path.isfile(path):
        print(f"[server.py] ℹ️ 未找到启动参数文件 {path}，使用命令行/内置默认值。")
        return {}
    try:
        import yaml
    except ImportError:
        print("[server.py] ⚠️ 未安装 PyYAML，无法读取 "
              f"{_START_PROPERTIES_NAME}（pip install pyyaml）；回退命令行/内置默认值。")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[server.py] ⚠️ 解析启动参数文件 {path} 失败，回退命令行/内置默认值：{e!r}")
        return {}
    if not isinstance(data, dict):
        print(f"[server.py] ⚠️ 启动参数文件 {path} 顶层不是键值映射，已忽略。")
        return {}
    print(f"[server.py] ⚙️ 已加载启动参数文件：{path}（命令行同名参数优先）")
    return data


def set_cpu_affinity(cpu_list_str) -> bool:
    """设置当前进程的 CPU 亲和性（Windows: SetProcessAffinityMask；Linux/macOS: sched_setaffinity）

    核心不足时自动回退：若配置的核心全部超出范围，自动取前 N 个可用核心
    （N = 请求的核心数量，但不超过总核心数），保证 CPU 限制始终生效。

    Args:
        cpu_list_str: 逻辑核心编号逗号分隔字符串，如 "0,1" 绑定到核心0和1
    Returns:
        是否设置成功
    """
    if not cpu_list_str:
        return False
    try:
        cores = [int(c.strip()) for c in str(cpu_list_str).split(",") if c.strip() != ""]
    except ValueError:
        print(f"[server.py] ⚠️ CPU 亲和性参数格式错误: {cpu_list_str}，应为如 '0,1'")
        return False

    total = os.cpu_count() or 1
    requested = cores
    cores = [c for c in cores if 0 <= c < total]

    if not cores:
        n = min(len(requested), total)
        cores = list(range(n))
        print(f"[server.py] ⚠️ CPU 亲和性核心 {requested} 超出范围 (0-{total-1})，"
              f"自动回退到前 {n} 个核心: {cores}")
    elif len(cores) < len(requested):
        ignored = [c for c in requested if c not in cores]
        print(f"[server.py] ⚠️ CPU 亲和性核心 {ignored} 超出范围 (0-{total-1}) 已忽略，"
              f"实际绑定: {cores}")

    mask = 0
    for c in cores:
        mask |= (1 << c)

    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            PROCESS_SET_INFORMATION = 0x0200
            PROCESS_QUERY_INFORMATION = 0x0400
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
            kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(
                PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION,
                False,
                os.getpid(),
            )
            if not handle:
                err = ctypes.get_last_error()
                print(f"[server.py] ⚠️ OpenProcess 失败，错误码: {err}")
                return False
            try:
                result = kernel32.SetProcessAffinityMask(handle, mask)
                if result:
                    print(f"[server.py] ✅ CPU 亲和性已设置: 核心 {cores} (mask=0x{mask:X})")
                    return True
                else:
                    err = ctypes.get_last_error()
                    print(f"[server.py] ⚠️ SetProcessAffinityMask 失败，错误码: {err}")
                    return False
            finally:
                kernel32.CloseHandle(handle)
        else:
            os.sched_setaffinity(0, cores)
            print(f"[server.py] ✅ CPU 亲和性已设置: 核心 {cores}")
            return True
    except Exception as e:
        print(f"[server.py] ⚠️ 设置 CPU 亲和性失败: {e}")
        return False


# default=None 表示「命令行未传」：未传回退 YAML，YAML 缺项再回退内置默认
_server_argp = argparse.ArgumentParser(description="HandDetecter 手势识别服务")
_server_argp.add_argument("--config", default=None,
                          help=f"启动参数 YAML 路径（默认=server.py 同目录 {_START_PROPERTIES_NAME}）")
_server_argp.add_argument("--host", default=None,
                          help=f"Flask 监听地址（默认读 YAML/内置: 127.0.0.1）")
_server_argp.add_argument("--port", type=int, default=None,
                          help=f"Flask 监听端口（默认读 YAML/内置: 8000）")
_server_argp.add_argument("--cpu-affinity", type=str, default=None,
                          help="CPU 亲和性，绑定到指定逻辑核心，如 '0,1'；不设则由系统调度")
_server_cli = _server_argp.parse_args()
_server_props = _load_start_properties(_server_cli.config)


def _pick_server_prop(cli_val, yaml_key, builtin_default):
    """命令行显式值优先；否则取 YAML；YAML 无此项再用内置默认。"""
    if cli_val is not None:
        return cli_val
    return _server_props.get(yaml_key, builtin_default)


# Flask 监听地址（三层合并结果；下面的单实例锁与 make_server 都用它）
_flask_host = str(_pick_server_prop(_server_cli.host, "host", "127.0.0.1"))
try:
    _flask_port = max(1, min(65535, int(_pick_server_prop(_server_cli.port, "port", 8000))))
except Exception:
    _flask_port = 8000
print(f"[server.py] Flask 监听: {_flask_host}:{_flask_port}（来源：命令行 > {_START_PROPERTIES_NAME} > 内置默认）")

# CPU 亲和性（命令行 > YAML > 不设置）
_cpu_affinity = _pick_server_prop(_server_cli.cpu_affinity, "cpu_affinity", None)
if _cpu_affinity is not None:
    set_cpu_affinity(str(_cpu_affinity))

# config.json 只读业务字段（摄像头/预览；host/port 已迁 start_proporties.yaml）
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
_default_camera_id = 0
_preview_default = True
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _raw_cid = cfg.get("default_camera_id", cfg.get("camera_id", 0))
        try:
            _default_camera_id = int(_raw_cid)
        except (TypeError, ValueError):
            # 兼容：config 里写字符串时按相机友好名字片段匹配（MF 实时清单）；正式配置推荐用数字 index（见 /cameras）
            _default_camera_id = str(_raw_cid).strip()
        print(f"[server.py] 从 config.json 读到默认 camera_id = {_default_camera_id!r}")
        # 读取 preview_default（/run 接口不传 preview 时的默认行为）
        try:
            _pv = cfg.get("preview_default", True)
            _preview_default = bool(_pv) if isinstance(_pv, bool) else (str(_pv) not in ("0", "false", "False", "no", "None"))
        except Exception:
            pass
        print(f"[server.py] preview_default={_preview_default}")
    except Exception as e:
        print(f"[server.py] 读取 config.json 失败，忽略（默认 camera_id=0）：{e}")

# =========================================================================
# ⓪.5 单实例启动锁（2026-10-04）
#   病根：Windows 的 SO_REUSEADDR（Werkzeug 默认会设）允许【不同进程】同时 bind+listen
#   同一端口，入站连接被随机分发。于是重复 `python server.py` 会留下多个僵尸实例：
#   旧进程一直用旧内存代码处理 /run、/stop，新代码永远不生效（当天日志实锤 3 个
#   server.py 同听 8000，新埋点会话全被旧进程接走）。
#   两道防线：
#     1) 启动先 HTTP 探活 /status：已有本服务（JSON 含 service_started）→ 直接退出；
#     2) 进程生命期持有 SO_EXCLUSIVEADDRUSE 独占监听 socket 作 OS 级兜底：
#        即使两个进程同时启动、探活双双落空，后绑者也会 WSAEADDRINUSE 失败。
# =========================================================================
def _probe_existing_server(host: str, port: int):
    """None=端口空闲可启动；'ours'=本服务已在跑；'other-http'=端口被别的程序占用。"""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        if probe.connect_ex((probe_host, port)) != 0:
            return None  # 没人监听 = 端口空闲
    finally:
        probe.close()
    # 端口已有人监听：分清"本服务旧实例"还是"其他程序"（禁走系统代理）。
    # 用首页固定标题做特征：/status 运行中会被重填（字段不稳定）且响应可能数十 KB，
    # 不能读截断的半截 JSON 去解析。
    try:
        _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with _opener.open(f"http://{probe_host}:{port}/", timeout=1.5) as resp:
            body = resp.read(2048)
        if b"HandDetecter" in body:
            return "ours"
        return "other-http"
    except Exception:
        return "other-http"


def _acquire_single_instance_lock(host: str, port: int) -> socket.socket:
    """持住独占监听 socket 直到进程退出；发现已有实例/端口被占则直接退出进程。"""
    reason = _probe_existing_server(host, port)
    if reason == "ours":
        print("\n" + "!" * 70)
        print(f"[server.py] ❌ 已有一个本服务在 http://{host}:{port}/ 运行——单实例锁拦截，本次不重复启动。")
        print("    /stop 只停摄像头线程、不会退出 Flask 进程；重复启动会留下僵尸实例，")
        print("    旧实例继续用旧代码处理 /run、/stop，新代码永远不生效。")
        print("    → 请直接用浏览器打开现有页面操作；确需重启：原窗口 Ctrl+C")
        print("      （或任务管理器结束旧 python.exe）后再启动。")
        print("!" * 70 + "\n")
        sys.exit(2)
    if reason is not None:
        print(f"\n[server.py] ❌ 端口 {host}:{port} 已被【其他程序】占用（不是本服务）。"
              f"请退出占用程序，或改 {_START_PROPERTIES_NAME} 的 port 后重试。\n")
        sys.exit(3)
    # OS 级兜底：独占绑定，持有到进程退出
    lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            # Windows：EXCLUSIVE 拒绝任何后来者（哪怕对方设了 REUSEADDR）抢占
            lock_sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # POSIX：REUSEADDR 只允许复用 TIME_WAIT，不允许复用活动监听，等价安全
            lock_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lock_sock.bind((host, port))
        lock_sock.listen(128)
    except OSError as e:
        print(f"\n[server.py] ❌ 独占绑定 {host}:{port} 失败（可能另一实例正在抢占启动）：{e}。"
              f"请确认没有重复 server.py 后重试。\n")
        lock_sock.close()
        sys.exit(4)
    print(f"[server.py] 🔒 单实例锁已持有 {host}:{port}（独占监听，进程退出自动释放）")
    return lock_sock


_single_instance_sock = _acquire_single_instance_lock(_flask_host, _flask_port)

# 解析参数 + clamp（不调 main.py），从 config.json 读 server_argv，没有就用默认
_server_argv = ["--use-tasks", "--roi-scale", "0.70"]
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg.get("server_argv"), list) and len(cfg["server_argv"]) > 0:
            _server_argv = list(cfg["server_argv"])
            print(f"[server.py] 从 config.json 读到 server_argv = {_server_argv}")
    except Exception as e:
        print(f"[server.py] 读取 config.json server_argv 失败，用默认：{e}")
_args = HT.parse_args(_server_argv)
HT.clamp_args(_args)
print(f"[server.py] 参数解析 ok: use_tasks={_args.use_tasks} roi={_args.roi_scale} roi_offset_y={_args.roi_offset_y} "
      f"anchor={_args.anchor} cursor_mode={_args.cursor_mode} "
      f"hardness={_args.tap_hardness} speed={_args.tap_speed}")

# 🔥 关键：启动时就 init_models 一次（= 加载 MediaPipe 模型 + 验证参数 + 创建所有组件）
#   · 这一步只做"预热加载"，绝对不打开摄像头
#   · 之后每次 /run 会再调一次 init_models（因为 detector.close() 后 MediaPipe 不让复用，
#     所以每次 /run 用全新实例最稳）
_prewarm_components = HT.init_models(_args)
_prewarm_detector_ok = _prewarm_components["detector"] is not None
print(f"[server.py Step 1 完成] 模型加载 ok={_prewarm_detector_ok}（未打开摄像头，等待 /run 接口）\n")


# =========================================================================
# 全局共享状态（刻意极简：就 3 个变量，没有任何自定义类）
# =========================================================================
app = Flask(__name__)

_run_thread: threading.Thread | None = None
_stop_event = threading.Event()        # 给 /stop 调 set()；run_loop 每帧开头检查
_status_lock = threading.Lock()        # 只在「启动/停止」这种边界操作用，读 status_store 不用锁
_status_store: dict = {}

# 初始 status（=服务启动后还没调 /run 的状态）
_status_store.update(dict(
    running=False, fps=0.0, current_camera_id=_default_camera_id,
    start_time_ms=None, uptime_ms=0,
    last_left_gesture=None, last_right_gesture=None,
    move_mode="IDLE",
    error=None, preview_active=False,
    service_started=True,
    # 把当前生效的配置也暴露出来，前端/排障一眼就能看到
    config_path=CONFIG_PATH,
    default_camera_id_now=_default_camera_id,
    preview_default=bool(_preview_default),
    flask_host=_flask_host,
    flask_port=_flask_port,
))


# =========================================================================
# ② Step 2：POST /run  —— 启动后台线程打开摄像头并识别手势（同时仅 1 个）
# =========================================================================
@app.route("/run", methods=["POST"])
def run():
    """Step 2 接口：启动后台线程调用 hands_tasks.run_loop"""
    global _run_thread, _stop_event, _status_store
    with _status_lock:
        # ----- 防重入 -----
        if _run_thread is not None and _run_thread.is_alive():
            return jsonify({"detail": "已经在运行中，先调用 /stop 再启动"}), 400

        # ----- 解析参数：camera_id / preview（支持 int 索引 或 str 相机名字片段） -----
        camera_id = _default_camera_id
        try:
            raw = request.args.get("camera_id", str(_default_camera_id))
            try:
                camera_id = max(0, min(9, int(raw)))
            except (TypeError, ValueError):
                # 非数字 = 按相机友好名字片段打开（MF 实时清单按名匹配，与 Windows 设置同源）
                camera_id = raw.strip() if isinstance(raw, str) and raw.strip() else _default_camera_id
        except Exception:
            camera_id = _default_camera_id
        # 🔥 preview 默认值用 config.json 的 preview_default（而不是写死 True）
        preview = bool(_preview_default)
        try:
            raw = request.args.get("preview", None)
            if raw is None:
                preview = bool(_preview_default)
            else:
                preview = (str(raw) not in ("0", "false", "False", "no"))
        except Exception:
            preview = bool(_preview_default)

        # ----- 重置信号量 + 状态 -----
        _stop_event.clear()
        _status_store.clear()
        _status_store.update(dict(
            running=False, fps=0.0, current_camera_id=camera_id,
            start_time_ms=None, uptime_ms=0,
            last_left_gesture=None, last_right_gesture=None,
            move_mode="IDLE",
            error=None, preview_active=False,
        ))

        # 🔥 每次 /run 都用**全新组件**（因为 detector.close() 后不能再用；
        #    最简单的办法就是重新 init_models()，跟原 main.py 行为完全一致）
        print(f"\n[server.py /run] ✨ 重新 init_models（为本次 /run 生成全新 detector/mouse 组件）")
        fresh_components = HT.init_models(_args)

        print(f"[server.py /run] 启动后台线程：camera_id={camera_id}  preview={preview}")
        _run_thread = threading.Thread(
            target=HT.run_loop,
            name=f"HandsLoop-{int(time.time())}",
            args=(fresh_components, camera_id, _stop_event, bool(preview), _status_store),
            daemon=True,
        )
        _run_thread.start()
        # 等 120ms 让线程内设置 running=True 或者写出 error
        time.sleep(0.12)
        return jsonify({
            "ok": True,
            "camera_id": camera_id,
            "preview": bool(preview),
            "thread": _run_thread.name,
            "running_now": _status_store.get("running"),
            "error_init": _status_store.get("error"),
        })


# =========================================================================
# ③ POST /stop  —— 通过 stop_event.set() 让 run_loop 下一帧干净退出
# =========================================================================
@app.route("/stop", methods=["POST"])
def stop():
    global _run_thread, _stop_event
    with _status_lock:
        if _run_thread is None or not _run_thread.is_alive():
            return jsonify({"ok": True, "detail": "当前未在运行", "running_now": False})

        print(f"\n[server.py /stop] ✨ 触发 stop_event，通知 run_loop 下一帧立即退出…")
        _stop_event.set()
        # 最多等线程自行退出 2.5 秒（足够跑几帧 finally 做 detector.close / cap.release / destroyAllWindows）
        _run_thread.join(timeout=2.5)
        still_alive = _run_thread.is_alive()
        print(f"[server.py /stop] 线程退出 ok={not still_alive}  最终 running={_status_store.get('running')}\n")
        return jsonify({
            "ok": True,
            "thread_joined": not still_alive,
            "running_now": _status_store.get("running"),
            "error": _status_store.get("error"),
        })


# =========================================================================
# ③.5 GET /cameras —— 枚举系统相机清单（MF 统一索引 + DSHOW 参考），供选择摄像头
#       对标 Windows 设置：MF 实时枚举为唯一索引源，按设备身份选择
# =========================================================================
@app.route("/cameras", methods=["GET"])
def list_cameras():
    """相机清单：
    mf = Media Foundation 实时枚举——**统一索引源**（与 Windows 设置 / OpenCV MSMF 同源；
    config.json 的 default_camera_id 和 /run?camera_id= 都用这里的 index 或名字）；
    dshow = DirectShow 过滤器表，仅在 MSMF 失败时按 VID/PID 身份映射做备选后端，
    其 index 与 MF 不一致且重插后可能残留陈旧项——⚠ 不要用 dshow 的 index 配置。"""
    try:
        mf = HT._list_mf_cameras()
    except Exception as e:
        mf, mf_err = [], repr(e)
    else:
        mf_err = None
    try:
        ds = HT._list_dshow_cameras()
    except Exception as e:
        ds, ds_err = [], repr(e)
    else:
        ds_err = None
    return jsonify({
        "ok": True,
        "canonical_index": "mf",  # 统一索引源：配置/接口里的 camera_id 一律以 mf_cameras 的 index 为准
        "count": len(mf),
        "cameras": [{"index": i, "name": n, "device_path": p} for i, n, p in mf],
        "mf_cameras": [{"index": i, "name": n, "device_path": p} for i, n, p in mf],
        "dshow_cameras": [{"index": i, "name": n, "device_path": p} for i, n, p in ds],
        "mf_error": mf_err, "dshow_error": ds_err,
        "default_camera_id": _default_camera_id,
    })


# =========================================================================
# ④ GET /status  —— 直接返回 status_store（简单 dict，不加锁）
# =========================================================================
@app.route("/status", methods=["GET"])
def status():
    """
    直接把 status_store 吐给前端。
    字段含义见 hands_tasks.run_loop 函数的 docstring。
    """
    # 附加上：线程是否存活
    alive = bool(_run_thread is not None and _run_thread.is_alive())
    out = dict(_status_store)
    out["thread_alive"] = alive
    out["default_camera_id"] = _default_camera_id
    return jsonify(out)


# =========================================================================
# 优雅停止（virtualdesk stop 调用）：回 200 后延迟 httpd.shutdown()——进程内
# 标志，无控制台进程（CREATE_NO_WINDOW）也可靠送达。CTRL_C 事件在无控制台
# 进程里投递不到 handler（GenerateConsoleCtrlEvent 成空操作），信号方案弃用。
# serve_forever 返回后进程自然退出（识别线程是 daemon，随进程收尾）。
# shutdown 失败不强退，交由 virtualdesk taskkill 兜底。
# =========================================================================
@app.route("/exit", methods=["POST"])
def graceful_exit():
    import threading

    def _self_shutdown():
        time.sleep(0.5)
        print("\n[exit] 收到 /exit 请求：httpd.shutdown()，serve_forever 收尾退出...", flush=True)
        try:
            _httpd.shutdown()
        except Exception as e:
            print(f"[exit] ⚠️ shutdown 失败（由外部 taskkill 兜底）：{e}", flush=True)

    threading.Thread(target=_self_shutdown, daemon=True).start()
    return jsonify({"success": True, "message": "exiting via httpd.shutdown()"})


# =========================================================================
# 首页（给浏览器看一眼有啥接口）
# =========================================================================
@app.route("/", methods=["GET"])
def index():
    # camera_id 可能是 int 或 str（名字片段），注入 JS 前转成 JS 字面量（字符串带引号）。
    # 注意：onclick="..." 属性本身用双引号定界，字面量里的双引号必须转成 &quot; 实体，
    # 否则渲染出 onclick="runCam("HD",...)" 会在属性边界截断 → 按钮点击 JS 语法错误无反应。
    _cid_js = json.dumps(_default_camera_id, ensure_ascii=False).replace('"', "&quot;")
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>HandDetecter 极简服务</title></head>
<body style="font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:24px">
<h2>✅ HandDetecter 极简服务已启动（Flask）</h2>
<p>Step 1 完成：模型已加载 & 参数已校验，<b>尚未打开摄像头</b>。</p>
<p>当前生效配置：默认摄像头ID = <b id="cid">{_default_camera_id}</b> | preview_default = <b id="pd">{_preview_default}</b> | 监听 = <b>{_flask_host}:{_flask_port}</b></p>
<ul>
  <li><b>POST /run?camera_id={_default_camera_id}&preview=1</b> → 启动识别（打开摄像头）</li>
  <li><b>POST /stop</b> → 停止识别（关闭摄像头+预览窗口）</li>
  <li><b>GET  /status</b> → 实时状态（FPS、手势、错误等）</li>
  <li><b>GET  /cameras</b> → 枚举系统相机清单（index + 友好名，选摄像头前先点它）</li>
</ul>
<p>ℹ️ 服务启动后会<b>自动以无预览方式后台运行</b>；如需预览窗口，请先点「停止」，再点「启动（带预览）」。</p>
<div id="st" style="background:#f5f5f5;padding:12px;border-radius:6px;white-space:pre-wrap;font-family:Consolas,monospace;font-size:13px">点击「刷新状态」查看</div>
<br>
<button onclick="r()">刷新状态</button>
&nbsp;
<button onclick="listCams()">列出摄像头（按名字选，不靠猜索引）</button>
<br><br>
<button onclick="runCam({_cid_js}, true)">启动摄像头{_default_camera_id}（带预览）</button>
<button onclick="runCam({_cid_js}, false)">启动摄像头{_default_camera_id}（无预览）</button>
&nbsp;
<button onclick="stop()">停止</button>
<script>
async function j(url, method='GET'){{
  return (await fetch(url,{{method}})).json();
}}
async function r(){{ document.getElementById('st').innerText = JSON.stringify(await j('/status'),null,2); }}
async function listCams(){{
  let d = await j('/cameras');
  let msg = '【相机清单 MF 实时枚举（= Windows 设置/OpenCV MSMF 同源）】共 ' + d.count + ' 台，\\n' +
    '配置 config.json 的 default_camera_id 一律用这里的 index：\\n' +
    (d.mf_cameras||[]).map(c => '  index=' + c.index + '  名字=' + c.name).join('\\n') +
    '\\n\\n（内部参考，⚠ 不要用这里的 index 配置）DSHOW 过滤器表 ' + (d.dshow_cameras||[]).length + ' 台：\\n' +
    (d.dshow_cameras||[]).map(c => '  dshow_index=' + c.index + '  名字=' + c.name).join('\\n') +
    '\\n\\n当前 config 默认 camera_id=' + d.default_camera_id +
    '；要用某台相机，把 config.json 的 default_camera_id 改成上面 MF 清单中该相机的 index 后重启服务。';
  alert(msg);
}}
async function runCam(cid, preview){{
  let p = preview ? '1' : '0';
  alert(JSON.stringify(await j('/run?camera_id='+encodeURIComponent(cid)+'&preview='+p,'POST'),null,2));
  setTimeout(r,400);
}}
async function stop(){{ alert(JSON.stringify(await j('/stop','POST'),null,2)); setTimeout(r,400); }}
r();
</script>
</body></html>"""
    return html, 200


# =========================================================================
# ④.5 启动即【无预览】自动运行：服务就绪后自己 POST 一次 /run?preview=0
#   · 默认形态 = 后台静默识别，不弹 cv2 预览窗口
#   · 要预览窗口：首页点「停止」再点「启动（带预览）」（按钮与接口均未改动）
#   · 走真实 HTTP 接口（而非直接调视图函数），与页面点按钮完全同链路；
#     失败只记日志、不影响 Flask，仍可在页面手动启动
# =========================================================================
def _autostart_no_preview():
    probe_host = "127.0.0.1" if _flask_host in ("0.0.0.0", "::", "") else _flask_host
    base = f"http://{probe_host}:{_flask_port}"
    cid = urllib.parse.quote(str(_default_camera_id), safe="")
    _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本机自调，绕过系统代理
    # 独占锁 socket 先于 Flask 完成 listen，光探端口会误判就绪 → 轮询 /status 直到返回 200
    ready, err = False, None
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            with _opener.open(base + "/status", timeout=1) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception as e:
            err = e
            time.sleep(0.3)
    if not ready:
        print(f"[server.py 自动启动] ⚠ 等待服务就绪超时，跳过自动无预览启动（可在页面手动启动）：{err}")
        return
    try:
        req = urllib.request.Request(base + f"/run?camera_id={cid}&preview=0", method="POST")
        with _opener.open(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", "replace")
        print(f"[server.py 自动启动] ✅ 已自动调用无预览接口 POST /run?camera_id={cid}&preview=0，响应：{body}")
        print("[server.py 自动启动] 识别已在后台静默运行（无预览窗口）；如需预览窗口，页面先「停止」再「启动（带预览）」。")
    except Exception as e:
        print(f"[server.py 自动启动] ⚠ 自动无预览 /run 失败（不影响服务，可在页面手动启动）：{e}")


# =========================================================================
# 入口：python server.py  →  跑 Flask（主线程），默认端口 8000
# =========================================================================
if __name__ == "__main__":
    # threaded=False 与历史行为一致（/status、/stop 串行处理，接口极少无所谓）
    # use_reloader=False → 不启动子进程重新 import（否则会 init_models 两次）
    #
    # ⭐ 不用 app.run：Werkzeug 自建 socket 会设 SO_REUSEADDR，在 Windows 上挡不住跨进程
    #   重复绑定（僵尸实例病根）。改为 make_server 先绑一个临时端口，再把服务 socket
    #   换成单实例锁持有的 SO_EXCLUSIVEADDRUSE 监听 socket（同一进程内替换，无竞态窗口）。
    from werkzeug.serving import make_server
    print("[server.py] ✅ Flask 启动：http://{}:{}/  （Ctrl+C 退出）\n".format(_flask_host, _flask_port))
    _httpd = make_server(_flask_host, 0, app, threaded=False)
    _httpd.socket.close()          # 释放 Werkzeug 的临时 REUSEADDR socket
    _httpd.socket = _single_instance_sock   # 挂上独占监听 socket（已 bind+listen）
    # 🔥 启动即无预览自动运行：后台静默识别；要预览窗口去页面先停止再带预览启动
    threading.Thread(target=_autostart_no_preview, name="AutoRunNoPreview", daemon=True).start()
    try:
        _httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server.py] 收到 Ctrl+C，退出（自动释放单实例锁）")
