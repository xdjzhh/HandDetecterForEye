"""
server.py  —— 极简 Flask 壳，按用户「三步拆解」严格实现，零复杂线程逻辑
========================================================================

三步：
  Step 1：启动时自动加载模型，验证参数（不打开摄像头！）
  Step 2：POST /run 接口 → 启动后台线程调用 hands_tasks.run_loop（打开摄像头+识别手势）
  Step 3：Step 2 的全部逻辑主体已经在 hands_tasks.py 里（本文件只是薄壳）

本文件刻意做得"简陋"：
  · 全局就 3 个变量：_run_thread / _stop_event / _status_store
  · 完全没有自定义类（没有什么 GestureServiceThread / PreviewThread）
  · 没有独立帧存储、没有复杂锁（status_store 是普通 dict，单写多读=GIL 安全）
  · Flask 就是普通 Flask，跑在主线程上（因为 imshow 会跑在另一个线程，两者互不影响）
"""

import os
import json
import time
import threading

from flask import Flask, jsonify, request

import hands_tasks as HT

# =========================================================================
# ① Step 1：启动时加载模型（不打开摄像头！）
# =========================================================================
print("\n" + "=" * 70)
print("[server.py Step 1] 启动时加载模型、校验参数（不打开摄像头）")
print("=" * 70)

# 如果同目录有 config.json，读它作为默认 camera_id（不传参数时用）
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
_default_camera_id = 0
_flask_host = "127.0.0.1"
_flask_port = 8000
_preview_default = True
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _default_camera_id = int(cfg.get("default_camera_id", cfg.get("camera_id", 0)))
        print(f"[server.py] 从 config.json 读到默认 camera_id = {_default_camera_id}")
        # 读取 Flask host / port
        try: _flask_host = str(cfg.get("flask_host", _flask_host)).strip() or "127.0.0.1"
        except Exception: pass
        try: _flask_port = max(1, min(65535, int(cfg.get("flask_port", _flask_port))))
        except Exception: pass
        # 读取 preview_default（/run 接口不传 preview 时的默认行为）
        try:
            _pv = cfg.get("preview_default", True)
            _preview_default = bool(_pv) if isinstance(_pv, bool) else (str(_pv) not in ("0", "false", "False", "no", "None"))
        except Exception:
            pass
        print(f"[server.py] Flask 监听: {_flask_host}:{_flask_port}  preview_default={_preview_default}")
    except Exception as e:
        print(f"[server.py] 读取 config.json 失败，忽略（默认 0/8000）：{e}")

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
    stretch_mode="IDLE", error=None, preview_active=False,
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

        # ----- 解析参数：camera_id / preview -----
        camera_id = _default_camera_id
        try:
            raw = request.args.get("camera_id", str(_default_camera_id))
            camera_id = max(0, min(9, int(raw)))
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
            stretch_mode="IDLE", error=None, preview_active=False,
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
# 首页（给浏览器看一眼有啥接口）
# =========================================================================
@app.route("/", methods=["GET"])
def index():
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>HandDetecter 极简服务</title></head>
<body style="font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:24px">
<h2>✅ HandDetecter 极简服务已启动（Flask）</h2>
<p>Step 1 完成：模型已加载 & 参数已校验，<b>尚未打开摄像头</b>。</p>
<p>当前生效配置：默认摄像头ID = <b id="cid">{_default_camera_id}</b> | preview_default = <b id="pd">{_preview_default}</b> | 监听 = <b>{_flask_host}:{_flask_port}</b></p>
<ul>
  <li><b>POST /run?camera_id={_default_camera_id}&preview=1</b> → 启动识别（打开摄像头）</li>
  <li><b>POST /stop</b> → 停止识别（关闭摄像头+预览窗口）</li>
  <li><b>GET  /status</b> → 实时状态（FPS、手势、stretch模式、错误等）</li>
</ul>
<div id="st" style="background:#f5f5f5;padding:12px;border-radius:6px;white-space:pre-wrap;font-family:Consolas,monospace;font-size:13px">点击「刷新状态」查看</div>
<br>
<button onclick="r()">刷新状态</button>
&nbsp;
<button onclick="runCam({_default_camera_id}, true)">启动摄像头{_default_camera_id}（带预览）</button>
<button onclick="runCam({_default_camera_id}, false)">启动摄像头{_default_camera_id}（无预览）</button>
&nbsp;
<button onclick="stop()">停止</button>
<script>
async function j(url, method='GET'){{
  return (await fetch(url,{{method}})).json();
}}
async function r(){{ document.getElementById('st').innerText = JSON.stringify(await j('/status'),null,2); }}
async function runCam(cid, preview){{
  let p = preview ? '1' : '0';
  alert(JSON.stringify(await j('/run?camera_id='+cid+'&preview='+p,'POST'),null,2));
  setTimeout(r,400);
}}
async function stop(){{ alert(JSON.stringify(await j('/stop','POST'),null,2)); setTimeout(r,400); }}
r();
</script>
</body></html>"""
    return html, 200


# =========================================================================
# 入口：python server.py  →  跑 Flask（主线程），默认端口 8000
# =========================================================================
if __name__ == "__main__":
    # threaded=True 让 Flask 可以并发处理 /status 和 /stop 请求（虽然就这几个接口，threaded=False 也没事）
    # use_reloader=False → 不启动子进程重新 import（否则会 init_models 两次，浪费时间+容易出问题）
    print("[server.py] ✅ Flask 启动：http://{}:{}/  （Ctrl+C 退出）\n".format(_flask_host, _flask_port))
    app.run(host=_flask_host, port=_flask_port, debug=False, threaded=False, use_reloader=False)
