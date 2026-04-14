"""Unit tests for the web/backend package."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web.backend import config as cfg_mod, process_manager, state as state_mod  # noqa: E402
from web.backend.log_buffer import LogBuffer, read_log_tail  # noqa: E402

try:
    from web.backend import main as main_mod  # noqa: E402
    UVICORN_OK = True
except Exception:  # pragma: no cover - optional outside the project venv
    main_mod = None  # type: ignore[assignment]
    UVICORN_OK = False

try:
    from fastapi.testclient import TestClient  # type: ignore
    from web.backend import auth  # noqa: E402
    from web.backend.routes import builds as builds_route  # noqa: E402
    from web.backend.routes import memory as memory_route  # noqa: E402
    FASTAPI_OK = True
except Exception:  # pragma: no cover - optional outside the project venv
    TestClient = None  # type: ignore[assignment]
    auth = None  # type: ignore[assignment]
    builds_route = None  # type: ignore[assignment]
    memory_route = None  # type: ignore[assignment]
    FASTAPI_OK = False


def _kill_pid(pid: int) -> None:
    with contextlib_suppress(Exception):
        os.kill(pid, signal.SIGTERM)


class contextlib_suppress:
    def __init__(self, *exceptions):
        self._exceptions = exceptions

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return exc_type is not None and issubclass(exc_type, self._exceptions)


def _write_fake_llama_server(base_dir: Path) -> Path:
    path = base_dir / "bin" / "llama-server"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "import time\n"
        'print("ready", flush=True)\n'
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


class ConfigTests(unittest.TestCase):
    def test_load_creates_token_when_missing(self):
        path = REPO_ROOT / "tests" / "_tmp_config.json"
        if path.exists():
            path.unlink()
        try:
            cfg = cfg_mod.load_config(path)
            self.assertTrue(cfg.token)
            self.assertTrue(path.exists())
            again = cfg_mod.load_config(path)
            self.assertEqual(cfg.token, again.token)
        finally:
            if path.exists():
                path.unlink()

    def test_init_config_force(self):
        path = REPO_ROOT / "tests" / "_tmp_init.json"
        if path.exists():
            path.unlink()
        try:
            first = cfg_mod.init_config(path, force=True)
            second = cfg_mod.init_config(path, force=True)
            self.assertNotEqual(first.token, second.token)
        finally:
            if path.exists():
                path.unlink()


@unittest.skipUnless(UVICORN_OK, "uvicorn not installed")
class MainTests(unittest.TestCase):
    def test_main_sets_relaxed_websocket_keepalive(self):
        cfg = cfg_mod.WebConfig(token="tok", host="0.0.0.0", port=8787)
        with mock.patch("web.backend.main.load_config", return_value=cfg), \
             mock.patch("web.backend.main.uvicorn.run") as run:
            rc = main_mod.main([])
        self.assertEqual(rc, 0)
        run.assert_called_once()
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["ws_ping_interval"], 45.0)
        self.assertEqual(kwargs["ws_ping_timeout"], 45.0)


@unittest.skipUnless(FASTAPI_OK, "fastapi not installed")
class AuthTests(unittest.TestCase):
    def test_compare_constant_time(self):
        cfg = cfg_mod.WebConfig(token="abc123")
        self.assertTrue(auth.verify_ws_token("abc123", cfg))
        self.assertFalse(auth.verify_ws_token("wrong", cfg))
        self.assertFalse(auth.verify_ws_token(None, cfg))
        self.assertFalse(auth.verify_ws_token("", cfg))


class LogBufferTests(unittest.TestCase):
    def test_emit_and_tail(self):
        log_file = REPO_ROOT / "tests" / "_tmp_log.log"
        if log_file.exists():
            log_file.unlink()
        try:
            buf = LogBuffer(capacity=10, log_file=log_file)
            buf.emit("hello")
            buf.emit("world")
            self.assertEqual([line.rstrip() for line in buf.snapshot()], ["hello", "world"])
            self.assertEqual([line.rstrip() for line in read_log_tail(log_file, 10)], ["hello", "world"])
        finally:
            if log_file.exists():
                log_file.unlink()

    def test_seed_from_file_rehydrates_history(self):
        log_file = REPO_ROOT / "tests" / "_tmp_seed.log"
        if log_file.exists():
            log_file.unlink()
        try:
            log_file.write_text("first\nsecond\n", encoding="utf-8")
            buf = LogBuffer(capacity=10, log_file=log_file)
            buf.seed_from_file()
            self.assertEqual([line.rstrip() for line in buf.snapshot()], ["first", "second"])
        finally:
            if log_file.exists():
                log_file.unlink()


class StateStoreTests(unittest.TestCase):
    def setUp(self):
        self.path = REPO_ROOT / "tests" / "_tmp_state.json"
        if self.path.exists():
            self.path.unlink()

    def tearDown(self):
        if self.path.exists():
            self.path.unlink()

    def test_upsert_and_delete(self):
        store = state_mod.StateStore(self.path)
        rec = state_mod.InstanceRecord(id="abc", name="one")
        store.upsert_instance(rec)
        self.assertIsNotNone(store.get_instance("abc"))
        raw = json.loads(self.path.read_text())
        self.assertEqual(len(raw["instances"]), 1)
        rec2 = state_mod.InstanceRecord(id="abc", name="two")
        store.upsert_instance(rec2)
        self.assertEqual(store.get_instance("abc").name, "two")
        self.assertTrue(store.delete_instance("abc"))
        self.assertIsNone(store.get_instance("abc"))


class ProcessManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.state_path = self.root / "state.json"
        self.cfg = cfg_mod.WebConfig(token="tok")
        self.store = state_mod.StateStore(self.state_path)
        self.pm = process_manager.ProcessManager(self.cfg, self.store)
        self._pids_to_cleanup: list[int] = []

    async def asyncTearDown(self):
        await self.pm.shutdown()
        for pid in self._pids_to_cleanup:
            _kill_pid(pid)
        self.tmpdir.cleanup()

    async def test_intentional_instance_stop_becomes_stopped(self):
        with mock.patch("loadmodel.resolve_gguf", return_value=Path("/tmp/fake.gguf")), \
             mock.patch("loadmodel.ensure_moe_flags_available", return_value=None), \
             mock.patch(
                 "web.backend.process_manager.build_llama_server_cmd",
                 return_value=[sys.executable, "-c", 'import time; print("ready", flush=True); time.sleep(30)'],
             ):
            inst = await self.pm.create_instance("test", {"model_ref": "/tmp/fake.gguf"}, auto_start=True)
            await asyncio.sleep(0.4)
            await self.pm.stop_instance(inst.record.id)
            data = self.pm.serialize_instance(inst.record.id)
            self.assertEqual(data["status"], "stopped")
            self.assertIsNone(data["pid"])
            logs = [line.rstrip() for line in self.pm.get_instance_logs(inst.record.id)]
            self.assertTrue(any("ready" in line for line in logs))

    def test_build_llama_server_cmd_auto_split_uses_selected_gpus(self):
        gpus = [
            process_manager.memory_utils.GPUInfo(index=0, name="GPU0", total=10, free=8),
            process_manager.memory_utils.GPUInfo(index=1, name="GPU1", total=20, free=16),
            process_manager.memory_utils.GPUInfo(index=2, name="GPU2", total=30, free=24),
        ]
        with mock.patch("web.backend.process_manager.memory_utils.detect_gpus", return_value=gpus):
            cmd = process_manager.build_llama_server_cmd(
                {
                    "model_ref": "/tmp/fake.gguf",
                    "gpu_devices": "1,2",
                    "tensor_split": "auto",
                    "auto_split_policy": "even",
                },
                Path("/tmp/fake.gguf"),
            )
        split_idx = cmd.index("--tensor-split")
        self.assertEqual(cmd[split_idx + 1], "50,50")

    async def test_start_instance_sets_cuda_visible_devices(self):
        seen: dict[str, str | None] = {}

        async def fake_spawn(cmd, *, log_file, cwd, env):
            seen["cuda_visible_devices"] = env.get("CUDA_VISIBLE_DEVICES")
            return await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
                start_new_session=True,
            )

        with mock.patch("loadmodel.resolve_gguf", return_value=Path("/tmp/fake.gguf")), \
             mock.patch("loadmodel.ensure_moe_flags_available", return_value=None), \
             mock.patch(
                 "web.backend.process_manager.memory_utils.detect_gpus",
                 return_value=[
                     process_manager.memory_utils.GPUInfo(index=0, name="GPU0", total=10, free=8),
                     process_manager.memory_utils.GPUInfo(index=1, name="GPU1", total=20, free=16),
                 ],
             ), \
             mock.patch("web.backend.process_manager._spawn_logged_process", side_effect=fake_spawn):
            inst = await self.pm.create_instance(
                "gpu-env",
                {"model_ref": "/tmp/fake.gguf", "gpu_devices": "0,1"},
                auto_start=True,
            )
            await asyncio.sleep(0.2)
            self.assertEqual(seen["cuda_visible_devices"], "0,1")
            await self.pm.stop_instance(inst.record.id)

    async def test_rehydrate_live_instance_is_controllable(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        self._pids_to_cleanup.append(proc.pid)
        rec = state_mod.InstanceRecord(
            id="rehydrated",
            name="rehydrated",
            config={"model_ref": "/tmp/fake.gguf"},
            pid=proc.pid,
            pgid=os.getpgid(proc.pid),
            status="running",
            started_at=time.time() - 5,
            cmdline=[sys.executable, "-c", "import time; time.sleep(30)"],
            log_file=str(self.root / "inst.log"),
        )
        self.store.upsert_instance(rec)
        await self.pm.startup()
        data = self.pm.serialize_instance("rehydrated")
        self.assertTrue(data["alive"])
        await self.pm.stop_instance("rehydrated")
        data = self.pm.serialize_instance("rehydrated")
        self.assertEqual(data["status"], "stopped")
        self.assertFalse(data["alive"])
        proc.wait(timeout=2)
        self.assertIsNotNone(proc.returncode)

    async def test_rehydrate_dead_running_instance_becomes_crashed(self):
        rec = state_mod.InstanceRecord(
            id="dead",
            name="dead",
            config={"model_ref": "/tmp/fake.gguf"},
            pid=999999,
            status="running",
            started_at=time.time() - 5,
            cmdline=[sys.executable, "-c", "import time; time.sleep(30)"],
            log_file=str(self.root / "dead.log"),
        )
        self.store.upsert_instance(rec)
        await self.pm.startup()
        data = self.pm.serialize_instance("dead")
        self.assertEqual(data["status"], "crashed")
        self.assertFalse(data["alive"])

    async def test_rehydrate_stopping_dead_instance_becomes_stopped(self):
        rec = state_mod.InstanceRecord(
            id="stopping",
            name="stopping",
            config={"model_ref": "/tmp/fake.gguf"},
            pid=999999,
            status="stopping",
            started_at=time.time() - 5,
            cmdline=[sys.executable, "-c", "import time; time.sleep(30)"],
            log_file=str(self.root / "stopping.log"),
        )
        self.store.upsert_instance(rec)
        await self.pm.startup()
        data = self.pm.serialize_instance("stopping")
        self.assertEqual(data["status"], "stopped")

    async def test_startup_recovers_orphan_repo_llama_server(self):
        fake_server = _write_fake_llama_server(self.root)
        log_file = self.root / "orphan.log"
        with log_file.open("a", encoding="utf-8") as sink:
            proc = subprocess.Popen(
                [
                    str(fake_server),
                    "--model",
                    "/tmp/fake.gguf",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "45566",
                    "--ctx-size",
                    "8192",
                    "--embeddings",
                    "--flash-attn",
                    "on",
                ],
                stdout=sink,
                stderr=sink,
                start_new_session=True,
            )
        self._pids_to_cleanup.append(proc.pid)
        await asyncio.sleep(0.4)

        with mock.patch.object(process_manager, "LLAMA_SERVER", fake_server):
            await self.pm.startup()

        instances = self.pm.list_instances()
        recovered = next(inst for inst in instances if inst["pid"] == proc.pid)
        self.assertTrue(recovered["alive"])
        self.assertEqual(recovered["status"], "running")
        self.assertEqual(recovered["host"], "0.0.0.0")
        self.assertEqual(recovered["port"], 45566)
        self.assertEqual(recovered["config"]["mode"], "embed")
        self.assertEqual(recovered["config"]["model_ref"], "/tmp/fake.gguf")
        self.assertEqual(recovered["log_file"], str(log_file))
        logs = [line.rstrip() for line in self.pm.get_instance_logs(recovered["id"])]
        self.assertTrue(any("ready" in line for line in logs))

        await self.pm.stop_instance(recovered["id"])
        proc.wait(timeout=2)
        self.assertIsNotNone(proc.returncode)

    async def test_running_build_without_identity_becomes_failure_on_startup(self):
        rec = state_mod.BuildRecord(
            id="build-no-id",
            config={"ref": "latest"},
            started_at=time.time() - 5,
            status="running",
            pid=12345,
            log_file=str(self.root / "build.log"),
        )
        self.store.upsert_build(rec)
        await self.pm.startup()
        data = self.pm.get_build("build-no-id")
        self.assertEqual(data["status"], "failure")
        self.assertFalse(data["alive"])

    async def test_build_stop_becomes_cancelled_and_kills_process_group(self):
        parent = (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
            "print(p.pid, flush=True);"
            "time.sleep(30)"
        )
        with mock.patch(
            "web.backend.process_manager._build_autodevops_cmd",
            return_value=[sys.executable, "-c", parent],
        ):
            rec = await self.pm.start_build({"ref": "latest"})
            await asyncio.sleep(0.6)
            logs = [line.rstrip() for line in self.pm.get_build_logs(rec.id, tail=20)]
            child_pid = int(next(line for line in logs if line.isdigit()))
            await self.pm.stop_build(rec.id)
            data = self.pm.get_build(rec.id)
            self.assertEqual(data["status"], "cancelled")
            with self.assertRaises(OSError):
                os.kill(child_pid, 0)

    async def test_finished_build_logs_remain_readable_without_live_buffer(self):
        with mock.patch(
            "web.backend.process_manager._build_autodevops_cmd",
            return_value=[sys.executable, "-c", 'print("hello-build")'],
        ):
            rec = await self.pm.start_build({"ref": "latest"})
            await asyncio.sleep(0.5)
            self.assertIsNone(self.pm.get_live_build_buffer(rec.id))
            logs = [line.rstrip() for line in self.pm.get_build_logs(rec.id)]
            self.assertTrue(any("hello-build" in line for line in logs))
            self.assertEqual(self.pm.get_build(rec.id)["status"], "success")


@unittest.skipUnless(FASTAPI_OK, "fastapi not installed")
class BuildFlagTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        builds_route._clear_flag_cache()

    async def test_probe_parses_autodevops_help(self):
        spec = await builds_route._probe_supported_flags()
        self.assertIn("--force-mmq", spec["choice_flags"])
        self.assertIn("--blas", spec["choice_flags"])
        self.assertIn("--fast-math", spec["bool_flags"])
        self.assertIn("--ref", spec["value_flags"])
        option_map = {option["flag"]: option for option in spec["options"]}
        self.assertEqual(option_map["--ref"]["kind"], "value")
        self.assertIn("git tag/branch/commit", option_map["--ref"]["description"])

    async def test_probe_skips_wrapped_usage_when_building_summary(self):
        help_text = """usage: autodevops.py [-h] [--now] [--ref REF]
                     [--force-mmq {auto,on,off}]

Automated llama.cpp build (CUDA + BLAS).

options:
  -h, --help            show this help message and exit
"""

        async def fake_exec(*args, **kwargs):
            class FakeProc:
                async def communicate(self):
                    return help_text.encode("utf-8"), b""

            return FakeProc()

        with mock.patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            spec = await builds_route._probe_supported_flags()

        self.assertEqual(spec["usage"], "usage: autodevops.py [-h] [--now] [--ref REF]")
        self.assertEqual(spec["summary"], "Automated llama.cpp build (CUDA + BLAS).")

    async def test_validate_rejects_unknown_choice(self):
        req = builds_route.BuildRequest(blas="bogus")
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            await builds_route._validate_request(req)


@unittest.skipUnless(FASTAPI_OK, "fastapi not installed")
class RouteTests(unittest.TestCase):
    def setUp(self):
        self.cfg_path = REPO_ROOT / "tests" / "_tmp_web_config.json"
        self.state_path = REPO_ROOT / "tests" / "_tmp_web_state.json"
        for path in (self.cfg_path, self.state_path):
            if path.exists():
                path.unlink()

        self._cfg_patch = mock.patch.object(cfg_mod, "CONFIG_PATH", self.cfg_path)
        self._state_patch = mock.patch.object(cfg_mod, "STATE_PATH", self.state_path)
        self._cfg_patch.start()
        self._state_patch.start()

        from web.backend import app as app_module
        self._app_state_patch = mock.patch.object(app_module, "STATE_PATH", self.state_path)
        self._app_state_patch.start()

        from web.backend.app import create_app

        self.cfg = cfg_mod.init_config(self.cfg_path, force=True)
        self.app = create_app(self.cfg)
        self.client = TestClient(self.app)
        self._pids_to_cleanup: list[int] = []

    def tearDown(self):
        self.client.close()
        self._app_state_patch.stop()
        self._state_patch.stop()
        self._cfg_patch.stop()
        for pid in self._pids_to_cleanup:
            _kill_pid(pid)
        for path in (self.cfg_path, self.state_path):
            if path.exists():
                path.unlink()

    def _hdr(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.cfg.token}"}

    def test_health_is_public(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_recover_route_adopts_orphan_with_backend_marker(self):
        self.client.get("/api/health")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_server = _write_fake_llama_server(tmp_path)
            log_file = tmp_path / "managed.log"
            env = os.environ.copy()
            env[process_manager.RECOVERY_MARKER_ENV] = "1"
            env[process_manager.RECOVERY_INSTANCE_ID_ENV] = "recoverme1234"
            env[process_manager.RECOVERY_INSTANCE_NAME_ENV] = "Recovered via API"
            env[process_manager.RECOVERY_LOG_FILE_ENV] = str(log_file)
            with log_file.open("a", encoding="utf-8") as sink:
                proc = subprocess.Popen(
                    [
                        str(fake_server),
                        "--model",
                        "/tmp/fake.gguf",
                        "--port",
                        "45567",
                    ],
                    stdout=sink,
                    stderr=sink,
                    env=env,
                    start_new_session=True,
                )
            self._pids_to_cleanup.append(proc.pid)
            time.sleep(0.3)

            with mock.patch.object(process_manager, "LLAMA_SERVER", fake_server):
                r = self.client.post("/api/instances/recover", headers=self._hdr())
            with contextlib_suppress(Exception):
                proc.terminate()
            with contextlib_suppress(Exception):
                proc.wait(timeout=2)

        self.assertEqual(r.status_code, 200)
        body = r.json()
        recovered = next(inst for inst in body["recovered"] if inst["id"] == "recoverme1234")
        self.assertEqual(recovered["id"], "recoverme1234")
        self.assertEqual(recovered["name"], "Recovered via API")
        self.assertEqual(recovered["port"], 45567)
        self.assertTrue(recovered["alive"])

    def test_instance_get_returns_file_backed_logs_when_inactive(self):
        log_file = REPO_ROOT / "tests" / "_tmp_route_instance.log"
        log_file.write_text("alpha\nbeta\n", encoding="utf-8")
        try:
            manager = self.app.state.manager
            rec = state_mod.InstanceRecord(
                id="inst1",
                name="inst1",
                status="stopped",
                log_file=str(log_file),
            )
            manager._instances["inst1"] = process_manager.ManagedInstance(
                record=rec,
                log_buffer=LogBuffer(log_file=log_file),
            )
            manager.store.upsert_instance(rec)
            r = self.client.get("/api/instances/inst1", headers=self._hdr())
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["logs"], ["alpha\n", "beta\n"])
        finally:
            if log_file.exists():
                log_file.unlink()

    def test_build_get_returns_file_backed_logs_when_inactive(self):
        log_file = REPO_ROOT / "tests" / "_tmp_route_build.log"
        log_file.write_text("one\ntwo\n", encoding="utf-8")
        try:
            rec = state_mod.BuildRecord(
                id="build1",
                status="success",
                log_file=str(log_file),
            )
            self.app.state.manager.store.upsert_build(rec)
            r = self.client.get("/api/builds/build1", headers=self._hdr())
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["logs"], ["one\n", "two\n"])
        finally:
            if log_file.exists():
                log_file.unlink()

    def test_memory_gpu_route_includes_system_and_processes(self):
        manager = self.app.state.manager
        manager._instances["gpuinst"] = process_manager.ManagedInstance(
            record=state_mod.InstanceRecord(
                id="gpuinst",
                name="chat-prod",
                status="running",
                pid=4242,
                host="127.0.0.1",
                port=45540,
            ),
            log_buffer=LogBuffer(),
        )

        captured: dict[str, object] = {}

        def fake_detect_gpu_runtime(*, managed_processes):
            captured["managed_processes"] = managed_processes
            return [
                {
                    "index": 0,
                    "name": "RTX 4090",
                    "uuid": "GPU-123",
                    "total": 24 * 1024**3,
                    "free": 8 * 1024**3,
                    "used": 16 * 1024**3,
                    "utilization_gpu": 81,
                    "utilization_memory": 44,
                    "memory_percent": 66.7,
                    "processes": [
                        {
                            "pid": 4242,
                            "process_name": "llama-server",
                            "raw_process_name": "llama-server",
                            "label": "chat-prod",
                            "kind": "instance",
                            "status": "running",
                            "detail": "127.0.0.1:45540",
                            "used_memory": 6 * 1024**3,
                            "memory_percent": 25.0,
                        }
                    ],
                }
            ]

        with mock.patch.object(memory_route.memory_utils, "detect_gpu_runtime", side_effect=fake_detect_gpu_runtime), \
             mock.patch.object(
                 memory_route.memory_utils,
                 "detect_system_usage",
                 return_value={
                     "cpu_percent": 41.5,
                     "cpu_count_logical": 32,
                     "cpu_count_physical": 16,
                     "load_1": 1.25,
                     "load_5": 1.1,
                     "load_15": 0.95,
                     "memory_total": 64 * 1024**3,
                     "memory_available": 20 * 1024**3,
                     "memory_used": 44 * 1024**3,
                     "memory_percent": 68.8,
                     "cores": [{"index": 0, "percent": 52.0}],
                 },
             ):
            r = self.client.get("/api/memory/gpus", headers=self._hdr())

        self.assertEqual(r.status_code, 200)
        self.assertIn(4242, captured["managed_processes"])
        self.assertEqual(captured["managed_processes"][4242]["label"], "chat-prod")

        body = r.json()
        self.assertEqual(body["system"]["cpu_percent"], 41.5)
        self.assertEqual(body["system"]["memory_used_h"], "44.00 GB")
        self.assertEqual(body["gpus"][0]["utilization_gpu"], 81)
        self.assertEqual(body["gpus"][0]["used_h"], "16.00 GB")
        self.assertEqual(body["gpus"][0]["processes"][0]["label"], "chat-prod")
        self.assertEqual(body["gpus"][0]["processes"][0]["used_memory_h"], "6.00 GB")

    def test_instance_log_websocket_accepts_active_reattached_process(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        log_file = REPO_ROOT / "tests" / "_tmp_ws_instance.log"
        log_file.write_text("boot\n", encoding="utf-8")
        try:
            manager = self.app.state.manager
            rec = state_mod.InstanceRecord(
                id="instws",
                name="instws",
                status="running",
                pid=proc.pid,
                pgid=os.getpgid(proc.pid),
                cmdline=[sys.executable, "-c", "import time; time.sleep(30)"],
                log_file=str(log_file),
            )
            buf = LogBuffer(log_file=log_file)
            buf.seed_from_file()
            manager._instances["instws"] = process_manager.ManagedInstance(record=rec, log_buffer=buf)
            with self.client.websocket_connect(f"/api/instances/instws/logs?token={self.cfg.token}") as ws:
                self.assertEqual(ws.receive_text(), "boot")
        finally:
            with contextlib_suppress(Exception):
                proc.terminate()
            with contextlib_suppress(Exception):
                proc.wait(timeout=2)
            if log_file.exists():
                log_file.unlink()

    def test_build_log_websocket_accepts_active_reattached_process(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        log_file = REPO_ROOT / "tests" / "_tmp_ws_build.log"
        log_file.write_text("build-boot\n", encoding="utf-8")
        try:
            manager = self.app.state.manager
            rec = state_mod.BuildRecord(
                id="buildws",
                status="running",
                pid=proc.pid,
                pgid=os.getpgid(proc.pid),
                cmdline=[sys.executable, "-c", "import time; time.sleep(30)"],
                log_file=str(log_file),
            )
            buf = LogBuffer(log_file=log_file)
            buf.seed_from_file()
            manager._builds["buildws"] = process_manager.ManagedBuild(record=rec, log_buffer=buf)
            manager.store.upsert_build(rec)
            with self.client.websocket_connect(f"/api/builds/buildws/logs?token={self.cfg.token}") as ws:
                self.assertEqual(ws.receive_text(), "build-boot")
        finally:
            with contextlib_suppress(Exception):
                proc.terminate()
            with contextlib_suppress(Exception):
                proc.wait(timeout=2)
            if log_file.exists():
                log_file.unlink()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
