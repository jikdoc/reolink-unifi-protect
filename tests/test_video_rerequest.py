"""Protect re-requesting a stream that is still running.

When Protect times out an ingest point (10 s with no stream) it drops that
socket and sends ChangeVideoSettings again, sometimes with a new streamName.
The old pipeline is still alive and writing into the dead socket, so the
watchdog sees output and start_video_stream used to leave it alone: Protect
then retried every 15 s until something else restarted the pipeline. These
tests pin the restart on re-request.
"""

import logging
import sys
import time

import pytest

from unifi.cams.base import SmartDetectObjectType  # noqa: F401
from unifi.cams.rtsp import RTSPCam
from unifi.main import parse_args


class FakeHandle:
    def __init__(self, rc=None, pid=4242):
        self._rc = rc
        self.pid = pid

    @property
    def returncode(self):
        return self._rc

    def poll(self):
        return self._rc

    def kill(self):
        self._rc = -9


@pytest.fixture
def cert(tmp_path_factory):
    import subprocess

    d = tmp_path_factory.mktemp("cert")
    key, crt = d / "k.pem", d / "c.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(key), "-out", str(crt), "-days", "1", "-subj", "/CN=t"],
        check=True,
        capture_output=True,
    )
    pem = d / "client.pem"
    pem.write_bytes(key.read_bytes() + crt.read_bytes())
    return pem


@pytest.fixture
def cam(cert, monkeypatch):
    # The real CLI parser, so process_video_settings sees every default it
    # reads while building its response.
    monkeypatch.setattr(
        sys,
        "argv",
        ["unifi-cam-proxy", "--host", "10.0.0.6", "--cert", str(cert),
         "rtsp", "-s", "rtsp://cam/main"],
    )
    args = parse_args()
    cam = RTSPCam(args, logging.getLogger("test"))
    calls = []
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    async def fake_start(idx, name, destination):
        # Like the real one: a live pipeline is left as it is.
        h = cam._ffmpeg_handles.get(idx)
        if h is not None and h.poll() is None:
            return
        calls.append(("start", idx, name, destination))
        cam._ffmpeg_handles[idx] = FakeHandle()
        cam._stream_meta[idx] = (name, destination)
        cam._stream_started[idx] = clock[0]

    def fake_stop(idx):
        calls.append(("stop", idx))
        cam._ffmpeg_handles.pop(idx, None)
        cam._stream_meta.pop(idx, None)

    monkeypatch.setattr(cam, "start_video_stream", fake_start)
    monkeypatch.setattr(cam, "stop_video_stream", fake_stop)
    cam._test_calls = calls
    cam._clock = clock
    return cam


def later(cam, seconds):
    cam._clock[0] += seconds


def request(stream_name, channel="video3", dest="tcp://10.0.0.6:7550?retryInterval=1"):
    return {
        "functionName": "ChangeVideoSettings",
        "messageId": 588235,
        "payload": {
            "video": {
                channel: {
                    "avSerializer": {
                        "type": "extendedFlv",
                        "parameters": {"streamName": stream_name},
                        "destinations": [dest],
                    },
                    "type": "h265",
                }
            }
        },
    }


async def test_first_request_starts_without_stop(cam):
    await cam.process_video_settings(request("abc"))
    assert cam._test_calls == [("start", "video3", "abc", ("10.0.0.6", 7550))]


async def test_rerequest_of_running_stream_restarts_it(cam):
    await cam.process_video_settings(request("abc"))
    later(cam, 15)  # Protect timed out the ingest point (10 s) and asks again
    await cam.process_video_settings(request("abc"))
    assert cam._test_calls == [
        ("start", "video3", "abc", ("10.0.0.6", 7550)),
        ("stop", "video3"),
        ("start", "video3", "abc", ("10.0.0.6", 7550)),
    ]


async def test_new_stream_name_is_used_on_restart(cam):
    await cam.process_video_settings(request("old"))
    await cam.process_video_settings(request("new"))
    assert cam._test_calls[-2:] == [
        ("stop", "video3"),
        ("start", "video3", "new", ("10.0.0.6", 7550)),
    ]
    assert cam._streams["video3"] == "new"


async def test_dead_pipeline_is_started_without_stop(cam):
    cam._ffmpeg_handles["video3"] = FakeHandle(rc=1)  # already exited
    await cam.process_video_settings(request("abc"))
    assert cam._test_calls == [("start", "video3", "abc", ("10.0.0.6", 7550))]


async def test_rerequest_only_touches_that_channel(cam):
    cam._ffmpeg_handles["video1"] = FakeHandle()
    await cam.process_video_settings(request("abc", channel="video3"))
    later(cam, 15)
    await cam.process_video_settings(request("abc", channel="video3"))
    assert ("stop", "video1") not in cam._test_calls
    assert "video1" in cam._ffmpeg_handles


async def test_rerequest_right_after_a_restart_keeps_fresh_pipeline(cam):
    # The watchdog (or a previous re-request) just respawned the stream;
    # Protect notices the drop and asks again within a few seconds. The new
    # pipeline already is a fresh connection, so restarting it again would
    # only kill the socket Protect is opening (observed: 6 restarts in 17 s).
    await cam.process_video_settings(request("abc"))
    for _ in range(3):  # re-requests 3, 6 and 9 s after the spawn
        later(cam, 3)
        await cam.process_video_settings(request("abc"))
    assert [c for c in cam._test_calls if c[0] == "stop"] == []


async def test_new_stream_name_restarts_even_within_grace(cam):
    await cam.process_video_settings(request("old"))
    later(cam, 2)
    await cam.process_video_settings(request("new"))
    assert cam._test_calls[-2:] == [
        ("stop", "video3"),
        ("start", "video3", "new", ("10.0.0.6", 7550)),
    ]


async def test_grace_is_measured_from_the_last_spawn(cam):
    await cam.process_video_settings(request("abc"))
    later(cam, cam.STREAM_REREQUEST_GRACE_SEC - 1)
    await cam.process_video_settings(request("abc"))
    assert ("stop", "video3") not in cam._test_calls
    later(cam, 2)  # now past the grace period since the original spawn
    await cam.process_video_settings(request("abc"))
    assert ("stop", "video3") in cam._test_calls
