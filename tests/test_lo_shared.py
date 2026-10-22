"""One ffmpeg for both lo channels (--lo-shared).

Protect opens video2 AND video3, and the proxy used to spawn one ffmpeg per
channel, so with --lo-h265-transcode --lo-decode cpu every camera's main
stream was decoded (and scaled) twice on the CPU. --lo-shared runs ONE
pipeline: one input, one decode + scale, split, one hevc_qsv encode per
channel, each into its own FLV -> hevc_flv -> clock_sync -> nc tail. The
encode and the mux stay in one process: a relay-made lo stream copied over
RTSP lost the first packet's PTS (Intel HEVC encoders flag has_b_frames=1),
which FLV rejects.
"""

import argparse
import asyncio
import logging
import os
import subprocess

import pytest

from unifi.cams.base import SmartDetectObjectType  # noqa: F401
from unifi.cams.rtsp import RTSPCam

DEST2 = ("10.0.0.6", 7550)
DEST3 = ("10.0.0.6", 7551)


class FakeHandle:
    def __init__(self, pid, rc=None):
        self.pid = pid
        self._rc = rc

    @property
    def returncode(self):
        return self._rc

    def poll(self):
        return self._rc

    def kill(self):
        self._rc = -9


def make_args(cert, **over):
    args = argparse.Namespace(
        cert=str(cert),
        source=["rtsp://relay/main"],
        snapshot_url="http://cam/snap",
        rtsp_transport="tcp",
        hi_width=2560,
        hi_height=1440,
        hi_codec="h265",
        ffmpeg_args="-c:v copy -ar 16000 -ac 1 -codec:a aac -b:a 32k",
        hw_transcode=False,
        lo_h265_transcode=True,
        lo_shared=True,
        lo_decode="cpu",
        lo_decode_threads=2,
        lo_scale=True,
        lo_width=1920,
        lo_height=1080,
        lo_fps=20,
        lo_transcode_bitrate="1M",
        lo_low_power=True,
        lo_audio_args="-ar 16000 -ac 1 -codec:a aac -b:a 32k",
        hw_device="/dev/dri/renderD128",
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


@pytest.fixture
def cert(tmp_path_factory):
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
def env(cert, monkeypatch):
    """A camera whose spawns and kills are recorded instead of run."""
    cam = RTSPCam(make_args(cert), logging.getLogger("test"))
    spawned = []  # (cmd, handle)
    killed = []  # pgids
    real_check_output = subprocess.check_output

    def fake_check_output(cmd, *a, **kw):
        if cmd[:2] == ["ffmpeg", "-h"]:
            return b"-timeout"
        return real_check_output(cmd, *a, **kw)

    def fake_popen(cmd, **kw):
        h = FakeHandle(pid=5000 + len(spawned))
        spawned.append((cmd, h, kw))
        return h

    def fake_killpg(pgid, sig):
        killed.append(pgid)
        for _, h, _ in spawned:
            if h.pid == pgid:
                h._rc = -9

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(os, "killpg", fake_killpg)
    return cam, spawned, killed


# --- command ---------------------------------------------------------------


async def test_both_lo_channels_share_one_pipeline(env):
    cam, spawned, killed = env
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    cmd, handle, kw = spawned[-1]
    assert cam._ffmpeg_handles["video2"] is handle
    assert cam._ffmpeg_handles["video3"] is handle
    assert cmd.count(" -i ") == 1  # one input, one decode
    assert cmd.count("scale=1920:1080") == 1  # one scale
    assert "split=2" in cmd
    assert cmd.count("-c:v hevc_qsv -b:v 1M -low_power 1 -g 20") == 2
    assert "streamName=MED" in cmd and "streamName=LOW" in cmd
    assert cmd.count("-m unifi.hevc_flv") == 2
    assert cmd.count("-m unifi.clock_sync") == 2
    assert "nc 10.0.0.6 7550" in cmd and "nc 10.0.0.6 7551" in cmd
    assert "3> >(" in cmd and "4> >(" in cmd
    assert "pipe:3" in cmd and "pipe:4" in cmd
    assert "-init_hw_device vaapi=va:/dev/dri/renderD128" in cmd
    assert "-hwaccel" not in cmd  # HEVC decode stays on the CPU
    assert kw.get("executable") == "/bin/bash"  # process substitution
    assert kw.get("start_new_session") is True


async def test_single_lo_channel_has_no_split(env):
    cam, spawned, _ = env
    await cam.start_video_stream("video3", "LOW", DEST3)
    cmd = spawned[-1][0]
    assert "split" not in cmd
    assert cmd.count("-c:v hevc_qsv") == 1
    assert "streamName=LOW" in cmd and "nc 10.0.0.6 7551" in cmd
    assert "pipe:3" in cmd and "pipe:4" not in cmd


async def test_no_audio_drops_audio_on_both_outputs(env):
    cam, spawned, _ = env
    cam.args.no_audio = True
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    cmd = spawned[-1][0]
    assert "0:a" not in cmd
    assert "-codec:a aac" not in cmd


async def test_audio_is_mapped_per_output(env):
    cam, spawned, _ = env
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    cmd = spawned[-1][0]
    assert cmd.count("-map 0:a?") == 2
    assert cmd.count("-ar 16000 -ac 1 -codec:a aac -b:a 32k") == 2


# --- lifecycle --------------------------------------------------------------


async def test_second_lo_request_respawns_once_covering_both(env):
    cam, spawned, killed = env
    await cam.start_video_stream("video2", "MED", DEST2)
    first = spawned[0][1]
    await cam.start_video_stream("video3", "LOW", DEST3)
    assert len(spawned) == 2
    assert killed == [first.pid]


async def test_covered_live_channel_is_left_alone(env):
    cam, spawned, killed = env
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    assert len(spawned) == 2


async def test_real_stop_of_one_lo_channel_keeps_the_other(env):
    cam, spawned, killed = env
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    shared = spawned[-1][1]
    cam.stop_video_stream("video3")  # Protect sent video3 to /dev/null
    assert shared.pid in killed
    assert "video3" not in cam._ffmpeg_handles
    assert "video3" not in cam._stream_meta
    await asyncio.sleep(0)  # the peer respawn is deferred one loop turn
    assert len(spawned) == 3
    cmd, h, _ = spawned[-1]
    assert cam._ffmpeg_handles["video2"] is h
    assert "streamName=MED" in cmd and "streamName=LOW" not in cmd


async def test_stop_then_start_restarts_the_pair_once(env):
    # Re-request and watchdog paths do stop(k) then start(k) back to back.
    cam, spawned, killed = env
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    cam.stop_video_stream("video2")
    await cam.start_video_stream("video2", "MED2", DEST2)
    await asyncio.sleep(0)
    assert len(spawned) == 3  # exactly one new pipeline, not two
    cmd, h, _ = spawned[-1]
    assert "streamName=MED2" in cmd and "streamName=LOW" in cmd
    assert cam._ffmpeg_handles["video2"] is h
    assert cam._ffmpeg_handles["video3"] is h


async def test_close_streams_does_not_respawn(env):
    cam, spawned, killed = env
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    cam.close_streams()
    await asyncio.sleep(0)
    assert len(spawned) == 2
    assert cam._ffmpeg_handles == {}


def test_stop_without_a_running_loop_respawns_immediately(env):
    # close() and tests call stop outside the event loop: respawn right away.
    cam, spawned, killed = env

    async def start_both():
        await cam.start_video_stream("video2", "MED", DEST2)
        await cam.start_video_stream("video3", "LOW", DEST3)

    asyncio.run(start_both())
    shared = spawned[-1][1]
    cam.stop_video_stream("video3")
    assert shared.pid in killed
    assert len(spawned) == 3
    assert "streamName=MED" in spawned[-1][0]
    assert "streamName=LOW" not in spawned[-1][0]


async def test_dead_shared_pipeline_is_respawned_once_by_the_watchdog(env, monkeypatch):
    cam, spawned, killed = env
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    spawned[-1][1]._rc = 1  # the shared pipeline exited
    monkeypatch.setattr(cam, "_stream_output_bytes", lambda i: None)
    await cam._watchdog_tick({}, now=0.0)
    await asyncio.sleep(0)
    assert len(spawned) == 3


async def test_hi_channel_is_unchanged(env):
    cam, spawned, _ = env
    await cam.start_video_stream("video1", "HI", ("10.0.0.6", 7552))
    cmd, h, kw = spawned[-1]
    assert "-c:v copy" in cmd and "hevc_qsv" not in cmd
    assert "split" not in cmd
    assert cmd.rstrip().endswith("nc 10.0.0.6 7552")
    assert kw.get("executable") is None


async def test_without_lo_shared_each_lo_channel_has_its_own_pipeline(env):
    cam, spawned, _ = env
    cam.args.lo_shared = False
    await cam.start_video_stream("video2", "MED", DEST2)
    await cam.start_video_stream("video3", "LOW", DEST3)
    assert len(spawned) == 2
    assert cam._ffmpeg_handles["video2"] is not cam._ffmpeg_handles["video3"]
    assert all("split" not in c for c, _, _ in spawned)


# --- watchdog ---------------------------------------------------------------


async def test_watchdog_rebaselines_a_replaced_pipeline(env, monkeypatch):
    # When a pipeline is replaced (new pid) its byte counter starts over; the
    # old baseline must not make the new one look silent.
    cam, spawned, killed = env
    restarts = []
    monkeypatch.setattr(cam, "stop_video_stream", lambda i: restarts.append(i))
    cam._stream_meta["video3"] = ("LOW", DEST3)
    cam._ffmpeg_handles["video3"] = FakeHandle(pid=1)
    out = {"video3": 1_000_000}
    monkeypatch.setattr(cam, "_stream_output_bytes", lambda i: out.get(i))
    progress = {}
    await cam._watchdog_tick(progress, now=0.0)
    cam._ffmpeg_handles["video3"] = FakeHandle(pid=2)  # replaced by a peer restart
    out["video3"] = 500  # fresh counter, below the old baseline
    await cam._watchdog_tick(progress, now=cam.STREAM_SILENCE_LIMIT_SEC + 1)
    assert restarts == []
    out["video3"] = 900_000
    await cam._watchdog_tick(progress, now=cam.STREAM_SILENCE_LIMIT_SEC + 20)
    assert restarts == []
