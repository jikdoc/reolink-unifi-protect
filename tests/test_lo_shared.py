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
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(crt),
            "-days",
            "1",
            "-subj",
            "/CN=t",
        ],
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


# --- EPIPE recovery ---------------------------------------------------------
# rc 224 = Protect closed an ingest socket: both channels' cached grants are
# void, so the bridge waits for fresh requests instead of respawning.


class FakeWS:
    def __init__(self, clock=None):
        self.closed = 0
        self.at = []
        self._clock = clock

    async def close(self):
        self.closed += 1
        if self._clock is not None:
            self.at.append(self._clock[0])


@pytest.fixture
def epipe(env, monkeypatch):
    import unifi.cams.base as base

    cam, spawned, killed = env
    clock = [1000.0]
    monkeypatch.setattr(base.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(cam, "_stream_output_bytes", lambda i: None)
    cam._session = FakeWS(clock)
    return cam, spawned, killed, clock


async def start_both(cam, n2="MED", n3="LOW"):
    await cam.start_video_stream("video2", n2, DEST2)
    await cam.start_video_stream("video3", n3, DEST3)


async def test_epipe_exit_is_not_blindly_respawned(epipe):
    cam, spawned, killed, clock = epipe
    await start_both(cam)
    assert len(spawned) == 2
    dead = spawned[-1][1]
    dead._rc = 224
    for _ in range(4):
        clock[0] += 15
        await cam._watchdog_tick({}, now=clock[0])
    assert len(spawned) == 2
    assert "video2" not in cam._ffmpeg_handles
    assert "video3" not in cam._ffmpeg_handles
    assert "video2" not in cam._stream_meta
    assert "video3" not in cam._stream_meta
    assert dead.pid in killed


async def test_broken_pipe_loop_ends_at_the_first_rerequest(epipe):
    cam, spawned, killed, clock = epipe
    await start_both(cam)
    t0 = clock[0]
    progress = {}
    for i in range(20):
        clock[0] = t0 + 15 * (i + 1)
        for cmd, h, _ in spawned:
            if h.poll() is None and "streamName=LOW" in cmd:
                h._rc = 224
        await cam._watchdog_tick(progress, now=clock[0])
        if i == 0:
            clock[0] = t0 + 16
            await cam.start_video_stream("video2", "MED2", DEST2)
    cmd, h, _ = spawned[-1]
    assert cam._ffmpeg_handles["video2"] is h and h.poll() is None
    assert "streamName=MED2" in cmd and "streamName=LOW" not in cmd
    assert len(spawned) == 3


async def test_rerequest_before_the_tick_drops_the_stale_peer(epipe):
    cam, spawned, killed, clock = epipe
    await start_both(cam)
    dead = spawned[-1][1]
    dead._rc = 224
    await cam.start_video_stream("video2", "MED2", DEST2)
    cmd, h, _ = spawned[-1]
    assert len(spawned) == 3
    assert "streamName=MED2" in cmd and "streamName=LOW" not in cmd
    assert "video3" in cam._lo_state()["waiting"]
    assert "video2" not in cam._lo_state()["waiting"]
    assert "video3" not in cam._stream_meta
    assert dead.pid in killed


async def test_peer_rejoins_on_its_own_rerequest(epipe):
    cam, spawned, killed, clock = epipe
    await start_both(cam)
    spawned[-1][1]._rc = 224
    await cam.start_video_stream("video2", "MED2", DEST2)
    solo = spawned[-1][1]
    await cam.start_video_stream("video3", "LOW2", DEST3)
    assert len(spawned) == 4
    cmd, h, _ = spawned[-1]
    assert "streamName=MED2" in cmd and "streamName=LOW2" in cmd
    assert cam._ffmpeg_handles["video2"] is h and cam._ffmpeg_handles["video3"] is h
    assert solo.pid in killed
    assert cam._lo_state()["waiting"] == {}


async def test_non_epipe_exit_still_respawns_and_sweeps(epipe):
    cam, spawned, killed, clock = epipe
    await start_both(cam)
    dead = spawned[-1][1]
    dead._rc = 1
    clock[0] += 15
    await cam._watchdog_tick({}, now=clock[0])
    await asyncio.sleep(0)
    assert len(spawned) == 3
    assert dead.pid in killed
    cmd = spawned[-1][0]
    assert "streamName=MED" in cmd and "streamName=LOW" in cmd


async def test_stuck_lo_reconnects_once_then_backs_off(epipe):
    cam, spawned, killed, clock = epipe
    ws = cam._session
    await start_both(cam)
    spawned[-1][1]._rc = 224
    t0 = clock[0]
    await cam._watchdog_tick({}, now=t0)
    clock[0] = t0 + 89
    await cam._watchdog_tick({}, now=clock[0])
    assert ws.closed == 0
    clock[0] = t0 + 90
    await cam._watchdog_tick({}, now=clock[0])
    assert ws.closed == 1
    cam.close_streams()  # Core's reconnect runs cam.close()
    await start_both(cam)  # Protect re-requests on the new connection
    spawned[-1][1]._rc = 224  # and the grants are rejected again
    await cam._watchdog_tick({}, now=clock[0])
    nxt = cam._lo_state()["next_reconnect"]
    assert nxt > clock[0] + 90
    clock[0] = nxt - 1
    await cam._watchdog_tick({}, now=clock[0])
    assert ws.closed == 1
    clock[0] = nxt
    await cam._watchdog_tick({}, now=clock[0])
    assert ws.closed == 2


async def test_rerequests_that_keep_hitting_epipe_still_escalate(epipe):
    cam, spawned, killed, clock = epipe
    ws = cam._session
    first_bad = None
    for _ in range(10):
        clock[0] += 16
        await start_both(cam)
        if first_bad is None:
            first_bad = clock[0]
        for _, h, _ in spawned:
            if h.poll() is None:
                h._rc = 224
        await cam._watchdog_tick({}, now=clock[0])
    assert ws.closed == 1
    assert ws.at[0] - first_bad >= 90


async def test_healthy_lo_clears_the_stuck_state(epipe):
    cam, spawned, killed, clock = epipe
    ws = cam._session
    await start_both(cam)
    spawned[-1][1]._rc = 224
    await cam._watchdog_tick({}, now=clock[0])
    assert cam._lo_state()["bad_since"] is not None
    await start_both(cam)
    clock[0] += 61
    await cam._watchdog_tick({}, now=clock[0])
    st = cam._lo_state()
    assert st["bad_since"] is None and st["reconnects"] == 0
    clock[0] += 300
    await cam._watchdog_tick({}, now=clock[0])
    assert ws.closed == 0


async def test_protect_stop_of_waiting_channels_cancels_escalation(epipe):
    cam, spawned, killed, clock = epipe
    ws = cam._session
    await start_both(cam)
    spawned[-1][1]._rc = 224
    await cam._watchdog_tick({}, now=clock[0])
    cam.stop_video_stream("video2")
    cam.stop_video_stream("video3")
    assert cam._lo_state()["waiting"] == {}
    clock[0] += 300
    await cam._watchdog_tick({}, now=clock[0])
    assert ws.closed == 0


async def test_epipe_on_non_shared_or_video1_is_unchanged(epipe):
    cam, spawned, killed, clock = epipe
    # video1 on a shared camera
    await cam.start_video_stream("video1", "HI", ("10.0.0.6", 7552))
    spawned[-1][1]._rc = 224
    clock[0] += 15
    await cam._watchdog_tick({}, now=clock[0])
    assert len(spawned) == 2
    assert killed == []
    # lo channel without --lo-shared
    cam.args.lo_shared = False
    await cam.start_video_stream("video2", "MED", DEST2)
    spawned[-1][1]._rc = 224
    clock[0] += 15
    await cam._watchdog_tick({}, now=clock[0])
    assert len(spawned) == 4
    assert killed == []
    assert cam._lo_state()["waiting"] == {}


def test_leftover_sweep_skips_a_pid_owned_by_a_live_pipeline(env):
    cam, spawned, killed = env
    live, dead = FakeHandle(pid=77), FakeHandle(pid=77, rc=1)
    cam._ffmpeg_handles["video2"] = live
    cam._kill_leftovers(dead)
    assert killed == []
