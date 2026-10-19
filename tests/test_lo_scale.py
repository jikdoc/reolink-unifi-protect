"""Cutting the medium/low channels from the main stream.

Real case: a Wi-Fi Reolink Elite Floodlight carried its 3 Mbps main stream alone at full
rate, but starved (249 of 400 frames) when main and sub were pulled together, so Protect
timed out every channel. With --lo-scale every channel is fed from one main stream and
the lo channels are scaled on the iGPU.
"""

import argparse

from unifi.cams.rtsp import RTSPCam


def make_cam(**args) -> RTSPCam:
    c = RTSPCam.__new__(RTSPCam)
    c.args = argparse.Namespace(
        hw_transcode=False,
        lo_h265_transcode=True,
        lo_transcode_bitrate="1M",
        lo_width=1920,
        lo_height=576,
        lo_fps=20,
        ffmpeg_args="-c:v copy -c:a copy",
        hi_codec="h265",
    )
    for k, v in args.items():
        setattr(c.args, k, v)
    return c


def test_lo_scale_adds_the_qsv_scaler_to_the_lo_channels():
    cam = make_cam(lo_scale=True)
    for stream in ("video2", "video3"):
        args = cam.get_extra_ffmpeg_args(stream)
        assert args.startswith('-vf "vpp_qsv=w=1920:h=576" ')
        assert "-c:v hevc_qsv -b:v 1M -g 20" in args


def test_without_lo_scale_the_transcode_keeps_its_size():
    args = make_cam().get_extra_ffmpeg_args("video3")
    assert "vpp_qsv" not in args
    assert args.startswith("-c:v hevc_qsv")


def test_the_main_channel_is_never_scaled():
    assert make_cam(lo_scale=True).get_extra_ffmpeg_args("video1") == (
        "-c:v copy -c:a copy"
    )


def test_no_audio_drops_audio_on_every_channel():
    cam = make_cam(lo_scale=True, no_audio=True)
    for stream in ("video1", "video2", "video3"):
        assert cam.get_extra_ffmpeg_args(stream).endswith(" -an")


def test_audio_is_kept_by_default():
    cam = make_cam(lo_scale=True)
    assert "-an" not in cam.get_extra_ffmpeg_args("video1").split()
    assert "-c:a copy" in cam.get_extra_ffmpeg_args("video3")


def test_lo_audio_args_replace_the_audio_copy():
    cam = make_cam(lo_scale=True, lo_audio_args="-ar 16000 -ac 1 -codec:a aac -b:a 32k")
    args = cam.get_extra_ffmpeg_args("video3")
    assert args.endswith("-g 20 -ar 16000 -ac 1 -codec:a aac -b:a 32k")
    assert "-c:a copy" not in args


# --lo-decode cpu: the QSV HEVC decoder hung the shared iGPU on the Reolink main
# streams, so decode and scale in software and only encode on the iGPU.


def test_lo_decode_defaults_to_qsv():
    cam = make_cam(lo_scale=True, hw_device="/dev/dri/renderD128")
    assert cam._decode_prefix("video3") == "-hwaccel qsv -qsv_device /dev/dri/renderD128 "


def test_lo_decode_cpu_scales_in_software_and_uploads_for_the_encoder():
    cam = make_cam(lo_scale=True, lo_decode="cpu")
    for stream in ("video2", "video3"):
        args = cam.get_extra_ffmpeg_args(stream)
        assert args.startswith('-vf "scale=1920:576,format=nv12,hwupload=extra_hw_frames=16" ')
        assert "vpp_qsv" not in args
        assert "-c:v hevc_qsv -b:v 1M -g 20" in args


def test_lo_decode_cpu_without_lo_scale_keeps_the_size():
    args = make_cam(lo_decode="cpu").get_extra_ffmpeg_args("video3")
    assert args.startswith('-vf "format=nv12,hwupload=extra_hw_frames=16" ')


def test_lo_decode_cpu_drops_hwaccel_and_creates_the_encode_device():
    cam = make_cam(lo_scale=True, lo_decode="cpu", hw_device="/dev/dri/renderD128",
                   lo_decode_threads=2)
    prefix = cam._decode_prefix("video3")
    assert "-hwaccel" not in prefix
    assert prefix == (
        "-init_hw_device vaapi=va:/dev/dri/renderD128 -init_hw_device qsv=hw@va"
        " -filter_hw_device hw -threads 2 "
    )


def test_lo_decode_cpu_leaves_the_main_channel_alone():
    cam = make_cam(lo_scale=True, lo_decode="cpu", hw_device="/dev/dri/renderD128")
    assert cam._decode_prefix("video1") == ""
    assert cam.get_extra_ffmpeg_args("video1") == "-c:v copy -c:a copy"
