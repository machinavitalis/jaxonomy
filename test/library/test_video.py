# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import typing as T

import numpy as np
import pytest

import jaxonomy
import jaxonomy.testing as test
from jaxonomy.lazy_loader import LazyLoader
from jaxonomy.library import (
    Gain,
    SignalDatatypeConversion,
    VideoSink,
    VideoSource,
    WhiteNoise,
)


cv2 = LazyLoader("cv2", globals(), "cv2")

if T.TYPE_CHECKING:
    import cv2


@pytest.mark.slow
def test_VideoSink(request):
    test_paths = test.get_paths(request)
    file_name = str(test_paths["workdir"] / "test_video.mp4")

    def _make_sink_diagram():
        builder = jaxonomy.DiagramBuilder()

        source = builder.add(WhiteNoise(0.1, 3, shape=(480, 640, 3), name="source"))
        gain = builder.add(Gain(255.0, name="gain"))
        convert = builder.add(SignalDatatypeConversion(np.uint8, name="convert"))
        sink = builder.add(VideoSink(0.1, file_name, name="sink"))

        builder.connect(source.output_ports[0], gain.input_ports[0])
        builder.connect(gain.output_ports[0], convert.input_ports[0])
        builder.connect(convert.output_ports[0], sink.input_ports[0])

        return builder.build()

    diagram = _make_sink_diagram()
    context = diagram.create_context()

    recorded_signals = {
        "frame_id": diagram["sink"].get_output_port("frame_id"),
    }
    results = jaxonomy.simulate(
        diagram, context, (0.0, 1.0), recorded_signals=recorded_signals
    )

    assert results.time[-1] == 1.0
    assert results.outputs["frame_id"][0] == 0
    assert results.outputs["frame_id"][-1] == 10

    # Check that the video file was created
    assert os.path.exists(file_name)

    # Probe the video file
    cap = cv2.VideoCapture(file_name)
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret
    assert frame.shape == (480, 640, 3)
    cap.release()


def test_VideoSource():
    srcdir = Path(os.path.dirname(__file__)).absolute()

    def _make_source_diagram(no_repeat: bool):
        builder = jaxonomy.DiagramBuilder()

        _source = builder.add(
            VideoSource(
                srcdir / "assets" / "test_video.mp4", no_repeat=no_repeat, name="source"
            )
        )

        return builder.build()

    # no repeat = True (no loop)

    diagram = _make_source_diagram(no_repeat=True)
    context = diagram.create_context()

    recorded_signals = {
        "frame_id": diagram["source"].get_output_port("frame_id"),
        "stopped": diagram["source"].get_output_port("stopped"),
    }

    results = jaxonomy.simulate(
        diagram, context, (0.0, 2.0), recorded_signals=recorded_signals
    )

    assert results.time[-1] == 2.0
    assert results.outputs["frame_id"][0] == 0
    assert results.outputs["frame_id"][-1] == 10
    assert results.outputs["stopped"][0] == 0
    assert results.outputs["stopped"][-1] == 1

    # no repeat = False (loop)

    diagram = _make_source_diagram(no_repeat=False)
    context = diagram.create_context()

    recorded_signals = {
        "frame_id": diagram["source"].get_output_port("frame_id"),
    }

    with pytest.raises(Exception):
        diagram["source"].get_output_port("stopped")

    results = jaxonomy.simulate(
        diagram, context, (0.0, 2.4), recorded_signals=recorded_signals
    )

    assert results.time[-1] == 2.4
    assert results.outputs["frame_id"][0] == 0
    assert results.outputs["frame_id"][-1] == 2  # 2.4s == 11 frames x 2 + 2


def test_video_source_outputs_are_time_derived():
    """``frame_id`` / ``stopped`` must not depend on host-callback bookkeeping.

    They used to read ``self.frame_id`` / ``self.reached_end`` — mutable host
    state written by the *frame* callback — so their values depended on how many
    times, and in what order, the callbacks had run. Callback scheduling is not
    a stable property of the model: it changes with how the enclosing loop is
    compiled, so a run could report a different ``frame_id`` depending on
    whether the simulation end time was a trace-time constant. Both are now
    pure functions of time.
    """
    srcdir = Path(os.path.dirname(__file__)).absolute()
    source = VideoSource(srcdir / "assets" / "test_video.mp4", no_repeat=False)

    fps, n_frames = source.fps, source.video_length

    # Pure function of time: same time in, same answer out, order-independent.
    for t in (0.0, 0.35, 1.0, 2.4, 5.0):
        expected = int(t * fps + 1e-4) % n_frames
        first = int(source._frame_id_cb(t))
        # Interleave an unrelated query; a stateful implementation would drift.
        source._frame_id_cb(t + 1.0)
        assert int(source._frame_id_cb(t)) == first == expected, (
            f"frame_id at t={t} is not a pure function of time"
        )


def test_video_source_stopped_is_time_derived():
    srcdir = Path(os.path.dirname(__file__)).absolute()
    source = VideoSource(srcdir / "assets" / "test_video.mp4", no_repeat=True)
    fps, n_frames = source.fps, source.video_length
    end_time = n_frames / fps

    assert not bool(source._stopped_cb(0.0))
    assert bool(source._stopped_cb(end_time + 1.0))
    # Querying out of order must not latch the flag on.
    assert not bool(source._stopped_cb(0.0)), "stopped latched from a later query"

    # Without looping the frame index holds on the final frame.
    assert int(source._frame_id_cb(end_time + 5.0)) == n_frames - 1
