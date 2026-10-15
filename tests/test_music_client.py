"""The client for the music server's job API. Each test runs against the stdlib fake server in tests/."""
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_music_server import FakeMusicServer          # noqa: E402
from music_client import (MAX_SECONDS, MIN_SECONDS, MusicClient, MusicServerError,   # noqa: E402
                          ServerBusy, ServerDown)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_submit_sends_the_fields_the_server_reads():
    with FakeMusicServer() as srv:
        jid = MusicClient(srv.url).submit("[Verse]\nla la", "warm folk", 20, seed=7, steps=12)
        assert jid == "job001"
        assert srv.submitted == [{"input": "[Verse]\nla la", "instructions": "warm folk", "audio_duration": 20,
                                  "response_format": "flac", "seed": 7, "num_inference_steps": 12}]


def test_optional_fields_are_left_out_when_not_given():
    with FakeMusicServer() as srv:
        MusicClient(srv.url).submit("x", "y", 20)
        assert "seed" not in srv.submitted[0] and "num_inference_steps" not in srv.submitted[0]


@pytest.mark.parametrize("lyrics,caption,seconds,seed,steps,fragment", [
    ("", "c", 20, None, None, "lyrics"),
    ("   ", "c", 20, None, None, "lyrics"),
    ("l", "", 20, None, None, "caption"),
    ("l", "c", MIN_SECONDS - 1, None, None, "seconds"),
    ("l", "c", MAX_SECONDS + 1, None, None, "seconds"),
    ("l", "c", 20, -1, None, "seed"),
    ("l", "c", 20, 1.5, None, "seed"),
    ("l", "c", 20, None, 0, "steps"),
    ("l", "c", 20, None, 101, "steps"),
])
def test_bad_input_is_refused_before_any_request(lyrics, caption, seconds, seed, steps, fragment):
    with FakeMusicServer() as srv:
        with pytest.raises(ValueError, match=fragment):
            MusicClient(srv.url).submit(lyrics, caption, seconds, seed=seed, steps=steps)
        assert srv.submitted == []


def test_full_queue_is_a_busy_error_with_a_plain_message():
    with FakeMusicServer(queue_limit=1) as srv:
        c = MusicClient(srv.url)
        c.submit("l", "c", 20)
        c.submit("l", "c", 20)
        with pytest.raises(ServerBusy, match="busy"):
            c.submit("l", "c", 20)


def test_server_down_is_reported_not_raised_as_a_raw_urlerror():
    c = MusicClient(f"http://127.0.0.1:{free_port()}", timeout=2)
    with pytest.raises(ServerDown, match="not reachable"):
        c.health()
    with pytest.raises(ServerDown):
        c.submit("l", "c", 20)


def test_health_returns_the_servers_fields():
    with FakeMusicServer(model="fiona-crapple-merged") as srv:
        h = MusicClient(srv.url).health()
        assert h["model"] == "fiona-crapple-merged" and h["status"] == "ok"


def test_generate_polls_to_done_and_reports_progress():
    seen = []
    with FakeMusicServer(steps_before_done=2) as srv:
        r = MusicClient(srv.url).generate("l", "c", 20, seed=3, poll_s=0.01, on_progress=lambda f, m, stage: seen.append((f, m, stage)))
    assert r.audio.startswith(b"fLaC-fake-audio-") and r.job_id == "job001"
    assert r.stats["denoise_s"] == 23.9 and r.stats["seed"] == 3
    assert [s[0] for s in seen] == sorted(s[0] for s in seen) and len(seen) >= 2    # frames never go backwards
    assert all(m == 500 for _, m, _ in seen)


def test_generate_raises_the_servers_error_text():
    with FakeMusicServer(fail_with="out of memory") as srv:
        with pytest.raises(MusicServerError, match="out of memory"):
            MusicClient(srv.url).generate("l", "c", 20, poll_s=0.01)


def test_generate_gives_up_at_the_timeout_and_cancels_the_job():
    with FakeMusicServer(steps_before_done=10_000) as srv:
        with pytest.raises(MusicServerError, match="timed out"):
            MusicClient(srv.url).generate("l", "c", 20, poll_s=0.01, timeout_s=0.2)
        assert srv.cancelled == ["job001"]


def test_generate_cancels_the_job_when_interrupted():
    def boom(*a):
        raise KeyboardInterrupt
    with FakeMusicServer(steps_before_done=10_000) as srv:
        with pytest.raises(KeyboardInterrupt):
            MusicClient(srv.url).generate("l", "c", 20, poll_s=0.01, on_progress=boom)
        assert srv.cancelled == ["job001"]
