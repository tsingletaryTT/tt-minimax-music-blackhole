"""The Gradio app: the generate logic against the fake server, and the real app driven through gradio_client."""
import sys
import threading
from pathlib import Path

import gradio as gr
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_music_server import FakeMusicServer          # noqa: E402
import app                                              # noqa: E402
from music_client import MAX_SECONDS, MIN_SECONDS, MusicClient   # noqa: E402


class Progress:
    def __init__(self):
        self.calls = []

    def __call__(self, frac, desc=None):
        self.calls.append((frac, desc))


def run(srv, lyrics="[Verse]\nla", caption="warm folk", seconds=20, seed=5, steps=None, progress=None):
    return app.run_generation(MusicClient(srv.url), lyrics, caption, seconds, seed, steps, progress or Progress(), poll_s=0.01)


def test_generation_returns_a_playable_file_with_the_servers_bytes():
    with FakeMusicServer() as srv:
        path, status, details = run(srv)
    assert Path(path).suffix == ".flac" and Path(path).read_bytes().startswith(b"fLaC-fake-audio-")


def test_status_and_details_report_seed_speed_and_stage_times():
    with FakeMusicServer() as srv:
        _, status, details = run(srv, seed=5, seconds=20)
    assert "seed 5" in status and "20" in status
    assert "3.2" in details                       # 65.0 s of compute for 20 s of audio
    for stage in ("llm", "depth", "denoise", "vocoder"):
        assert stage in details.lower()


@pytest.mark.parametrize("blank", [None, "", -1])
def test_a_blank_seed_is_replaced_by_a_random_one_and_shown(blank):
    with FakeMusicServer() as srv:
        _, status, _ = run(srv, seed=blank)
        sent = srv.submitted[0]["seed"]
    assert isinstance(sent, int) and sent >= 0 and f"seed {sent}" in status


def test_progress_is_reported_as_a_fraction_with_the_stage():
    p = Progress()
    with FakeMusicServer(steps_before_done=3) as srv:
        run(srv, progress=p)
    fracs = [f for f, _ in p.calls]
    assert fracs == sorted(fracs) and 0 <= fracs[0] <= fracs[-1] <= 1
    assert any("semantic" in (d or "") for _, d in p.calls)


def test_a_full_queue_is_shown_as_a_plain_busy_message():
    with FakeMusicServer(queue_limit=-1) as srv:
        with pytest.raises(gr.Error, match="busy"):
            run(srv)


def test_a_server_that_is_down_is_shown_as_not_reachable():
    with pytest.raises(gr.Error, match="not reachable"):
        app.run_generation(MusicClient("http://127.0.0.1:9", timeout=1), "l", "c", 20, 1, None, Progress(), poll_s=0.01)


@pytest.mark.parametrize("kw,fragment", [({"lyrics": "  "}, "lyrics"), ({"caption": ""}, "caption"),
                                         ({"seconds": MAX_SECONDS + 1}, "seconds")])
def test_bad_input_is_shown_as_a_message_before_any_request(kw, fragment):
    with FakeMusicServer() as srv:
        with pytest.raises(gr.Error, match=fragment):
            run(srv, **kw)
        assert srv.submitted == []


def test_a_server_error_text_reaches_the_user():
    with FakeMusicServer(fail_with="out of memory") as srv:
        with pytest.raises(gr.Error, match="out of memory"):
            run(srv)


def test_server_status_names_the_model_or_says_it_is_down():
    with FakeMusicServer(model="fiona-crapple-merged") as srv:
        assert "fiona-crapple-merged" in app.server_status(MusicClient(srv.url))
    assert "not reachable" in app.server_status(MusicClient("http://127.0.0.1:9", timeout=1))


def test_every_example_is_valid_input():
    assert len(app.EXAMPLES) >= 4
    for lyrics, caption, seconds, seed in app.EXAMPLES:
        assert lyrics.strip() and caption.strip() and MIN_SECONDS <= seconds <= MAX_SECONDS and seed >= 0


def test_one_example_uses_the_adapters_trigger_phrase():
    assert any("fiona crapple" in caption for _, caption, _, _ in app.EXAMPLES)


def test_the_real_app_generates_through_gradio_client():
    from gradio_client import Client
    with FakeMusicServer() as srv:
        demo = app.build_ui(MusicClient(srv.url), poll_s=0.01)
        import socket
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        _, url, _ = demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
        try:
            c = Client(url, verbose=False)
            audio, status, details = c.predict("[Verse]\nla la", "warm folk", 20, 9, None, api_name="/generate")
            assert str(audio).endswith(".flac") or (isinstance(audio, dict) and str(audio.get("path", "")).endswith(".flac"))
            assert "seed 9" in status
            assert srv.submitted[-1]["seed"] == 9
        finally:
            demo.close()


def test_zero_steps_means_the_servers_default_not_zero_steps():
    with FakeMusicServer() as srv:
        run(srv, steps=0)
        assert "num_inference_steps" not in srv.submitted[0]


def test_a_busy_server_is_shown_with_the_busy_message_and_not_the_generic_failure_text():
    with FakeMusicServer(queue_limit=-1) as srv:
        with pytest.raises(gr.Error) as e:
            run(srv)
        assert "could not make the clip" not in str(e.value)


def test_both_trigger_examples_use_the_trigger_phrase():
    assert sum("fiona crapple" in c for _, c, _, _ in app.EXAMPLES) == 2


def test_a_song_that_ended_early_says_so_in_plain_words():
    with FakeMusicServer(ended_audio_s=5.18966) as srv:
        _, status, details = run(srv, seconds=20)
    assert "5.2 s" in status and "before the 20 s" in status and "lyrics" in status
    assert "5.18966" not in status and "5.18966" not in details


def test_a_full_length_song_has_no_early_end_note_and_its_length_has_one_decimal():
    with FakeMusicServer() as srv:
        _, status, details = run(srv, seconds=20)
    assert "20.0 s of audio" in status and "before the" not in status


def test_a_song_that_ended_within_five_percent_of_the_request_has_no_note():
    with FakeMusicServer(ended_audio_s=19.5) as srv:
        _, status, _ = run(srv, seconds=20)
    assert "before the" not in status


def test_a_short_clip_without_the_servers_ended_flag_has_no_note():
    with FakeMusicServer(short_audio_s=5.0) as srv:
        _, status, _ = run(srv, seconds=20)
    assert "before the" not in status


def test_the_real_app_accepts_zero_steps_and_means_the_servers_default():
    # The browser sends a blank number box as 0 once the "More" panel has been touched; a minimum of 1 on the
    # component rejected that before run_generation could treat it as "use the default".
    from gradio_client import Client
    import socket
    with FakeMusicServer() as srv:
        demo = app.build_ui(MusicClient(srv.url), poll_s=0.01)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        _, url, _ = demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
        try:
            audio, status, _ = Client(url, verbose=False).predict("[Verse]\nla", "warm folk", 20, 9, 0, api_name="/generate")
            assert "seed 9" in status
            assert "num_inference_steps" not in srv.submitted[-1]
        finally:
            demo.close()


@pytest.mark.parametrize("steps", [101, 500])
def test_steps_above_the_servers_limit_are_a_plain_message(steps):
    with FakeMusicServer() as srv:
        with pytest.raises(gr.Error, match="steps"):
            run(srv, steps=steps)
        assert srv.submitted == []
