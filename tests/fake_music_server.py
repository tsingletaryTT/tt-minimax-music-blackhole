"""A stdlib fake of the MiniMax-Music3 TT server's job API, for tests. No chip, no network beyond loopback.

Behaviour is set through the FakeMusicServer attributes before a request: `queue_limit` jobs may be waiting
(a further submit returns 429, like the real server), a job reports `steps_before_done` running polls first, and
`fail_with` makes every job end in status "error".
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeMusicServer:
    def __init__(self, queue_limit=9, steps_before_done=2, fail_with=None, model="fiona-crapple-merged"):
        self.queue_limit, self.steps_before_done, self.fail_with, self.model = queue_limit, steps_before_done, fail_with, model
        self.jobs = {}             # id -> dict(body, polls)
        self.submitted = []        # request bodies, in order
        self.cancelled = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj=None, raw=None, ctype="application/json"):
                data = raw if raw is not None else json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                p = self.path
                if p == "/v1/health":
                    return self._send(200, {"status": "ok", "model": outer.model, "mesh": "1x1", "device_name": "P150",
                                            "busy": False, "queue_depth": 0, "error": None})
                if p == "/v1/models":
                    return self._send(200, {"object": "list", "data": [{"id": outer.model}]})
                if p.startswith("/v1/music/jobs/") and p.endswith("/audio"):
                    jid = p.split("/")[4]
                    if jid not in outer.jobs:
                        return self._send(404, {"detail": "no such job"})
                    return self._send(200, raw=b"fLaC-fake-audio-" + jid.encode(), ctype="audio/flac")
                if p.startswith("/v1/music/jobs/"):
                    jid = p.split("/")[4]
                    job = outer.jobs.get(jid)
                    if job is None:
                        return self._send(404, {"detail": "no such job"})
                    job["polls"] += 1
                    frames = job["body"].get("audio_duration", 20) * 25
                    if outer.fail_with:
                        return self._send(200, {"id": jid, "status": "error", "error": outer.fail_with})
                    if job["polls"] <= outer.steps_before_done:
                        return self._send(200, {"id": jid, "status": "running", "frames_done": int(frames * job["polls"] / (outer.steps_before_done + 1)),
                                                "frames_max": int(frames), "stage": "semantic"})
                    return self._send(200, {"id": jid, "status": "done", "frames_done": int(frames), "frames_max": int(frames),
                                            "stage": "done", "audio_url": f"/v1/music/jobs/{jid}/audio",
                                            "stats": {"audio_s": job["body"].get("audio_duration", 20), "total_s": 65.0, "llm_s": 18.0,
                                                      "depth_s": 19.1, "denoise_s": 23.9, "vocoder_s": 14.6, "seed": job["body"].get("seed")}})
                return self._send(404, {"detail": "not found"})

            def do_POST(self):
                if self.path != "/v1/music/jobs":
                    return self._send(404, {"detail": "not found"})
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                waiting = sum(1 for j in outer.jobs.values())
                if waiting > outer.queue_limit:
                    return self._send(429, {"detail": "queue full"})
                if not body.get("input") or not body.get("instructions"):
                    return self._send(422, {"detail": "input and instructions are required"})
                jid = f"job{len(outer.jobs) + 1:03d}"
                outer.jobs[jid] = {"body": body, "polls": 0}
                outer.submitted.append(body)
                return self._send(200, {"id": jid, "status": "queued", "queue_position": waiting})

            def do_DELETE(self):
                jid = self.path.split("/")[4] if self.path.startswith("/v1/music/jobs/") else None
                outer.cancelled.append(jid)
                return self._send(200, {"id": jid, "status": "cancelled"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()
