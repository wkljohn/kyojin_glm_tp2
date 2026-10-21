"""
OpenAI-compatible server for the two-host expert-parallel layout (exllamav3/model/ep.py).

Rank 0 runs the normal GLM server (tools/glm/serve.py: create_app, chat template, tool calls, streaming) around an
EPEngine. Rank 1 has no HTTP listener: it runs the same model and replays, in the same order, every generator operation
rank 0 performs. Rank 0 owns admission, ordering, cancellation and sampling seeds, and sends one command per generator
operation over a dedicated gloo group (TCP, no NCCL watchdog while idle):

    ("job", ids, max_tokens, sampler, stops, seed)   enqueue one job
    ("step", cancel)                                 one iterate(), or cancel the job
    ("end",)                                         drop any leftover job, release deferred pages
    ("rebuild",)                                     fresh Generator (cache_salt change)
    ("check", digest)                                end-of-job token agreement; rank 1 answers with its own digest

One request runs at a time (--sessions 1). Speculation decisions depend on generator state only, which both ranks
build from the same command stream, so the collective schedules match. A digest mismatch marks the pair faulted and
every later request fails with 503 until restart.

    tools/glm/ep_launch.sh tools/glm/ep_serve.py --model <pack> --draft <DFlash2 dir> --host <ip> --port <port>
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serve   # noqa: E402  (applies nothing at import; SPEED_ENV is applied below)

for _k, _v in serve.SPEED_ENV.items():
    if _k != "EXL3_GLM_DYN":   # MTP drafter policy of the single-host server, not used with DFlash2
        os.environ.setdefault(_k, _v)
os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
# Timing-driven speculation decisions differ per rank and would desynchronize the EP collectives
os.environ["EXL3_SPEC_GATE"] = "0"

import torch   # noqa: E402
import torch.distributed as dist   # noqa: E402
from aiohttp import web   # noqa: E402


class EPEngine(serve.ResidentEngine):
    supports_prefix_reuse = True
    row_budget = 0

    def __init__(self, model_path: str, draft_path: str | None, max_ctx: int, num_draft: int,
                 dynamic: float | None, ngram: tuple | None):
        from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
        from exllamav3.model.ep import ep_state
        import exllamav3.generator.sampler.custom as sampler_custom

        torch.set_grad_enabled(False)
        self.torch = torch
        self.Job = Job
        self.Generator = Generator
        self.ep = ep_state()
        assert self.ep is not None, "ep_serve.py needs KYOJIN_EP_WORLD=2 (use tools/glm/ep_launch.sh)"
        self.rank = self.ep.rank
        self.ctl = dist.new_group(backend = "gloo", timeout = timedelta(days = 365))
        self.faulted = None
        # Rank 0 runs every generator and control-channel operation on this one thread, so the command order rank 1
        # replays is exactly the order rank 0 executed (including the end-of-request bookkeeping after the reply)
        self.pool = ThreadPoolExecutor(1, thread_name_prefix = "ep-gen")
        self._last = None
        # Sampling draws come from Job(seed=...) (job.rng); the module-level `random` of the sampler is also replaced by a
        # private stream reseeded per job from rank 0, so nothing else in the process can shift one rank's draws
        self.rng = random.Random(0)
        sampler_custom.random = self.rng

        self.sessions = 1
        self.dyn = None
        self.lookup = None
        self.lookup_stats = {}
        self.pec = None
        self.slot_store = None
        self.vision = None
        self.supports_vision = False
        self.emb_cache = serve.make_image_cache(0, 1, 1)
        self.emb_disk = None
        self.image_cache_format = "fp32"

        self.config = Config.from_directory(model_path)
        self.model = Model.from_config(self.config)
        self.tokenizer = Tokenizer.from_config(self.config)
        self.env_fingerprint = None
        cd = self.config.config_dict
        max_position = int(cd.get("max_position_embeddings") or cd.get("text_config", {}).get("max_position_embeddings") or 32768)
        self.ctx = ctx = min(max_position, max_ctx) // 256 * 256
        self.num_draft = self.max_draft = num_draft if draft_path else 0
        hist = max(1, self.num_draft, ngram[1] if ngram and draft_path else 0)
        self.cache = Cache(self.model, max_num_tokens = ctx + 4096, max_history = hist, max_batch_size = 1)
        self.model.load(device = "cuda:0", progressbar = False)
        self.draft_model = self.draft_cache = None
        if draft_path:
            dconfig = Config.from_directory(draft_path)
            self.draft_model = Model.from_config(dconfig)
            self.draft_cache = Cache(self.draft_model, max_num_tokens = ctx + 4096, max_history = hist, max_batch_size = 1)
            self.draft_model.load(device = "cuda:0", progressbar = False)
        self.dynamic = dynamic
        self.ngram = ngram if draft_path else None
        self.vocab_size = int(self.config.vocab_size or self.tokenizer.actual_vocab_size)
        self.greedy_generator = self._make_generator()

    def _make_generator(self):
        g = self.Generator(model = self.model, cache = self.cache, tokenizer = self.tokenizer,
                           draft_model = self.draft_model, draft_cache = self.draft_cache,
                           num_draft_tokens = self.num_draft or None, record_draft_stats = True,
                           dynamic_draft_tokens = self.dynamic is not None,
                           draft_confidence = self.dynamic if self.dynamic is not None else 0.6)
        if self.ngram:
            from ep_spec import install_dflash_lookup
            install_dflash_lookup(g, self.ngram, self.lookup_stats)
        return g

    # --- command stream ---------------------------------------------------------------------------------------------

    def send(self, cmd):
        box = [cmd]
        dist.broadcast_object_list(box, src = 0, group = self.ctl)
        return box[0]

    def recv(self):
        box = [None]
        dist.broadcast_object_list(box, src = 0, group = self.ctl)
        return box[0]

    def answer(self, value = None):
        """Rank 1 -> rank 0 reply (the end-of-job digest)."""
        box = [value]
        dist.broadcast_object_list(box, src = 1, group = self.ctl)
        return box[0]

    @staticmethod
    def make_sampler(spec):
        from exllamav3.generator.sampler import ComboSampler, GreedySampler
        temperature, top_p, kw = spec
        if temperature == 0:
            return ComboSampler(temperature = 0.0, **kw) if kw else GreedySampler()
        return ComboSampler(temperature = max(temperature, 1e-6), top_p = top_p, min_p = 0.0, **kw)

    def make_job(self, ids, max_tokens, spec, stops, seed):
        self.rng.seed(seed)
        ids = torch.tensor([ids], dtype = torch.long)
        return self.Job(input_ids = ids, max_new_tokens = max_tokens, sampler = self.make_sampler(spec),
                        stop_conditions = stops, seed = seed)

    def _rebuild_generator(self) -> None:
        import gc
        self.greedy_generator = None
        gc.collect()
        torch.cuda.empty_cache()
        self.greedy_generator = self._make_generator()

    def run(self, fn, *args):
        """Queue fn on the generator thread; awaitable from any event loop."""
        self._last = self.pool.submit(fn, *args)
        return asyncio.wrap_future(self._last)

    async def settle_reply(self) -> None:
        if self._last is not None:
            await asyncio.wrap_future(self.pool.submit(lambda: None))

    async def reset_prefix_cache(self) -> None:
        await self.run(lambda: (self.send(("rebuild",)), self._rebuild_generator()))

    # --- rank 0: one request ----------------------------------------------------------------------------------------

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                       top_p: float, stop: list[str], sampler_kw: dict | None = None,
                       cancel: asyncio.Event | None = None,
                       images: list | None = None, stats: dict | None = None, img_stats: dict | None = None,
                       prompt_ids: list[int] | None = None, ignore_eos: bool = False):
        if self.faulted:
            raise web.HTTPServiceUnavailable(text = f"expert-parallel pair faulted: {self.faulted}")
        if images:
            raise serve._bad("image input is not supported by the expert-parallel server")
        if prompt_ids is not None:
            ids = [int(t) for t in prompt_ids]
        else:
            ids = self.prompt_ids(prompt).reshape(-1).tolist()
        self.last_stats = {}
        if cancel is not None and cancel.is_set():
            return
        kw = dict(sampler_kw if sampler_kw is not None else serve.DEFAULT_SAMPLER_KW)
        spec = (float(temperature), float(top_p), kw)
        stops = ([] if ignore_eos else list(self.config.eos_token_id_list)) + list(stop)
        seed = random.SystemRandom().randrange(1 << 62)
        g = self.greedy_generator
        holder = []

        def start():
            holder.append(self.make_job(ids, max_tokens, spec, stops, seed))
            self.send(("job", ids, max_tokens, spec, stops, seed))
            g.enqueue(holder[0])

        def step():
            self.send(("step", False))
            return list(g.iterate())

        def stop_job():
            self.send(("step", True))
            g.cancel(holder[0])

        await self.run(start)
        toks = []
        finished = False
        try:
            while g.num_remaining_jobs():
                if cancel is not None and cancel.is_set():
                    await self.run(stop_job)
                    return
                batch = await self.run(step)
                for event in batch:
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    if event.get("token_ids") is not None:
                        toks += event["token_ids"].flatten().tolist()
                    text = event.get("text", "")
                    if text:
                        yield text
                    if event.get("eos"):
                        self.last_stats = event
                        if stats is not None:
                            stats.update(event)
                        finished = True
            torch.cuda.synchronize()
        finally:
            def end():
                self.send(("end",))
                if g.num_remaining_jobs():
                    g.clear_queue()
                g.flush_deferred()
                if finished:
                    digest = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]
                    self.send(("check", digest))
                    peer = self.answer(None)
                    if peer != digest:
                        self.faulted = f"token digest mismatch rank0 {digest} rank1 {peer}"
                        print(f"ep_serve: FAULT {self.faulted}", file = sys.stderr, flush = True)
            # the reply goes out now; the bookkeeping runs on the generator thread before anything queued later
            self._last = self.pool.submit(end)

    # --- rank 1: replay ---------------------------------------------------------------------------------------------

    def follow(self):
        print("ep_serve: rank 1 following rank 0", flush = True)
        job, toks = None, []
        while True:
            cmd = self.recv()
            op = cmd[0]
            g = self.greedy_generator
            if op == "job":
                _, ids, max_tokens, spec, stops, seed = cmd
                job, toks = self.make_job(ids, max_tokens, spec, stops, seed), []
                g.enqueue(job)
            elif op == "step":
                if cmd[1]:
                    g.cancel(job)
                elif g.num_remaining_jobs():
                    for event in g.iterate():
                        if event.get("token_ids") is not None:
                            toks += event["token_ids"].flatten().tolist()
                else:
                    print("ep_serve: WARNING step with no job (rank 0 ahead?)", file = sys.stderr, flush = True)
            elif op == "end":
                if g.num_remaining_jobs():
                    g.clear_queue()
                g.flush_deferred()
                job = None
            elif op == "check":
                digest = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]
                if digest != cmd[1]:
                    print(f"ep_serve: FAULT digest rank0 {cmd[1]} rank1 {digest}", file = sys.stderr, flush = True)
                self.answer(digest)
            elif op == "rebuild":
                self._rebuild_generator()
            elif op == "exit":
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--draft", default = None, help = "DFlash / DFlash2 drafter directory")
    ap.add_argument("--num-draft", type = int, default = 7)
    ap.add_argument("--dynamic-draft", type = float, default = 0.4, help = "draft confidence; 0 = fixed depth")
    ap.add_argument("--ngram", default = "2,7,16,3", help = "n-gram lookup ahead of the drafter; off = none")
    ap.add_argument("--model-id", default = "glm-5.3-exl3")
    ap.add_argument("--alias", action = "append", default = [],
                    help = "another model name accepted in requests (answered as --model-id); repeatable")
    ap.add_argument("--host", default = "127.0.0.1")
    ap.add_argument("--port", type = int, default = 8000)
    ap.add_argument("-c", "--max-ctx", type = int, default = 131072)
    ap.add_argument("--chat-template", default = None)
    ap.add_argument("--default-temperature", type = float, default = None)
    ap.add_argument("--default-top-p", type = float, default = None)
    ap.add_argument("--default-max-tokens", type = int, default = None)
    ap.add_argument("--default-reasoning-effort", choices = ("low", "medium", "high", "max"), default = None)
    args = ap.parse_args()

    serve.SERVE_DEFAULTS.update(serve.pack_sampling(args.model))
    serve.SERVE_DEFAULTS.update({k: v for k, v in (("temperature", args.default_temperature), ("top_p", args.default_top_p),
                                                   ("reasoning_effort", args.default_reasoning_effort),
                                                   ("max_tokens", args.default_max_tokens)) if v is not None})
    ngram = None if args.ngram.lower() in ("off", "0", "none") else tuple(int(x) for x in args.ngram.split(","))
    t0 = time.time()
    engine = EPEngine(args.model, args.draft, args.max_ctx, args.num_draft,
                      args.dynamic_draft or None, ngram)
    free, total = torch.cuda.mem_get_info()
    print(f"ep_serve: rank {engine.rank} loaded in {time.time() - t0:.0f} s, ctx {engine.ctx}, "
          f"draft {args.draft} x{engine.num_draft}, free {free / 2 ** 30:.1f} GiB", flush = True)
    if engine.rank != 0:
        engine.follow()
        return
    template = Path(args.chat_template or serve.DEFAULT_TEMPLATE).expanduser().read_text(encoding = "utf-8")
    if os.environ.get("EXL3_SERVE_WARMUP", "1") != "0":
        print(f"ep_serve: warm-up {serve.warmup(engine, template):.1f} s", flush = True)
        engine.pool.submit(lambda: None).result()
    app = serve.create_app(engine, args.model_id, template)
    if args.alias:
        aliases = set(args.alias)

        @web.middleware
        async def model_alias(request, handler):
            # clients configured for the previous deployment send its model name: answer them as this model
            if request.method == "POST" and request.content_type == "application/json":
                raw = await request.read()
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if isinstance(body, dict) and body.get("model") in aliases:
                    body["model"] = args.model_id
                    request._read_bytes = json.dumps(body).encode()   # what request.read()/json() return from now on
            return await handler(request)
        app.middlewares.insert(0, model_alias)
    print(f"ep_serve: listening on {args.host}:{args.port} as {args.model_id}", flush = True)
    web.run_app(app, host = args.host, port = args.port)


if __name__ == "__main__":
    main()
