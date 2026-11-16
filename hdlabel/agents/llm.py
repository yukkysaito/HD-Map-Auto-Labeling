"""Provider-independent access to the agent models (Claude, ChatGPT, ...): one call = a prompt + images -> text / JSON.

A role (verifier, annotator, researcher, engineer, ...) is mapped to a provider and a model in the config (`agents.roles`),
so a better model is used by changing one line, and the annotation and the verification can use different vendors.

Providers (add one by writing a function `_<name>(model, system, prompt, images, tools, timeout) -> Reply`):
  claude_cli    Claude Code in headless mode (`claude -p`); images are read with its Read tool; tools e.g. WebSearch
  codex_cli     OpenAI Codex CLI (`codex exec`), images attached with -i; tools: "web" enables web search
  openai        OpenAI API (Responses API, package `openai`, OPENAI_API_KEY)
  anthropic     Anthropic API (package `anthropic`, ANTHROPIC_API_KEY)
  replay        answers from a JSON file of {prompt_sha: text} (tests, offline re-runs)
Every call is cached by (provider, model, system, prompt, image contents): a re-run of the pipeline costs nothing and
gives the same answers (reproducible jobs). Every call is appended to the ledger (role, provider, model, cost, time),
and a run stops calling when the budget (`agents.budget_usd`) is spent.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field


class BudgetExceeded(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    provider: str
    model: str | None
    cost_usd: float = 0.0
    seconds: float = 0.0
    cached: bool = False
    meta: dict = field(default_factory=dict)

    def json(self):
        """the first JSON object / array in the text (models often wrap it in prose or ``` fences)."""
        return parse_json(self.text)


def parse_json(text: str):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i >= 0 and j > i:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in reply: {text[:300]!r}")


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- providers
def _run(cmd, timeout, cwd=None, stdin=None):
    """run a CLI in its own process group; on timeout the whole group is killed (CLIs start helper processes)."""
    import signal

    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE if stdin else
                             subprocess.DEVNULL, text=True, cwd=cwd, start_new_session=True)  # fmt: skip
    except FileNotFoundError as e:
        raise ProviderError(f"{cmd[0]} not installed") from e
    try:
        out, err = p.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise ProviderError(f"{os.path.basename(cmd[0])} timed out after {timeout} s") from e
    r = subprocess.CompletedProcess(cmd, p.returncode, out, err)
    if r.returncode:
        raise ProviderError(f"{cmd[0]} failed ({r.returncode}): {(r.stderr or r.stdout)[-500:]}")
    return r.stdout


def _claude_cli(model, system, prompt, images, tools, timeout, cwd=None):
    if images:
        prompt += "\n\nImages (read each with the Read tool):\n" + "\n".join(os.path.abspath(i) for i in images)
    allowed = list(tools) + (["Read"] if images and "Read" not in tools else [])
    cmd = [BIN.get("claude", "claude"), "-p", prompt, "--output-format", "json"]
    cmd += ["--model", model] if model else []
    cmd += ["--system-prompt", system] if system else []
    cmd += ["--allowedTools", ",".join(allowed)] if allowed else []
    cmd += ["--add-dir", *sorted({os.path.dirname(os.path.abspath(i)) for i in images})] if images else []
    out = json.loads(_run(cmd, timeout, cwd))
    if out.get("is_error"):
        raise ProviderError(f"claude: {out.get('result', '')[:300]}")
    return Reply(out.get("result", ""), "claude_cli", model, cost_usd=float(out.get("total_cost_usd") or 0),
                 meta=dict(turns=out.get("num_turns"), usage=out.get("usage")))  # fmt: skip


def _codex_cli(model, system, prompt, images, tools, timeout, cwd=None):
    import tempfile

    with tempfile.NamedTemporaryFile("r", suffix=".txt") as last:
        cmd = [
            BIN.get("codex", "codex"),
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--output-last-message",
            last.name,
        ]
        cmd += ["-m", model] if model else []
        cmd += ["-c", "tools.web_search=true"] if "web" in tools or "WebSearch" in tools else []
        for i in images:
            cmd += ["-i", os.path.abspath(i)]
        cmd += ["--", (system + "\n\n" if system else "") + prompt]
        out = _run(cmd, timeout, cwd)
        text = open(last.name).read().strip()
    m = re.search(r"tokens used\s*\n?\s*([\d,]+)", out)
    # the CLI runs on a subscription: no price per call; tokens are recorded (cost_usd from agents.price_per_mtok)
    return Reply(text, "codex_cli", model, meta=dict(tokens=int(m.group(1).replace(",", "")) if m else None))


def _openai(model, system, prompt, images, tools, timeout, cwd=None):
    from openai import OpenAI  # optional dependency

    content = [{"type": "input_text", "text": prompt}]
    for i in images:
        b64 = base64.b64encode(open(i, "rb").read()).decode()
        content.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"})
    kw = dict(tools=[{"type": "web_search"}]) if "web" in tools or "WebSearch" in tools else {}
    r = OpenAI(timeout=timeout).responses.create(
        model=model or "gpt-5", instructions=system or None, input=[{"role": "user", "content": content}], **kw
    )
    u = r.usage
    return Reply(r.output_text, "openai", model, meta=dict(tokens=u.total_tokens if u else None))


def _anthropic(model, system, prompt, images, tools, timeout, cwd=None):
    import anthropic  # optional dependency

    content = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(open(i, "rb").read()).decode(),
            },
        }  # fmt: skip
        for i in images
    ] + [{"type": "text", "text": prompt}]
    kw = dict(tools=[{"type": "web_search_20250305", "name": "web_search"}]) if "WebSearch" in tools else {}
    r = anthropic.Anthropic(timeout=timeout).messages.create(
        model=model or "claude-sonnet-5-5", max_tokens=8000, system=system or anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": content}], **kw,
    )  # fmt: skip
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return Reply(text, "anthropic", model, meta=dict(tokens=r.usage.input_tokens + r.usage.output_tokens))


BIN = {}  # executables of the CLI providers (config agents.bin: {claude: /path/to/claude, codex: ...})
PROVIDERS = dict(claude_cli=_claude_cli, codex_cli=_codex_cli, openai=_openai, anthropic=_anthropic)


# ---------------------------------------------------------------- the client
class Agents:
    """the configured roles of one run: ask(role, prompt, images) with cache, ledger and budget.

    cfg = config `agents` (configs/default.yaml); root = directory for cache/ and ledger.jsonl."""

    def __init__(self, cfg: dict, root: str):
        self.cfg, self.root = cfg or {}, root
        self.roles = self.cfg.get("roles", {})
        self.budget = float(self.cfg.get("budget_usd") or 0)  # 0 = no limit
        self.price = self.cfg.get("price_per_mtok") or {}  # {provider: usd per 1M tokens} for token-only providers
        self.lock = threading.Lock()
        self.spent = 0.0
        BIN.update({k: os.path.expanduser(v) for k, v in (self.cfg.get("bin") or {}).items()})
        os.makedirs(f"{root}/cache", exist_ok=True)

    def model_of(self, role: str) -> dict:
        r = self.roles.get(role) or self.roles.get("default") or {}
        return dict(provider=r.get("provider", "none"), model=r.get("model"), timeout=int(r.get("timeout", 900)))

    def available(self, role: str) -> bool:
        """is the role's provider usable here (CLI installed / API key set)?"""
        import shutil

        p = self.model_of(role)["provider"]
        need = dict(claude_cli=lambda: shutil.which(BIN.get("claude", "claude")), codex_cli=lambda: shutil.which(BIN.get("codex", "codex")),
                    openai=lambda: os.environ.get("OPENAI_API_KEY"), anthropic=lambda: os.environ.get("ANTHROPIC_API_KEY"),
                    replay=lambda: self.cfg.get("replay"))  # fmt: skip
        return bool(p in need and need[p]())

    def ask(self, role, prompt, images=(), system="", tools=(), cache=True, cwd=None) -> Reply:
        m = self.model_of(role)
        prov = m["provider"]
        imgs = [open(i, "rb").read() for i in images]
        key = _sha(prov, m["model"], system, prompt, sorted(tools), *[hashlib.sha256(b).digest() for b in imgs])
        cf = f"{self.root}/cache/{key[:2]}/{key}.json"
        if cache and os.path.exists(cf):
            d = json.load(open(cf))
            rep = Reply(**{k: v for k, v in d.items() if k in Reply.__dataclass_fields__})
            rep.cached = True
            self._log(role, rep, key)
            return rep
        if prov == "replay":
            table = json.load(open(self.cfg["replay"]))
            text = table.get(key) or table.get(_sha(prompt)) or table.get("*")
            if text is None:
                raise ProviderError(f"replay: no answer for {key[:12]}")
            rep = Reply(text if isinstance(text, str) else json.dumps(text), "replay", m["model"])
        else:
            if prov not in PROVIDERS:
                raise ProviderError(f"role {role}: no provider configured (agents.roles.{role}.provider)")
            with self.lock:
                if self.budget and self.spent >= self.budget:
                    raise BudgetExceeded(f"agent budget {self.budget} USD spent")
            t = time.time()
            try:
                rep = PROVIDERS[prov](m["model"], system, prompt, list(images), tuple(tools), m["timeout"], cwd)
            except ProviderError:
                raise
            except Exception as e:  # SDK errors (auth, rate limit, network, missing package): one kind for the callers
                raise ProviderError(f"{prov}: {type(e).__name__}: {str(e)[:300]}") from e
            rep.seconds = round(time.time() - t, 1)
            if not rep.cost_usd and rep.meta.get("tokens") and prov in self.price:
                rep.cost_usd = rep.meta["tokens"] / 1e6 * float(self.price[prov])
        with self.lock:
            self.spent += rep.cost_usd
        if cache:
            os.makedirs(os.path.dirname(cf), exist_ok=True)
            json.dump(asdict(rep), open(cf, "w"))
        self._log(role, rep, key)
        return rep

    def _log(self, role, rep: Reply, key):
        with self.lock, open(f"{self.root}/ledger.jsonl", "a") as f:
            f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, provider=rep.provider,
                                    model=rep.model, cost_usd=round(rep.cost_usd, 5), seconds=rep.seconds,
                                    cached=rep.cached, key=key[:16])) + "\n")  # fmt: skip


def ledger_summary(root: str) -> dict:
    """cost / calls per role and provider from ledger.jsonl."""
    out = {}
    p = f"{root}/ledger.jsonl"
    for line in open(p) if os.path.exists(p) else []:
        e = json.loads(line)
        k = f"{e['role']}:{e['provider']}:{e.get('model')}"
        s = out.setdefault(k, dict(calls=0, cached=0, cost_usd=0.0, seconds=0.0))
        s["calls"] += 1
        s["cached"] += bool(e["cached"])
        s["cost_usd"] = round(s["cost_usd"] + (0 if e["cached"] else e["cost_usd"]), 4)
        s["seconds"] = round(s["seconds"] + (0 if e["cached"] else e["seconds"]), 1)
    return out
