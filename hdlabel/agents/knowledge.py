"""Knowledge store of the agents: what was found, tried and learned, searchable by text (SQLite FTS5, one file).

Each record: kind (problem | cause | fix | attempt | tool_eval | research | verification | training | note), title,
text, tags (space separated: tool ids, element kinds, ...), status (open | done | failed | superseded), data (JSON:
metrics, references), area and time. Agents get the best matches for their task in their prompt (`context()`), so they
start from what is known instead of from zero; a record is never treated as final - a later record can supersede it
(e.g. a method that failed with an older model is tried again).

CLI: ./autolabel kb search "curb planter"   ./autolabel kb add --kind note --title ... --text ...   ./autolabel kb list
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

KINDS = ("problem", "cause", "fix", "attempt", "tool_eval", "research", "verification", "training", "spec", "note")
SCHEMA = """
create table if not exists rec (id integer primary key, kind text, title text, text text, tags text, status text,
                                data text, area text, time text, supersedes integer);
create virtual table if not exists rec_fts using fts5(title, text, tags, content='rec', content_rowid='id');
create trigger if not exists rec_ai after insert on rec begin
  insert into rec_fts(rowid, title, text, tags) values (new.id, new.title, new.text, new.tags); end;
create trigger if not exists rec_au after update on rec begin
  insert into rec_fts(rec_fts, rowid, title, text, tags) values ('delete', old.id, old.title, old.text, old.tags);
  insert into rec_fts(rowid, title, text, tags) values (new.id, new.title, new.text, new.tags); end;
"""


class Knowledge:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(self, kind, title, text="", tags=(), status="done", data=None, area="", supersedes=None) -> int:
        assert kind in KINDS, f"kind must be one of {KINDS}"
        tags = " ".join(tags) if not isinstance(tags, str) else tags
        cur = self.db.execute(
            "insert into rec(kind,title,text,tags,status,data,area,time,supersedes) values (?,?,?,?,?,?,?,?,?)",
            (kind, title, text, tags, status, json.dumps(data or {}), area, time.strftime("%Y-%m-%d %H:%M:%S"),
             supersedes),
        )  # fmt: skip
        if supersedes:
            self.db.execute("update rec set status='superseded' where id=?", (supersedes,))
        self.db.commit()
        return cur.lastrowid

    def update(self, rid: int, **kw):
        if "data" in kw:
            kw["data"] = json.dumps(kw["data"])
        sets = ", ".join(f"{k}=?" for k in kw)
        self.db.execute(f"update rec set {sets} where id=?", (*kw.values(), rid))
        self.db.commit()

    def get(self, rid: int) -> dict | None:
        r = self.db.execute("select * from rec where id=?", (rid,)).fetchone()
        return self._d(r) if r else None

    def search(self, query: str, kind=None, status=None, limit=10) -> list[dict]:
        """full-text match (any of the words), best first; kind / status filter."""
        words = [w for w in "".join(c if c.isalnum() or c == "_" else " " for c in query).split() if len(w) > 1]
        if not words:
            return self.list(kind=kind, status=status, limit=limit)
        q = " OR ".join(f'"{w}"' for w in words)
        sql = "select rec.* from rec_fts join rec on rec.id = rec_fts.rowid where rec_fts match ?"
        args = [q]
        if kind:
            sql += " and rec.kind = ?"
            args.append(kind)
        if status:
            sql += " and rec.status = ?"
            args.append(status)
        sql += " order by bm25(rec_fts) limit ?"
        return [self._d(r) for r in self.db.execute(sql, (*args, limit))]

    def list(self, kind=None, status=None, tag=None, limit=50) -> list[dict]:
        sql, args = "select * from rec where 1", []
        for col, v in (("kind", kind), ("status", status)):
            if v:
                sql += f" and {col} = ?"
                args.append(v)
        if tag:
            sql += " and (' ' || tags || ' ') like ?"
            args.append(f"% {tag} %")
        sql += " order by id desc limit ?"
        return [self._d(r) for r in self.db.execute(sql, (*args, limit))]

    def context(self, query: str, limit=8, chars=3000) -> str:
        """the best matching records as compact text for an agent prompt."""
        out, n = [], 0
        for r in self.search(query, limit=limit):
            line = f"[{r['id']} {r['kind']} {r['status']} {r['time'][:10]}] {r['title']}: {r['text']}"
            if r["data"]:
                line += f" data={json.dumps(r['data'])[:300]}"
            if n + len(line) > chars:
                break
            out.append(line)
            n += len(line)
        return "\n".join(out) or "(nothing recorded yet)"

    @staticmethod
    def _d(r) -> dict:
        d = dict(r)
        d["data"] = json.loads(d["data"] or "{}")
        return d


def main(argv=None, path=None):
    import argparse

    p = argparse.ArgumentParser(prog="autolabel kb")
    p.add_argument("action", choices=["search", "add", "list", "show"])
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--kind")
    p.add_argument("--status")
    p.add_argument("--tags", default="")
    p.add_argument("--title")
    p.add_argument("--text", default="")
    p.add_argument("--limit", type=int, default=20)
    a = p.parse_args(argv)
    kb = Knowledge(path)
    if a.action == "add":
        print(kb.add(a.kind or "note", a.title or a.query, a.text, a.tags, a.status or "done"))
        return
    if a.action == "show":
        print(json.dumps(kb.get(int(a.query)), indent=1, ensure_ascii=False))
        return
    rows = (
        kb.search(a.query, a.kind, a.status, a.limit)
        if a.action == "search"
        else kb.list(a.kind, a.status, limit=a.limit)
    )
    for r in rows:
        print(f"{r['id']:>4} {r['kind']:<12} {r['status']:<10} {r['time'][:10]} {r['title']}  [{r['tags']}]")
