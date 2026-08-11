"""Multi-use access codes gate the demo without needing accounts.

There's still no login and no user table — a code is just a redemption
ticket, now good for `uses_total` redemptions instead of exactly one. Codes
are minted ahead of time (via /admin/codes or the mint_codes script), handed
out, and redeemed each time someone actually launches a job (POST
/api/analysis). Backed by SQLite: one table doesn't need a real RDBMS
service, but it does need real transactions — redeem() is a single
conditional UPDATE decrementing uses_remaining, atomic without any locking of
our own, and correct even across multiple worker processes on the same file
(unlike a hand-rolled read-modify-write over a JSON file, which only
serializes within one process).
"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiosqlite

from app.models import CamelModel

_ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"  # no 0/O/1/I/L — easy to read out loud
CodeStatus = Literal["unknown", "exhausted", "valid"]
CallStatus = Literal["unknown", "exhausted", "rate_limited", "ok"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS access_codes (
    code TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    uses_total INTEGER NOT NULL,
    uses_remaining INTEGER NOT NULL,
    prompt_calls INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS code_redemptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL REFERENCES access_codes(code),
    job_id TEXT NOT NULL,
    redeemed_at TEXT NOT NULL
);
"""


def _generate_code() -> str:
    chars = "".join(secrets.choice(_ALPHABET) for _ in range(8))
    return f"{chars[:4]}-{chars[4:]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CodeRecord(CamelModel):
    code: str
    created_at: str
    uses_total: int
    uses_remaining: int
    prompt_calls: int


class CodeStore:
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._path)
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("CodeStore.connect() must be called before use")
        return self._conn

    async def mint(self, count: int, uses: int = 1) -> list[str]:
        new_codes: list[str] = []
        for _ in range(count):
            while True:
                code = _generate_code()
                try:
                    await self._db.execute(
                        "INSERT INTO access_codes (code, created_at, uses_total, uses_remaining) "
                        "VALUES (?, ?, ?, ?)",
                        (code, _now(), uses, uses),
                    )
                except aiosqlite.IntegrityError:
                    continue  # collision on the code itself — draw another
                new_codes.append(code)
                break
        await self._db.commit()
        return new_codes

    async def status(self, code: str) -> CodeStatus:
        async with self._db.execute(
            "SELECT uses_remaining FROM access_codes WHERE code = ?", (code,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return "unknown"
        return "valid" if row[0] > 0 else "exhausted"

    async def redeem(self, code: str, job_id: str) -> bool:
        """Atomically spends one use. Returns False if unknown or exhausted."""
        cursor = await self._db.execute(
            "UPDATE access_codes SET uses_remaining = uses_remaining - 1 WHERE code = ? AND uses_remaining > 0",
            (code,),
        )
        if cursor.rowcount == 0:
            await self._db.rollback()
            return False
        await self._db.execute(
            "INSERT INTO code_redemptions (code, job_id, redeemed_at) VALUES (?, ?, ?)",
            (code, job_id, _now()),
        )
        await self._db.commit()
        return True

    async def register_prompt_call(self, code: str, max_calls: int) -> CallStatus:
        """Soft throttle on the unconsumed /api/prompts path: capped, not spent."""
        async with self._db.execute(
            "SELECT uses_remaining, prompt_calls FROM access_codes WHERE code = ?", (code,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return "unknown"
        uses_remaining, prompt_calls = row
        if uses_remaining <= 0:
            return "exhausted"
        if prompt_calls >= max_calls:
            return "rate_limited"

        cursor = await self._db.execute(
            "UPDATE access_codes SET prompt_calls = prompt_calls + 1 WHERE code = ? AND prompt_calls < ?",
            (code, max_calls),
        )
        await self._db.commit()
        return "ok" if cursor.rowcount > 0 else "rate_limited"

    async def list_all(self) -> list[CodeRecord]:
        async with self._db.execute(
            "SELECT code, created_at, uses_total, uses_remaining, prompt_calls "
            "FROM access_codes ORDER BY created_at"
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            CodeRecord(code=r[0], created_at=r[1], uses_total=r[2], uses_remaining=r[3], prompt_calls=r[4])
            for r in rows
        ]


_store: CodeStore | None = None


def get_code_store() -> CodeStore:
    global _store
    if _store is None:
        from app.config import get_settings

        _store = CodeStore(get_settings().codes_db_path)
    return _store
