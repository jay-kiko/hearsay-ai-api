"""Access-code uses: a job that delivers nothing gives its use back, once."""
import aiosqlite
import pytest

from app.code_store import CodeStore


@pytest.fixture
async def store(tmp_path):
    s = CodeStore(str(tmp_path / "codes.db"))
    await s.connect()
    yield s
    await s.close()


async def _remaining(store: CodeStore, code: str) -> int:
    return next(r.uses_remaining for r in await store.list_all() if r.code == code)


async def test_failed_job_is_refunded_once(store):
    [code] = await store.mint(1, uses=1)
    assert await store.redeem(code, "j1")
    assert await store.status(code) == "exhausted"
    assert await store.refund("j1")
    assert not await store.refund("j1")  # never twice
    assert await _remaining(store, code) == 1


async def test_finished_job_is_never_refunded(store):
    [code] = await store.mint(1, uses=2)
    await store.redeem(code, "j1")
    await store.finish("j1")
    assert not await store.refund("j1")
    assert await _remaining(store, code) == 1


async def test_only_stale_unfinished_jobs_are_refunded(store):
    [code] = await store.mint(1, uses=4)
    for job in ("done", "lost", "failed", "mine"):
        await store.redeem(code, job)
    await store.finish("done")
    await store.refund("failed")
    # Young jobs may belong to another process mid-deploy: never refunded.
    assert await store.refund_stale(older_than_seconds=3600) == 0
    # Once old enough, "lost" is refunded; this process's running job isn't.
    assert await store.refund_stale(older_than_seconds=-1, running={"mine"}) == 1
    assert await _remaining(store, code) == 2


async def test_redemptions_from_before_outcomes_existed_are_not_refunded(tmp_path):
    path = tmp_path / "old.db"
    async with aiosqlite.connect(path) as db:
        await db.executescript(
            "CREATE TABLE access_codes (code TEXT PRIMARY KEY, created_at TEXT NOT NULL, uses_total INTEGER NOT NULL,"
            " uses_remaining INTEGER NOT NULL, prompt_calls INTEGER NOT NULL DEFAULT 0);"
            "CREATE TABLE code_redemptions (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL,"
            " job_id TEXT NOT NULL, redeemed_at TEXT NOT NULL);"
            "INSERT INTO access_codes VALUES ('AAAA-BBBB', 'x', 1, 0, 0);"
            "INSERT INTO code_redemptions (code, job_id, redeemed_at) VALUES ('AAAA-BBBB', 'old', 'x');"
        )
        await db.commit()
    s = CodeStore(str(path))
    await s.connect()
    assert await s.refund_stale(older_than_seconds=-1) == 0
    assert await s.status("AAAA-BBBB") == "exhausted"
    await s.close()


async def test_a_half_finished_outcome_migration_is_completed_on_the_next_boot(tmp_path):
    """The column was added but the crash came before old rows were marked."""
    path = tmp_path / "half.db"
    async with aiosqlite.connect(path) as db:
        await db.executescript(
            "CREATE TABLE access_codes (code TEXT PRIMARY KEY, created_at TEXT NOT NULL, uses_total INTEGER NOT NULL,"
            " uses_remaining INTEGER NOT NULL, prompt_calls INTEGER NOT NULL DEFAULT 0, revoked INTEGER NOT NULL DEFAULT 0);"
            "CREATE TABLE code_redemptions (id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL,"
            " job_id TEXT NOT NULL, redeemed_at TEXT NOT NULL, outcome TEXT);"
            "INSERT INTO access_codes VALUES ('AAAA-BBBB', 'x', 1, 0, 0, 0);"
            "INSERT INTO code_redemptions (code, job_id, redeemed_at) VALUES ('AAAA-BBBB', 'old', '2020-01-01T00:00:00+00:00');"
        )
        await db.commit()
    s = CodeStore(str(path))
    await s.connect()
    assert await s.refund_stale(older_than_seconds=0) == 0
    await s.close()
    s = CodeStore(str(path))  # and a second boot doesn't re-mark new rows
    await s.connect()
    [code] = await s.mint(1)
    await s.redeem(code, "new")
    await s.close()
    s = CodeStore(str(path))
    await s.connect()
    assert await s.refund_stale(older_than_seconds=-1) == 1
    await s.close()
