"""Mint multi-use access codes offline, without needing ADMIN_SECRET/HTTP.

Usage (inside the running container):
    docker compose exec api python -m app.scripts.mint_codes 10        # 10 codes, 1 use each
    docker compose exec api python -m app.scripts.mint_codes 10 5      # 10 codes, 5 uses each
"""
import asyncio
import sys

from app.code_store import get_code_store


async def main(count: int, uses: int) -> None:
    store = get_code_store()
    await store.connect()
    try:
        codes = await store.mint(count, uses=uses)
        for code in codes:
            print(code)
    finally:
        await store.close()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    u = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    asyncio.run(main(n, u))
