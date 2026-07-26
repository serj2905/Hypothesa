import asyncio

from bot import KeyedLockPool


def test_keyed_lock_serializes_same_participant_and_cleans_up() -> None:
    async def scenario() -> tuple[list[str], int]:
        pool = KeyedLockPool()
        events: list[str] = []

        async def worker(name: str) -> None:
            async with pool.hold(42):
                events.append(f"{name}:start")
                await asyncio.sleep(0)
                events.append(f"{name}:end")

        await asyncio.gather(worker("first"), worker("second"))
        return events, len(pool)

    events, remaining = asyncio.run(scenario())

    assert events in (
        ["first:start", "first:end", "second:start", "second:end"],
        ["second:start", "second:end", "first:start", "first:end"],
    )
    assert remaining == 0
