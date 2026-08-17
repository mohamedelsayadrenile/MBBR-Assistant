import json

from services.memory import RedisMemory


class FakePipeline:
    def __init__(self, redis: "FakeRedis") -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple[object, ...]]] = []

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def rpush(self, key: str, value: str) -> None:
        self._commands.append(("rpush", (key, value)))

    def ltrim(self, key: str, start: int, end: int) -> None:
        self._commands.append(("ltrim", (key, start, end)))

    def expire(self, key: str, ttl: int) -> None:
        self._commands.append(("expire", (key, ttl)))

    async def execute(self) -> None:
        for command, args in self._commands:
            if command == "rpush":
                key, value = args
                self._redis.data.setdefault(key, []).append(value)
            elif command == "ltrim":
                key, start, end = args
                values = self._redis.data.get(key, [])
                self._redis.data[key] = values[start : end + 1 if end != -1 else None]
            elif command == "expire":
                key, ttl = args
                self._redis.ttls[key] = ttl
                self._redis.expire_calls.append(key)


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}
        self.expire_calls: list[str] = []

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.data.get(key, [])
        return values[start : end + 1 if end != -1 else None]

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        assert transaction is True
        return FakePipeline(self)


def build_memory(redis: FakeRedis, max_messages: int = 12) -> RedisMemory:
    return RedisMemory(redis=redis, ttl_seconds=1800, max_messages=max_messages)  # type: ignore[arg-type]


async def test_memory_keeps_only_the_latest_n_messages() -> None:
    redis = FakeRedis()
    memory = build_memory(redis, max_messages=4)

    for index in range(10):
        await memory.append("conversation-1", "user", f"message-{index}")

    messages = await memory.load("conversation-1")

    assert len(messages) == 4
    assert [message["content"] for message in messages] == [
        "message-6",
        "message-7",
        "message-8",
        "message-9",
    ]


async def test_memory_ttl_is_1800_seconds() -> None:
    redis = FakeRedis()
    memory = build_memory(redis)

    await memory.append("conversation-1", "user", "عايز درجة حرارة الماية")

    assert redis.ttls["conversation:conversation-1"] == 1800


async def test_memory_ttl_is_refreshed_on_every_update() -> None:
    redis = FakeRedis()
    memory = build_memory(redis)

    await memory.append("conversation-1", "user", "عايز درجة حرارة الماية")
    await memory.append("conversation-1", "assistant", "أنهي جهاز؟")
    await memory.append("conversation-1", "user", "جهاز 2")

    # A sliding TTL means expire() is reissued on each write, not just the first.
    assert redis.expire_calls == ["conversation:conversation-1"] * 3
    assert redis.ttls["conversation:conversation-1"] == 1800


async def test_conversations_are_isolated_from_each_other() -> None:
    redis = FakeRedis()
    memory = build_memory(redis)

    await memory.append("conversation-1", "user", "رسالة أولى")
    await memory.append("conversation-2", "user", "رسالة تانية")

    first = await memory.load("conversation-1")
    second = await memory.load("conversation-2")

    assert [message["content"] for message in first] == ["رسالة أولى"]
    assert [message["content"] for message in second] == ["رسالة تانية"]
    assert set(redis.data) == {"conversation:conversation-1", "conversation:conversation-2"}


async def test_memory_round_trips_roles_and_arabic_content() -> None:
    redis = FakeRedis()
    memory = build_memory(redis)

    await memory.append("conversation-1", "user", "عايز درجة حرارة الماية دلوقتي")
    await memory.append("conversation-1", "assistant", "عندك جهاز 1 وجهاز 2، عايز أنهي جهاز؟")

    messages = await memory.load("conversation-1")

    assert messages == [
        {"role": "user", "content": "عايز درجة حرارة الماية دلوقتي"},
        {"role": "assistant", "content": "عندك جهاز 1 وجهاز 2، عايز أنهي جهاز؟"},
    ]


async def test_memory_never_stores_the_jwt() -> None:
    redis = FakeRedis()
    memory = build_memory(redis)

    await memory.append("conversation-1", "user", "عايز درجة حرارة الماية")
    await memory.append("conversation-1", "assistant", "أنهي جهاز؟")

    assert "runtime-jwt" not in json.dumps(redis.data, ensure_ascii=False)


async def test_memory_skips_unreadable_entries() -> None:
    redis = FakeRedis()
    memory = build_memory(redis)
    redis.data["conversation:conversation-1"] = [
        "not json at all",
        json.dumps({"role": "tool", "content": "{}"}, ensure_ascii=False),
        json.dumps({"role": "user", "content": "جهاز 2"}, ensure_ascii=False),
    ]

    messages = await memory.load("conversation-1")

    assert messages == [{"role": "user", "content": "جهاز 2"}]
