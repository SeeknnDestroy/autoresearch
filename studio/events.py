from __future__ import annotations

import threading
from typing import Any


class EventBroker:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._events: list[dict[str, Any]] = []
        self._next_id = 1

    def publish(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            event = {
                "id": self._next_id,
                "type": event_type,
                "payload": payload,
            }
            self._next_id += 1
            self._events.append(event)
            if len(self._events) > 200:
                self._events = self._events[-200:]
            self._condition.notify_all()
            return event

    def wait_for_events(self, after_id: int, timeout: float = 15.0) -> list[dict[str, Any]]:
        with self._condition:
            events = [event for event in self._events if event["id"] > after_id]
            if events:
                return events
            self._condition.wait(timeout=timeout)
            return [event for event in self._events if event["id"] > after_id]
