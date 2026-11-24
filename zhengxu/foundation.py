"""Account boundaries, operation events and service errors shared across modules."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import os
from uuid import UUID


@dataclass(frozen=True)
class AccountScope:
    local_id: int
    cloud_id: UUID | None

    @classmethod
    def from_session(cls, account):
        if not isinstance(account, dict) or type(account.get("id")) is not int or account["id"] <= 0:
            raise ValueError("Invalid account scope")
        return cls(account["id"], cloud_identifier(account.get("supabase_user_id")))

    def matches_cloud(self, configured_id):
        return self.cloud_id is not None and self.cloud_id == cloud_identifier(configured_id)


def cloud_identifier(value):
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            pass
    return None


def feature_enabled(name):
    return os.environ.get(name) == "1"


class Failure(str, Enum):
    AUTH = "authentication"
    FORBIDDEN = "permission"
    UNAVAILABLE = "unavailable"


class ServiceReadError(RuntimeError):
    def __init__(self, status=None):
        self.reason = (
            Failure.AUTH if status == 401 else Failure.FORBIDDEN if status == 403
            else Failure.UNAVAILABLE
        )
        super().__init__(self.reason.value)


def failure_message(error, fallback):
    if isinstance(error, ServiceReadError):
        if error.reason is Failure.AUTH:
            return "登入狀態已失效。請重新登入後再試。"
        if error.reason is Failure.FORBIDDEN:
            return "目前帳號無法讀取這項資料。"
    return fallback


ACTIVITY_KEY = "platform_module_activity"
ACTIVITY_LIMIT = 30


@dataclass(frozen=True)
class ModuleEvent:
    occurred_at: datetime
    source: str | None
    target: str
    status: str


def _activity(session, scope):
    state = session.get(ACTIVITY_KEY)
    if not isinstance(state, dict) or state.get("owner") != scope:
        state = {"owner": scope, "current": None, "events": []}
        session[ACTIVITY_KEY] = state
    return state


def record_module_event(session, scope, target, status="已開啟"):
    state = _activity(session, scope)
    current = state["current"]
    if current == (target, status):
        return
    source = current[0] if current else None
    state["current"] = (target, status)
    state["events"] = [
        ModuleEvent(datetime.now(timezone.utc), source, target, status),
        *state["events"][:ACTIVITY_LIMIT - 1],
    ]


def module_events(session, scope):
    return tuple(_activity(session, scope)["events"])
