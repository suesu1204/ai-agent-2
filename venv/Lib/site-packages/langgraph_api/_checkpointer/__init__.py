from collections.abc import Callable
from typing import Any

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver

from langgraph_api._checkpointer import _adapter
from langgraph_api._checkpointer.protocol import CheckpointerProtocol

logger = structlog.stdlib.get_logger(__name__)


async def get_checkpointer(
    *,
    conn: Any | None = None,
    unpack_hook: Callable[[int, bytes], Any] | None = None,
    use_direct_connection: bool = False,
) -> CheckpointerProtocol:
    return await _adapter.get_checkpointer(
        conn=conn,
        unpack_hook=unpack_hook,
        use_direct_connection=use_direct_connection,
    )


async def start_checkpointer() -> None:
    """Start the checkpointer resources."""
    # If the custom checkpointer is provided, it will be started here / enter the stack.
    checkpointer = await get_checkpointer()
    if not isinstance(checkpointer, (BaseCheckpointSaver, CheckpointerProtocol)):
        # Should only occur if we are using a custom checkpointer.
        logger.warning(
            "Custom checkpointer does not implement the expected checkpointer protocol; "
            "expected to be a subclass of BaseCheckpointSaver and export the proper "
            "async methods: aget_tuple/aput/aput_writes. "
            "Check your `checkpointer.path` target and ensure it returns a "
            "BaseCheckpointSaver instance or equivalent."
        )


async def exit_checkpointer() -> None:
    """Close the checkpointer resources."""
    # This will close the exit stack if a given custom checkpointer is provided.
    await _adapter.exit_checkpointer()


__all__ = [
    "exit_checkpointer",
    "get_checkpointer",
    "start_checkpointer",
]
