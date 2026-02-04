"""Checkpointer gRPC servicer implementation.

This module implements the Checkpointer gRPC service, exposing the Python
checkpointer implementation to the Go server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc
from langgraph_grpc_common.proto.checkpointer_pb2_grpc import CheckpointerServicer

if TYPE_CHECKING:
    from google.protobuf.empty_pb2 import Empty  # type: ignore[import]
    from grpc import aio as grpc_aio  # type: ignore[import]
    from langgraph_grpc_common.proto import checkpointer_pb2


class CheckpointerServicerImpl(CheckpointerServicer):
    """Implementation of the Checkpointer gRPC service.

    This servicer delegates to the Python checkpointer implementation,
    allowing the Go server to use Python-based checkpoint storage.

    The checkpointer is obtained from the global checkpointer instance
    configured during server startup.
    """

    async def Put(
        self,
        request: checkpointer_pb2.PutRequest,
        context: grpc_aio.ServicerContext,
    ) -> checkpointer_pb2.PutResponse:
        """Store a checkpoint with its configuration and metadata."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Put not yet implemented")
        raise NotImplementedError("Put not yet implemented")

    async def PutWrites(
        self,
        request: checkpointer_pb2.PutWritesRequest,
        context: grpc_aio.ServicerContext,
    ) -> Empty:
        """Store intermediate writes linked to a checkpoint (pending writes)."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("PutWrites not yet implemented")
        raise NotImplementedError("PutWrites not yet implemented")

    async def GetCapabilities(
        self,
        request: Empty,
        context: grpc_aio.ServicerContext,
    ) -> checkpointer_pb2.Capabilities:
        """Return supported operations and batching limits."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("GetCapabilities not yet implemented")
        raise NotImplementedError("GetCapabilities not yet implemented")

    async def List(
        self,
        request: checkpointer_pb2.ListRequest,
        context: grpc_aio.ServicerContext,
    ) -> checkpointer_pb2.ListResponse:
        """Return checkpoints that match a given configuration and filter criteria."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("List not yet implemented")
        raise NotImplementedError("List not yet implemented")

    async def GetTuple(
        self,
        request: checkpointer_pb2.GetTupleRequest,
        context: grpc_aio.ServicerContext,
    ) -> checkpointer_pb2.GetTupleResponse:
        """Fetch a checkpoint tuple for a given configuration."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("GetTuple not yet implemented")
        raise NotImplementedError("GetTuple not yet implemented")

    async def DeleteThread(
        self,
        request: checkpointer_pb2.DeleteThreadRequest,
        context: grpc_aio.ServicerContext,
    ) -> Empty:
        """Delete all checkpoints and writes for a thread."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("DeleteThread not yet implemented")
        raise NotImplementedError("DeleteThread not yet implemented")

    async def DeleteForRuns(
        self,
        request: checkpointer_pb2.DeleteForRunsRequest,
        context: grpc_aio.ServicerContext,
    ) -> Empty:
        """Delete all checkpoints and writes for a set of runs (rollbacks)."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("DeleteForRuns not yet implemented")
        raise NotImplementedError("DeleteForRuns not yet implemented")

    async def CopyThread(
        self,
        request: checkpointer_pb2.CopyThreadRequest,
        context: grpc_aio.ServicerContext,
    ) -> Empty:
        """Copy checkpoint data from one thread to another."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("CopyThread not yet implemented")
        raise NotImplementedError("CopyThread not yet implemented")

    async def Prune(
        self,
        request: checkpointer_pb2.PruneRequest,
        context: grpc_aio.ServicerContext,
    ) -> Empty:
        """Delete checkpoints and related data for a set of threads."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Prune not yet implemented")
        raise NotImplementedError("Prune not yet implemented")
