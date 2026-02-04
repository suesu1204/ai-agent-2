"""Encryption gRPC servicer implementation.

This module implements the Encryption gRPC service, exposing the Python
custom encryption implementation to the Go server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import grpc
import orjson
import structlog
from langgraph_grpc_common.proto.encryption_pb2 import (
    DecryptResponse,
    EncryptResponse,
)
from langgraph_grpc_common.proto.encryption_pb2_grpc import EncryptionServicer
from langgraph_sdk import EncryptionContext

from langgraph_api.encryption.shared import get_encryption

if TYPE_CHECKING:
    from grpc import aio as grpc_aio  # type: ignore[import]

    from langgraph_api.encryption.custom import ModelType

logger = structlog.stdlib.get_logger(__name__)


def _parse_metadata(metadata: dict[str, bytes]) -> dict[str, Any]:
    """Parse metadata from proto map<string, bytes> to dict.

    Args:
        metadata: Map of string keys to JSON-encoded bytes values

    Returns:
        Dict with parsed JSON values

    Raises:
        ValueError: If any value is not valid JSON
    """
    result = {}
    for k, v in metadata.items():
        if v:
            try:
                result[k] = orjson.loads(v)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse metadata value for key '{k}' as JSON: {e}"
                ) from e
        else:
            result[k] = None
    return result


def _build_encryption_context(
    model: str | None,
    field: str | None,
    metadata: dict[str, bytes],
) -> EncryptionContext:
    """Build an EncryptionContext from proto fields.

    Args:
        model: Model type (e.g., "thread", "run")
        field: Field name (e.g., "metadata", "kwargs")
        metadata: Proto metadata map

    Returns:
        EncryptionContext for SDK encryption handlers
    """
    return EncryptionContext(
        model=model or None,
        field=field or None,
        metadata=_parse_metadata(metadata),
    )


class EncryptionServicerImpl(EncryptionServicer):
    """Implementation of the Encryption gRPC service.

    This servicer delegates to the Python custom encryption implementation,
    allowing the Go server to use Python-based encryption when custom
    encryption is configured.

    For JSON operations, uses the JsonEncryptionWrapper which handles:
    - Key preservation validation
    - Encryption context marker storage
    - Migration routing between AES and custom encryption
    """

    async def EncryptJSON(
        self,
        request,
        context: grpc_aio.ServicerContext,
    ) -> EncryptResponse:
        """Encrypt JSON data using the configured encryption.

        Uses the JsonEncryptionWrapper for proper key validation and context storage.
        """
        try:
            encryption_instance = get_encryption()
            if encryption_instance is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("No encryption configured")
                raise RuntimeError("No encryption configured")

            data: dict[str, Any] = orjson.loads(request.data)

            model_type: ModelType | None = request.context.model or None
            field = request.context.field or None
            ctx = _build_encryption_context(
                model_type, field, dict(request.context.metadata)
            )

            encryptor = encryption_instance.get_json_encryptor(model_type)
            if encryptor is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                model_err = f"No encryptor configured for model type: {model_type}"
                context.set_details(model_err)
                raise RuntimeError(model_err)

            encrypted = await encryptor(ctx, data)
            encrypted_bytes = orjson.dumps(encrypted)

            return EncryptResponse(data=encrypted_bytes)

        except Exception as e:
            await logger.aerror("EncryptJSON failed", error=str(e), exc_info=True)
            if context.code() is None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Encryption failed: {e}")
            raise

    async def DecryptJSON(
        self,
        request,
        context: grpc_aio.ServicerContext,
    ) -> DecryptResponse:
        """Decrypt JSON data using the configured encryption.

        Uses the JsonEncryptionWrapper which routes to the appropriate decryptor
        based on the encryption context marker (handles AES migration).
        The encryption context is extracted from __encryption_context__ in the data
        itself. The optional model and field parameters are used for routing.
        """
        try:
            encryption_instance = get_encryption()
            if encryption_instance is None:
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("No encryption configured")
                raise RuntimeError("No encryption configured")

            data: dict[str, Any] = orjson.loads(request.data)

            # Model and field are optional, used for decryptor selection/routing
            model_type: ModelType | None = request.model or None
            field = request.field or None

            # Get the decryptor from the wrapper (handles routing based on
            # __encryption_context__ marker in the data)
            decryptor = encryption_instance.get_json_decryptor(model_type)
            if decryptor is None:
                model_err = f"No decryptor configured for model type: {model_type}"
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(model_err)
                raise RuntimeError(model_err)

            ctx = EncryptionContext(model=model_type, field=field, metadata={})
            decrypted = await decryptor(ctx, data)
            decrypted_bytes = orjson.dumps(decrypted)

            return DecryptResponse(data=decrypted_bytes)

        except Exception as e:
            await logger.aerror("DecryptJSON failed", error=str(e), exc_info=True)
            if context.code() is None:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(f"Decryption failed: {e}")
            raise
