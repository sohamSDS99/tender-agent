"""
NexusClient — HTTP client for the Nexus AMS API.

Provides:
- register(): Send agent config + get agentId
- heartbeat(): Periodic liveness ping
- track(): Emit a metric event (batched, flushed every 5s or 50 events)
- reply(thread_id, content): POST a reply to /api/chat/agent-reply
- stream(thread_id, generator): POST tokens to /api/chat/stream-token
- poll_inbox(handler): GET from /api/agents/{name}/inbox every 2s
- on_message(): Register a handler for incoming messages
- download_file(minio_key, dest_path): Download a file from AMS storage
- upload_file(file_path): Upload a file to AMS and get attachment metadata
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Generator, Iterator, Optional

import httpx

from .types import (
    AgentConfig,
    ChatMessage,
    HeartbeatPayload,
    IngestResponse,
    MetricEvent,
    RegisterResponse,
)

logger = logging.getLogger(__name__)

# Batch settings
_FLUSH_INTERVAL_SECONDS = 5
_FLUSH_BATCH_SIZE = 50
_INBOX_POLL_INTERVAL = 2.0  # seconds


class NexusClient:
    """
    HTTP client for the Nexus AMS tRPC API.

    Usage::

        client = NexusClient(base_url="http://localhost:3000", agent_id="<uuid>", agent_name="my-agent")
        client.register(AgentConfig(name="my-agent", display_name="My Agent"))
        client.heartbeat()
        client.track("latency", 142.5)

        # Reply to a chat thread
        client.reply("thread-uuid", "Here is my analysis...")

        # Stream tokens to a thread
        def my_generator():
            for word in ["Hello", " ", "world"]:
                yield word
        client.stream("thread-uuid", "msg-uuid", my_generator())

        # Poll inbox for incoming messages
        def handle_message(msg):
            print(f"Got message: {msg['content']}")
        client.poll_inbox(handle_message)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.agent_name = agent_name
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._build_headers(),
        )
        self._metric_buffer: list[MetricEvent] = []
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._message_handlers: list[Callable[[dict[str, Any]], None]] = []
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._inbox_thread: Optional[threading.Thread] = None
        self._inbox_running = False

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _trpc_url(self, procedure: str) -> str:
        return f"{self.base_url}/api/trpc/{procedure}"

    def _post(self, procedure: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a tRPC mutation (POST)."""
        try:
            response = self._client.post(
                self._trpc_url(procedure),
                json={"json": payload},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("result", {}).get("data", {}).get("json", data)
        except httpx.HTTPStatusError as e:
            logger.error("tRPC %s failed: %s", procedure, e.response.text)
            raise
        except httpx.RequestError as e:
            logger.error("tRPC %s request error: %s", procedure, e)
            raise

    def _api_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a POST to a REST API endpoint (not tRPC)."""
        try:
            response = self._client.post(
                f"{self.base_url}{path}",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("POST %s failed: %s", path, e.response.text)
            raise
        except httpx.RequestError as e:
            logger.error("POST %s request error: %s", path, e)
            raise

    def _api_get(self, path: str) -> dict[str, Any]:
        """Send a GET to a REST API endpoint."""
        try:
            response = self._client.get(f"{self.base_url}{path}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("GET %s failed: %s", path, e.response.text)
            raise
        except httpx.RequestError as e:
            logger.error("GET %s request error: %s", path, e)
            raise

    def register(self, config: AgentConfig) -> RegisterResponse:
        """Register this agent with Nexus. Stores returned agentId and agentName."""
        payload = config.model_dump(exclude_none=True, by_alias=False)
        # Convert snake_case to camelCase for tRPC
        camel = _to_camel(payload)
        result = self._post("agents.register", camel)
        resp = RegisterResponse(**result)
        self.agent_id = resp.agent_id
        self.agent_name = config.name
        logger.info("Registered agent %s → id=%s", config.name, self.agent_id)
        return resp

    def heartbeat(self, status: Optional[str] = None) -> None:
        """Ping Nexus to report liveness. Call every 30–60 seconds."""
        if not self.agent_id:
            raise RuntimeError("agent_id not set — call register() first")
        payload: dict[str, Any] = {"agentId": self.agent_id}
        if status:
            payload["status"] = status
        self._post("agents.heartbeat", payload)
        logger.debug("Heartbeat sent for %s", self.agent_id)

    def track(
        self,
        metric_type: str,
        value: float,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Buffer a metric event. Automatically flushed every 5s or when
        the buffer reaches 50 events.
        """
        if not self.agent_id:
            raise RuntimeError("agent_id not set — call register() first")

        event = MetricEvent(
            agent_id=self.agent_id,
            metric_type=metric_type,
            value=value,
            metadata=metadata,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        self._metric_buffer.append(event)
        logger.debug("Buffered metric %s=%.4f (buffer=%d)", metric_type, value, len(self._metric_buffer))

        if len(self._metric_buffer) >= _FLUSH_BATCH_SIZE:
            self._flush_sync()

    def flush(self) -> IngestResponse:
        """Force-flush all buffered metrics to the API."""
        return self._flush_sync()

    def _flush_sync(self) -> IngestResponse:
        if not self._metric_buffer:
            return IngestResponse(inserted=0)

        batch = self._metric_buffer.copy()
        self._metric_buffer.clear()

        payload = [
            {
                "agentId": e.agent_id,
                "metricType": e.metric_type,
                "value": e.value,
                "metadata": e.metadata,
                "timestamp": e.timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z") if e.timestamp else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            }
            for e in batch
        ]

        try:
            result = self._post("metrics.ingest", payload)
            resp = IngestResponse(**result) if isinstance(result, dict) else IngestResponse(inserted=len(batch))
            logger.info("Flushed %d metrics", resp.inserted)
            return resp
        except Exception as e:
            # Re-queue on failure
            self._metric_buffer = batch + self._metric_buffer
            logger.error("Metric flush failed, re-queued %d events: %s", len(batch), e)
            raise

    def reply(
        self,
        thread_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Send a message reply to a Nexus chat thread.
        POSTs to /api/chat/agent-reply which writes to DB and broadcasts via Socket.io.
        """
        if not self.agent_name:
            raise RuntimeError("agent_name not set — call register() first or pass agent_name to constructor")

        payload: dict[str, Any] = {
            "threadId": thread_id,
            "content": content,
            "agentName": self.agent_name,
        }
        if metadata:
            payload["metadata"] = metadata

        result = self._api_post("/api/chat/agent-reply", payload)
        logger.info("reply() to thread %s → messageId=%s", thread_id, result.get("messageId"))

    def audit(
        self,
        action_type: str,
        description: str,
        *,
        conversation_id: Optional[str] = None,
        node_name: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost_usd: Optional[float] = None,
        duration_ms: Optional[int] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        input_payload: Optional[dict[str, Any]] = None,
        output_payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Write an audit log entry to the Nexus AMS.

        Every significant agent action should be audited: LLM calls, searches,
        form fills, document reads, errors. This builds a complete activity trail
        visible in the AMS Audit Log page.

        Args:
            action_type: Category of the action. Standard types:
                - "llm_call": Any LLM invocation
                - "tool_call": External tool usage (SERP search, etc.)
                - "task_started": Beginning of a user request
                - "task_completed": Successful completion
                - "task_failed": Error during processing
                - "document_read": Reading context documents
                - "form_parsed": Parsing a tender form
                - "form_filled": Filling form fields
                - "intent_classified": User intent detection
                - "config_changed": Agent config updates
            description: Human-readable summary of what happened.
            conversation_id: Thread ID if this action is part of a chat.
            node_name: LangGraph node name (discover, evaluate, etc.)
            model_used: LLM model identifier.
            tokens_input: Prompt tokens consumed.
            tokens_output: Completion tokens generated.
            cost_usd: Estimated cost of this action.
            duration_ms: Wall-clock time for the action.
            status: "success", "failure", or "partial".
            error_message: Error details if status is "failure".
            input_payload: The input to this action (prompt, query, form fields).
            output_payload: The output (LLM response, search results, filled values).
        """
        if not self.agent_name:
            raise RuntimeError("agent_name not set — call register() first")

        payload: dict[str, Any] = {
            "agentName": self.agent_name,
            "actionType": action_type,
            "description": description,
            "status": status,
        }

        if conversation_id:
            payload["conversationId"] = conversation_id
        if node_name:
            payload["nodeName"] = node_name
        if model_used:
            payload["modelUsed"] = model_used
        if tokens_input is not None:
            payload["tokensInput"] = tokens_input
        if tokens_output is not None:
            payload["tokensOutput"] = tokens_output
        if cost_usd is not None:
            payload["costUsd"] = cost_usd
        if duration_ms is not None:
            payload["durationMs"] = duration_ms
        if error_message:
            payload["errorMessage"] = error_message
        if input_payload:
            payload["inputPayload"] = input_payload
        if output_payload:
            payload["outputPayload"] = output_payload

        try:
            self._api_post("/api/agents/audit", payload)
            logger.debug("audit() %s: %s", action_type, description[:80])
        except Exception as e:
            # Audit logging should never crash the agent
            logger.warning("audit() failed (non-fatal): %s", e)

    def stream(
        self,
        thread_id: str,
        message_id: str,
        generator: Iterator[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Stream tokens to a Nexus chat thread.
        Each token is POSTed to /api/chat/stream-token for real-time broadcast.
        Call with done=True and metadata after the last token.

        Args:
            thread_id: UUID of the conversation
            message_id: Unique ID for this streaming message
            generator: Iterator that yields string tokens
            metadata: Optional dict with tokensInput, tokensOutput, costUsd, durationMs
        """
        if not self.agent_id:
            raise RuntimeError("agent_id not set — call register() first")
        if not self.agent_name:
            raise RuntimeError("agent_name not set — call register() first or pass agent_name to constructor")

        start_time = time.time()

        for token in generator:
            try:
                self._api_post("/api/chat/stream-token", {
                    "threadId": thread_id,
                    "agentId": self.agent_id,
                    "agentName": self.agent_name,
                    "messageId": message_id,
                    "token": token,
                    "done": False,
                })
            except Exception as e:
                logger.warning("stream() token broadcast failed: %s", e)

        # Send done signal
        duration_ms = int((time.time() - start_time) * 1000)
        done_payload: dict[str, Any] = {
            "threadId": thread_id,
            "agentId": self.agent_id,
            "agentName": self.agent_name,
            "messageId": message_id,
            "token": "",
            "done": True,
        }
        if metadata:
            done_payload["metadata"] = metadata
        else:
            done_payload["metadata"] = {
                "tokensInput": 0,
                "tokensOutput": 0,
                "costUsd": 0.0,
                "durationMs": duration_ms,
            }

        try:
            self._api_post("/api/chat/stream-token", done_payload)
            logger.info("stream() complete for thread %s message %s (%dms)", thread_id, message_id, duration_ms)
        except Exception as e:
            logger.warning("stream() done signal failed: %s", e)

    def poll_inbox(
        self,
        handler: Callable[[dict[str, Any]], None],
        interval: float = _INBOX_POLL_INTERVAL,
        run_once: bool = False,
    ) -> None:
        """
        Poll /api/agents/{name}/inbox for incoming messages.
        Calls handler for each message. Runs in a blocking loop unless
        run_once=True or close() is called.

        For background polling, call start_inbox_loop() instead.
        """
        if not self.agent_name:
            raise RuntimeError("agent_name not set — call register() first or pass agent_name to constructor")

        path = f"/api/agents/{self.agent_name}/inbox"

        while True:
            try:
                result = self._api_get(path)
                messages = result.get("messages", [])
                for msg in messages:
                    try:
                        handler(msg)
                    except Exception as e:
                        logger.error("Inbox handler error: %s", e)
            except Exception as e:
                logger.warning("poll_inbox() error: %s", e)

            if run_once:
                break
            time.sleep(interval)

    def start_inbox_loop(
        self,
        handler: Callable[[dict[str, Any]], None],
        interval: float = _INBOX_POLL_INTERVAL,
    ) -> None:
        """
        Start polling the inbox in a background daemon thread.
        Non-blocking. Call close() to stop.
        """
        if self._inbox_thread and self._inbox_thread.is_alive():
            logger.warning("Inbox loop already running")
            return

        self._inbox_running = True

        def _run() -> None:
            while self._inbox_running:
                self.poll_inbox(handler, interval=interval, run_once=True)
                time.sleep(interval)

        self._inbox_thread = threading.Thread(target=_run, daemon=True)
        self._inbox_thread.start()
        logger.info("Inbox polling started (interval=%.1fs)", interval)

    def on_message(self, handler: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback for incoming chat messages."""
        self._message_handlers.append(handler)
        logger.debug("Registered message handler #%d", len(self._message_handlers))

    # ----- Human-in-the-Loop Approval Workflow -----

    def submit_for_approval(
        self,
        thread_id: Optional[str],
        action_type: str,
        title: str,
        description: Optional[str] = None,
        input_summary: Optional[dict[str, Any]] = None,
        output_summary: Optional[dict[str, Any]] = None,
        file_path: Optional[str] = None,
        filename: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Submit an agent action for human approval instead of sending directly.

        High-stakes actions (filled forms, draft proposals, tender submissions)
        are queued in the Approvals inbox for the user to review.

        Args:
            thread_id: Conversation UUID
            action_type: "form_fill", "draft_submission", "tender_response", or "custom"
            title: Short name for the submission (e.g. "Form: tender_response.pdf")
            description: Why the agent took this action
            input_summary: Dict of input parameters
            output_summary: Dict of output (field counts, etc.)
            file_path: Local path to file to attach (uploaded to MinIO)
            filename: Display name for the file
            metadata: Additional info (cost_usd, tokens, model, duration_ms)

        Returns:
            {"submissionId": "uuid", "status": "pending"}
        """
        if not self.agent_name:
            raise RuntimeError("agent_name not set — call register() first")

        payload: dict[str, Any] = {
            "agentName": self.agent_name,
            "actionType": action_type,
            "title": title,
        }
        # Only include conversationId when truthy. AMS's /api/submissions/create
        # Zod schema accepts a nullable UUID, but sending JSON `null` was
        # making round-trip diagnostics noisier than needed. The other
        # optional fields (description / inputSummary / metadata) already
        # follow this pattern just below.
        if thread_id:
            payload["conversationId"] = thread_id
        if description:
            payload["description"] = description
        if input_summary:
            payload["inputSummary"] = input_summary
        if output_summary:
            payload["outputSummary"] = output_summary
        if metadata:
            payload["metadata"] = metadata

        # Upload file if provided. Best-effort: if MinIO is unreachable (or
        # any other upload failure), log and continue WITHOUT a documentId.
        # The submission still gets created with the full content in
        # `outputSummary`; the operator can review + approve via the editor
        # which reads from outputSummary, not from MinIO. Without this guard,
        # one missing piece of infrastructure (object storage not provisioned
        # yet) silently kills the entire draft-email flow.
        if file_path:
            try:
                file_meta = self.upload_file(file_path, filename)
                payload["documentId"] = file_meta.get("fileId")
            except Exception as exc:
                logger.warning(
                    "submit_for_approval: file upload failed (continuing without document): %s",
                    exc,
                )

        result = self._api_post("/api/submissions/create", payload)
        logger.info(
            "submit_for_approval() → submissionId=%s, status=%s",
            result.get("submissionId"),
            result.get("status"),
        )
        return result

    def check_approval_status(self, submission_id: str) -> dict[str, Any]:
        """Check the current status of a submission.

        Returns:
            {"id": "uuid", "status": "pending|approved|rejected", "notes": "...", "reason": "..."}
        """
        return self._api_get(f"/api/submissions/{submission_id}")

    def start_heartbeat_loop(self, interval: float = 30.0, status: Optional[str] = None) -> None:
        """
        Start a synchronous heartbeat loop (blocking).
        Run in a thread if you need it in the background.
        """
        logger.info("Starting heartbeat loop (interval=%ds)", interval)
        while True:
            try:
                self.heartbeat(status=status)
            except Exception as e:
                logger.warning("Heartbeat failed: %s", e)
            time.sleep(interval)

    def download_file(self, minio_key: str, dest_path: str) -> str:
        """
        Download a file from AMS storage (MinIO) to a local path.

        Args:
            minio_key: The MinIO storage key (e.g. "chat-attachments/abc.pdf")
            dest_path: Local filesystem path to write the file to

        Returns:
            The dest_path for convenience.
        """
        url = f"{self.base_url}/api/chat/files/{minio_key}"
        # Use self._client so the auth headers from _build_headers() apply.
        # Module-level `httpx.stream(...)` would bypass NEXUS_AGENT_API_KEY and
        # 401 the moment AMS turns on agent auth.
        try:
            with self._client.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
            logger.info("Downloaded %s → %s", minio_key, dest_path)
            return dest_path
        except Exception as e:
            logger.error("download_file() failed for %s: %s", minio_key, e)
            raise

    def upload_file(self, file_path: str, filename: Optional[str] = None) -> dict[str, Any]:
        """
        Upload a file to AMS storage and get attachment metadata back.

        Args:
            file_path: Local filesystem path of the file to upload.
            filename: Optional display filename. Defaults to basename of file_path.

        Returns:
            dict with keys: fileId, filename, mimeType, sizeBytes, minioKey
        """
        import mimetypes
        from pathlib import Path

        path = Path(file_path)
        display_name = filename or path.name
        mime_type = mimetypes.guess_type(display_name)[0] or "application/octet-stream"

        url = f"{self.base_url}/api/chat/agent-upload"

        # Use self._client.post so the auth header carries through. Note
        # we explicitly avoid passing `json=...` so httpx uses multipart
        # (Content-Type is set automatically per `files`). The default
        # JSON Content-Type from _build_headers() is overridden by httpx
        # when `files=` is set.
        with open(file_path, "rb") as f:
            files = {"file": (display_name, f, mime_type)}
            data = {"agentName": self.agent_name or "unknown-agent"}
            resp = self._client.post(url, files=files, data=data, timeout=60.0)
            resp.raise_for_status()

        result = resp.json()
        logger.info("Uploaded %s → minioKey=%s", display_name, result.get("minioKey"))
        return result

    def reply_with_file(
        self,
        thread_id: str,
        content: str,
        file_path: str,
        filename: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Upload a file and send a chat reply with it attached in one step.

        Args:
            thread_id: UUID of the conversation
            content: Markdown text for the reply message
            file_path: Local path to the file to attach
            filename: Optional display name for the attachment
            metadata: Optional metadata dict
        """
        attachment = self.upload_file(file_path, filename)

        payload: dict[str, Any] = {
            "threadId": thread_id,
            "content": content,
            "agentName": self.agent_name,
            "attachments": [attachment],
        }
        if metadata:
            payload["metadata"] = metadata

        result = self._api_post("/api/chat/agent-reply", payload)
        logger.info(
            "reply_with_file() to thread %s → messageId=%s, file=%s",
            thread_id, result.get("messageId"), attachment.get("filename"),
        )

    def notify(
        self,
        notification_type: str,
        title: str,
        *,
        body: Optional[str] = None,
        action_url: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Send an in-app notification to users.

        Args:
            notification_type: One of: approval_required, approval_resolved,
                watchlist_results, agent_error, agent_status, form_completed,
                document_uploaded, system
            title: Short notification title shown in the bell dropdown.
            body: Optional longer description text.
            action_url: Deep link (e.g. "/approvals", "/chat/xxx").
            user_id: Specific user to notify. If omitted, notifies all admins.
            metadata: Extra data to attach (agentId, submissionId, etc.)

        Returns:
            {"id": "uuid"} or {"count": N, "ids": [...]}
        """
        payload: dict[str, Any] = {
            "type": notification_type,
            "title": title,
        }
        if body:
            payload["body"] = body
        if action_url:
            payload["actionUrl"] = action_url
        if user_id:
            payload["userId"] = user_id
        if metadata:
            payload["metadata"] = metadata

        return self._api_post("/api/notifications/create", payload)

    def search_knowledge_base(
        self,
        query: str,
        top_k: int = 15,
    ) -> dict[str, Any]:
        """
        Search the agent's knowledge base for relevant document chunks.

        Uses semantic vector search via pgvector cosine similarity.
        Falls back to full-text context if KB hasn't been indexed yet.

        Args:
            query: The search query
            top_k: Maximum number of chunks to return

        Returns:
            Dict with keys: results, totalResults, totalTokens,
            formattedContext (ready to inject into LLM prompt),
            mode ("semantic" or "legacy")
        """
        if not self.agent_name:
            raise RuntimeError("agent_name not set — call register() first")

        return self._api_post(
            f"/api/agents/{self.agent_name}/search",
            {"query": query, "topK": top_k},
        )

    def get_active_prompt(self) -> dict[str, Any] | None:
        """
        Fetch the active system prompt from Prompt Studio.

        Returns:
            Dict with keys: systemPrompt, version, name, rawInput, promptId
            None if no active prompt is configured.
        """
        if not self.agent_name:
            raise RuntimeError("agent_name not set — call register() first")

        try:
            result = self._api_get(f"/api/agents/{self.agent_name}/prompt")
            if result.get("hasActivePrompt"):
                return result
            return None
        except Exception as e:
            logger.warning("get_active_prompt() failed: %s", e)
            return None

    def close(self) -> None:
        """Flush remaining metrics, stop inbox loop, and close the HTTP client."""
        self._inbox_running = False
        try:
            self._flush_sync()
        except Exception:
            pass
        self._client.close()
        logger.info("NexusClient closed")

    def __enter__(self) -> "NexusClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_camel(d: dict[str, Any]) -> dict[str, Any]:
    """Convert dict keys from snake_case to camelCase."""
    result: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split("_")
        camel_key = parts[0] + "".join(p.capitalize() for p in parts[1:])
        if isinstance(value, dict):
            result[camel_key] = _to_camel(value)
        elif isinstance(value, list):
            result[camel_key] = [
                _to_camel(v) if isinstance(v, dict) else v for v in value
            ]
        else:
            result[camel_key] = value
    return result
