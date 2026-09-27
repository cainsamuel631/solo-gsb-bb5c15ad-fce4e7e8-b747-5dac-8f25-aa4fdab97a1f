"""Merge-order persistence and state transitions."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import audit, db, evidence, rules, undo
from .merge import MISSING, delete_at, merge3, parse_pointer, set_at
from .patch import generate_patch, sha256_hex
from .release_plan import build_preview
from .reuse import (
    build_reuse_plan,
    match_to_batch_item,
    match_to_view,
)
from .validation import check_result_schema, validate_result
from .rules import (
    InvalidPattern,
    InvalidRuleSet,
    RuleApplicationFailed,
    apply_plan,
    compile_rules,
    group_matches,
    plan_resolutions,
)

STATUS_PENDING = "pending"
STATUS_RESOLVED = "resolved"

DRAFT_OPEN = "open"
DRAFT_COMMITTED = "committed"


class MergeNotFound(Exception):
    pass


class MergeNotResolved(Exception):
    """Raised when the deterministic patch is requested before resolution."""


class StaleRevision(Exception):
    def __init__(self, current_revision: int):
        self.current_revision = current_revision
        super().__init__(f"stale revision, current is {current_revision}")


class AlreadyResolved(Exception):
    pass


class UnknownConflictPointer(Exception):
    def __init__(self, pointer: str):
        self.pointer = pointer
        super().__init__(f"no open conflict at pointer {pointer!r}")


class InvalidResolution(Exception):
    pass


class DraftNotFound(Exception):
    pass


class DraftNotOpen(Exception):
    """The draft was already committed (or otherwise invalidated)."""


class DraftConflictSetChanged(Exception):
    """The merge's open conflicts no longer cover the draft's pointers."""


class SourceMergeInvalid(Exception):
    """The source merge of a reuse request cannot provide resolutions."""


class NothingToUndo(Exception):
    """No effective (non-compensated) adjudication event exists to undo."""


class AuditChainBroken(Exception):
    """The merge's audit chain does not verify; rebuilding it is unsafe."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict:
    return {
        "merge_id": row["id"],
        "status": row["status"],
        "revision": row["revision"],
        "document": json.loads(row["document"]),
        "conflicts": json.loads(row["conflicts"]),
        "conflict_count": len(json.loads(row["conflicts"])),
        "result": json.loads(row["result"]) if row["result"] is not None else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_merge(merge_id: str) -> dict:
    row = db.query_one("SELECT * FROM merges WHERE id = ?", (merge_id,))
    if row is None:
        raise MergeNotFound(merge_id)
    return _row_to_dict(row)


def get_merge_patch(merge_id: str) -> dict:
    """Return the RFC 6902 patch from base_doc to the immutable result.

    Raises MergeNotFound (404) for an unknown merge and MergeNotResolved
    (409) while the merge is still pending.
    """
    row = db.query_one(
        "SELECT status, base_doc, result FROM merges WHERE id = ?",
        (merge_id,),
    )
    if row is None:
        raise MergeNotFound(merge_id)
    if row["status"] != STATUS_RESOLVED or row["result"] is None:
        raise MergeNotResolved(merge_id)
    base = json.loads(row["base_doc"])
    result = json.loads(row["result"])
    return {
        "merge_id": merge_id,
        "status": STATUS_RESOLVED,
        "patch": generate_patch(base, result),
        "base_sha256": sha256_hex(base),
        "result_sha256": sha256_hex(result),
    }


def get_audit_trail(merge_id: str) -> dict:
    """Return the merge order's adjudication audit trail and chain verdict.

    Events are returned in sequence order, each re-verified against the
    per-merge SHA-256 chain; the response reports whether the whole chain
    verifies and, when it does not, the first anomalous sequence number.
    Raises MergeNotFound (404) for an unknown merge.
    """
    if db.query_one("SELECT id FROM merges WHERE id = ?", (merge_id,)) is None:
        raise MergeNotFound(merge_id)
    return audit.get_audit_trail(merge_id)


def export_evidence(merge_id: str) -> dict:
    """Build the signed portable evidence package for a resolved merge.

    The package carries the original base document, the immutable result,
    the deterministic base-to-result patch, the complete audit history and
    the generation time, signed with the configured HMAC key. Raises
    MergeNotFound (404) for an unknown merge and MergeNotResolved (409)
    while the merge is still pending.
    """
    row = db.query_one(
        "SELECT status, base_doc, result FROM merges WHERE id = ?",
        (merge_id,),
    )
    if row is None:
        raise MergeNotFound(merge_id)
    if row["status"] != STATUS_RESOLVED or row["result"] is None:
        raise MergeNotResolved(merge_id)
    return evidence.build_package(
        merge_id,
        json.loads(row["base_doc"]),
        json.loads(row["result"]),
        audit.list_event_dicts(merge_id),
    )


def create_merge(
    base: Any,
    left: Any,
    right: Any,
    idempotency_key: str | None,
    result_schema: Any = None,
) -> tuple[dict, bool]:
    """Create a merge order. Returns (merge_view, created).

    A repeated idempotency key replays the existing merge order instead of
    creating a new one. When ``result_schema`` is given it must itself be a
    valid Draft 2020-12 JSON Schema, and a conflict-free (immediately
    resolved) merge must produce a document satisfying it; either failure
    aborts creation before any row is inserted.
    """
    if idempotency_key:
        row = db.query_one(
            "SELECT * FROM merges WHERE idempotency_key = ?", (idempotency_key,)
        )
        if row is not None:
            return _row_to_dict(row), False

    # Gate 1: the schema itself must be a valid Draft 2020-12 schema.
    if result_schema is not None:
        check_result_schema(result_schema)

    document, conflicts = merge3(base, left, right)
    now = _now()
    status = STATUS_RESOLVED if not conflicts else STATUS_PENDING
    # No conflicts: the result is final and immutable from the start, so it
    # must pass the result schema (gate 2) before anything is persisted.
    if not conflicts and result_schema is not None:
        validate_result(result_schema, document)
    result = document if not conflicts else None

    merge_id = uuid.uuid4().hex
    try:
        db.execute(
            """
            INSERT INTO merges (
                id, idempotency_key, status, revision,
                base_doc, left_doc, right_doc,
                document, conflicts, source_conflicts, initial_document,
                result, result_schema,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                merge_id,
                idempotency_key,
                status,
                1,
                json.dumps(base),
                json.dumps(left),
                json.dumps(right),
                json.dumps(document),
                json.dumps(conflicts),
                # Immutable snapshot of the conflicts as first recorded:
                # resolutions shrink ``conflicts`` but reuse matching needs
                # the original base/left/right/missing of every conflict.
                json.dumps(conflicts),
                # Immutable initial working document: resolution undo
                # replays audit events from this snapshot instead of
                # reverse-editing the current document.
                json.dumps(document),
                json.dumps(result) if result is not None else None,
                json.dumps(result_schema) if result_schema is not None else None,
                now,
                now,
            ),
        )
    except Exception:
        # Concurrent insert with the same idempotency key: replay wins.
        if idempotency_key:
            row = db.query_one(
                "SELECT * FROM merges WHERE idempotency_key = ?", (idempotency_key,)
            )
            if row is not None:
                return _row_to_dict(row), False
        raise
    return get_merge(merge_id), True


def _apply_batch(
    document: Any, conflicts: list[dict], resolutions: list[dict]
) -> tuple[Any, list[dict]]:
    """Validate and apply a batch of resolutions in memory.

    Returns (document, remaining_conflicts). Pure: the caller decides
    whether the outcome is persisted (apply_resolutions) or discarded
    (preview_resolutions). A repeated pointer fails because the first
    occurrence closes the conflict.
    """
    open_conflicts = {c["pointer"]: c for c in conflicts}
    for item in resolutions:
        pointer = item["pointer"]
        conflict = open_conflicts.pop(pointer, None)
        if conflict is None:
            raise UnknownConflictPointer(pointer)
        tokens = parse_pointer(pointer)

        if item.get("choice") is not None:
            side = item["choice"]
            if side in conflict["missing"]:
                # Chosen side deleted the key: honour the deletion.
                document = delete_at(document, tokens)
            else:
                document = set_at(document, tokens, conflict[side])
        elif item.get("value_provided"):
            document = set_at(document, tokens, item["value"])
        else:
            raise InvalidResolution(
                f"resolution for {pointer!r} needs 'choice' or 'value'"
            )

    remaining = [c for c in conflicts if c["pointer"] in open_conflicts]
    return document, remaining


def _resolution_detail_items(resolutions: list[dict]) -> list[dict]:
    """Compact per-pointer record of what a resolution batch decided.

    Mirrors the choice/value semantics of the batch items: an explicit
    null ``value`` is preserved as ``{"value": null}``.
    """
    items = []
    for item in resolutions:
        entry = {"pointer": item["pointer"]}
        if item.get("choice") is not None:
            entry["choice"] = item["choice"]
        else:
            entry["value"] = item["value"]
        items.append(entry)
    return items


# ---------------------------------------------------------------------------
# Unified adjudication plan: the shared "plan, then commit" stage used by
# explicit resolutions, rule adjudication, historical reuse and draft commit.
#
# Every entry point reads and validates the merge row (status + revision) and
# any extra inputs it needs *inside* its transaction, then builds a pure,
# in-memory AdjudicationPlan: the candidate working document, the conflicts
# that stay open, the audit method/pointers/detail and (for draft commits)
# the draft to invalidate. Constructing a plan never writes; only
# ``_commit_adjudication`` does, and it does everything every entry has in
# common exactly once: the final result_schema gate, the merge status
# update, the single revision increment, the audit append and the draft
# invalidation — all on the caller's connection so they commit or roll back
# together. A zero-match rule/reuse run simply builds no plan, performs no
# commit and therefore writes nothing and bumps no revision.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjudicationPlan:
    """One fully-prepared, not-yet-persisted adjudication batch.

    Pure data: building it validates/applies the batch in memory but never
    touches the database; ``_commit_adjudication`` is the sole writer.

    Fields:
        merge_row:       the merge row re-read inside the transaction.
        document:        candidate working document after the batch.
        remaining:       conflicts still open after the batch.
        audit_method:    method recorded on the chained audit event.
        audit_pointers:  pointers the event records (in its effective
                         order — request order, sorted pointer order …).
        audit_detail:    method-specific event detail, in the exact shape
                         undo replay and evidence verification expect.
        result_schema:   the merge row's stored result schema (parsed), so
                         the committer runs the final-document gate.
        draft_id:        when set, this draft is marked committed in the
                         same transaction (draft-commit entry only).
    """

    merge_row: Any
    document: Any
    remaining: list[dict]
    audit_method: str
    audit_pointers: list[str]
    audit_detail: dict
    result_schema: Any = None
    draft_id: str | None = None

    @property
    def resolved(self) -> bool:
        """Whether the batch closes the merge's very last conflict."""
        return not self.remaining


def _load_pending_merge(
    conn, merge_id: str, expected_revision: int
):
    """Re-read a merge inside a transaction and enforce the shared guards.

    Every committing adjudication entry rejects, before any write: an
    unknown merge (MergeNotFound, 404), an already resolved/immutable merge
    (AlreadyResolved, 409) and a stale expected revision (StaleRevision,
    409, carrying the current revision).
    """
    row = conn.execute(
        "SELECT * FROM merges WHERE id = ?", (merge_id,)
    ).fetchone()
    if row is None:
        raise MergeNotFound(merge_id)
    if row["status"] == STATUS_RESOLVED:
        raise AlreadyResolved(merge_id)
    if row["revision"] != expected_revision:
        raise StaleRevision(row["revision"])
    return row


def _build_resolution_plan(
    row, resolutions: list[dict]
) -> AdjudicationPlan:
    """Plan an explicit ``resolutions`` batch (pure, no writes)."""
    document, remaining = _apply_batch(
        json.loads(row["document"]), json.loads(row["conflicts"]), resolutions
    )
    return AdjudicationPlan(
        merge_row=row,
        document=document,
        remaining=remaining,
        audit_method=audit.METHOD_RESOLUTIONS,
        # The event records the batch exactly in request order.
        audit_pointers=[item["pointer"] for item in resolutions],
        audit_detail={"items": _resolution_detail_items(resolutions)},
        result_schema=(
            json.loads(row["result_schema"])
            if row["result_schema"] is not None
            else None
        ),
    )


def _build_rule_plan(row, winners: list, conflicts: list[dict]) -> AdjudicationPlan | None:
    """Apply already-selected rule winners in memory; None on a zero hit.

    Compiling/matching happens in the caller (so the response's grouped
    matches are produced by the same run); this only applies the winners
    and packages the outcome. A failure applying a hit raises before any
    plan (and therefore any write) exists.
    """
    if not winners:
        # Zero matches: the caller reports the run without committing.
        return None
    document, remaining = apply_plan(
        json.loads(row["document"]), conflicts, winners
    )
    # Winners apply in sorted pointer order (see apply_plan); the event
    # records that same effective order.
    ordered = sorted(winners, key=lambda w: w.pointer)
    return AdjudicationPlan(
        merge_row=row,
        document=document,
        remaining=remaining,
        audit_method=audit.METHOD_RULE_RESOLUTIONS,
        audit_pointers=[w.pointer for w in ordered],
        audit_detail={
            "items": [
                {
                    "pointer": w.pointer,
                    "side": w.side,
                    "rule_index": w.rule_index,
                }
                for w in ordered
            ]
        },
        result_schema=(
            json.loads(row["result_schema"])
            if row["result_schema"] is not None
            else None
        ),
    )


def _build_reuse_plan(row, source_row, source_merge_id: str):
    """Match the source merge's adjudications in memory.

    Returns (plan, matches, skipped). ``plan`` is None on a zero-match run
    (a successful no-op); ``matches``/``skipped`` are always returned so
    the entry can serialize its response regardless of whether it commits.
    """
    target_conflicts = json.loads(row["conflicts"])
    source_conflicts = json.loads(source_row["source_conflicts"])
    source_result = json.loads(source_row["result"])
    matches, skipped = build_reuse_plan(
        target_conflicts, source_conflicts, source_result
    )
    if not matches:
        return None, matches, skipped
    conflict_by_pointer = {c["pointer"]: c for c in target_conflicts}
    batch = [
        match_to_batch_item(match, conflict_by_pointer[match.pointer])
        for match in matches
    ]
    document, remaining = _apply_batch(
        json.loads(row["document"]), target_conflicts, batch
    )
    # Matches are yielded in the target's conflict order; that is both the
    # application order and the event's recorded pointer order.
    plan = AdjudicationPlan(
        merge_row=row,
        document=document,
        remaining=remaining,
        audit_method=audit.METHOD_REUSE_RESOLUTIONS,
        audit_pointers=[match.pointer for match in matches],
        audit_detail={
            "source_merge_id": source_merge_id,
            "items": [match_to_view(match) for match in matches],
        },
        result_schema=(
            json.loads(row["result_schema"])
            if row["result_schema"] is not None
            else None
        ),
    )
    return plan, matches, skipped


def _build_draft_plan(row, draft_row) -> AdjudicationPlan:
    """Validate and apply an open draft's resolutions in memory (no writes).

    Assumes the shared merge guards and the draft guards (open, pinned
    revision, conflict-set coverage) have already been enforced by the
    caller. An empty draft is rejected as an invalid resolution.
    """
    conflicts = json.loads(row["conflicts"])
    stored = json.loads(draft_row["resolutions"])
    if not stored:
        raise InvalidResolution("draft has no resolutions to commit")
    open_pointers = {c["pointer"] for c in conflicts}
    if any(pointer not in open_pointers for pointer in stored):
        raise DraftConflictSetChanged(
            "the merge's open conflicts no longer cover the draft"
        )
    batch = _stored_to_batch(stored)
    document, remaining = _apply_batch(
        json.loads(row["document"]), conflicts, batch
    )
    return AdjudicationPlan(
        merge_row=row,
        document=document,
        remaining=remaining,
        audit_method=audit.METHOD_DRAFT_COMMIT,
        # The draft stores resolutions in a JSON object; the event records
        # its pointers in that stored (insertion) order.
        audit_pointers=list(stored.keys()),
        audit_detail={
            "draft_id": draft_row["id"],
            "items": _resolution_detail_items(batch),
        },
        result_schema=(
            json.loads(row["result_schema"])
            if row["result_schema"] is not None
            else None
        ),
        draft_id=draft_row["id"],
    )


def _commit_adjudication(conn, plan: AdjudicationPlan) -> None:
    """Persist one prepared adjudication plan atomically on ``conn``.

    The single shared commit stage for every adjudication entry point. In
    order: the final-document schema gate, the merge status/revision update,
    the optional draft invalidation, and the chained audit event. The gate
    is enforced here rather than at plan construction so every entry shares
    one exact check; it raises (ResultValidationFailed, 422) *before* the
    UPDATE, leaving the merge, the audit chain and the draft untouched. The
    UPDATE increments the revision exactly once and only writes the
    immutable result for the batch that resolves the last conflict. The
    audit append and the draft invalidation run on the same connection, so
    any failure rolls all of it back together.
    """
    row = plan.merge_row
    merge_id = row["id"]
    resolved = plan.resolved
    # Gate: only a batch that resolves the last conflict produces a final
    # document, so only that batch is validated. A partial batch still
    # carries conflict placeholders and is committed unchecked.
    if resolved and plan.result_schema is not None:
        validate_result(plan.result_schema, plan.document)

    now = _now()
    conn.execute(
        """
        UPDATE merges
        SET status = ?, revision = revision + 1, document = ?,
            conflicts = ?, result = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            STATUS_RESOLVED if resolved else STATUS_PENDING,
            json.dumps(plan.document),
            json.dumps(plan.remaining),
            # Once written the result is never updated again: immutable.
            json.dumps(plan.document) if resolved else None,
            now,
            merge_id,
        ),
    )
    if plan.draft_id is not None:
        # Same transaction as the merge update and the audit append: a
        # committed draft is invalidated together with the adjudication it
        # carried; a rollback leaves it open with its resolutions intact.
        conn.execute(
            "UPDATE merge_drafts SET status = ?, updated_at = ? WHERE id = ?",
            (DRAFT_COMMITTED, now, plan.draft_id),
        )
    # Same transaction as the merge update: the audit event commits or
    # rolls back together with the state change it describes.
    audit.record_event(
        conn,
        merge_id,
        plan.audit_method,
        row["revision"],
        row["revision"] + 1,
        plan.audit_pointers,
        plan.audit_detail,
    )


def apply_resolutions(
    merge_id: str, expected_revision: int, resolutions: list[dict]
) -> dict:
    """Apply a batch of conflict resolutions.

    Raises StaleRevision (409) if expected_revision does not match the
    current revision, and AlreadyResolved (409) once the result is final.
    The batch is validated and applied in memory as a plan; the shared
    committer then performs the schema gate, the single revision bump and
    the audit append in one transaction. A rejected batch leaves neither.
    """
    with db.transaction() as conn:
        row = _load_pending_merge(conn, merge_id, expected_revision)
        plan = _build_resolution_plan(row, resolutions)
        _commit_adjudication(conn, plan)
    return get_merge(merge_id)


def preview_resolutions(
    merge_id: str, expected_revision: int, resolutions: list[dict]
) -> dict:
    """Dry-run a resolution batch on an in-memory copy of the merge.

    Same revision/status guards and per-item validation as
    apply_resolutions, but stricter: an item carrying both 'choice' and
    'value' is rejected. Nothing is persisted — no write, no revision
    bump, no status change.
    """
    row = db.query_one("SELECT * FROM merges WHERE id = ?", (merge_id,))
    if row is None:
        raise MergeNotFound(merge_id)
    if row["status"] == STATUS_RESOLVED:
        raise AlreadyResolved(merge_id)
    if row["revision"] != expected_revision:
        raise StaleRevision(row["revision"])

    for item in resolutions:
        if item.get("choice") is not None and item.get("value_provided"):
            raise InvalidResolution(
                f"resolution for {item['pointer']!r} gives both 'choice' and 'value'"
            )

    document, remaining = _apply_batch(
        json.loads(row["document"]), json.loads(row["conflicts"]), resolutions
    )
    would_resolve = not remaining
    preview = {
        "merge_id": merge_id,
        "revision": row["revision"],
        "document": document,
        "remaining_conflicts": remaining,
        "would_resolve": would_resolve,
    }
    if would_resolve:
        # Same deterministic patch/digest the resolved merge would yield.
        base = json.loads(row["base_doc"])
        preview["patch"] = generate_patch(base, document)
        preview["result_sha256"] = sha256_hex(document)
    return preview


def apply_rule_resolutions(
    merge_id: str, expected_revision: int, rule_items: list[dict]
) -> dict:
    """Resolve open conflicts with ordered path-pattern rules.

    Every open conflict is matched against the ordered rules and the unique
    best match (more literal segments, then fewer '*', then declaration
    order) supplies its side. All hits are validated before any write: an
    invalid pattern, a rule without any pattern or a failure applying a
    hit rejects the whole request with no partial modification. The
    revision is bumped at most once, inside the single transaction that
    also appends the audit event; a run that matched nothing writes
    nothing and leaves no event.
    """
    with db.transaction() as conn:
        row = _load_pending_merge(conn, merge_id, expected_revision)

        # Compiling, matching and planning happens entirely in memory;
        # nothing is touched until every selected conflict has been proven
        # applicable and the shared committer runs.
        compiled_rules = compile_rules(rule_items)
        conflicts = json.loads(row["conflicts"])
        winners = plan_resolutions(conflicts, compiled_rules)
        rule_matches = group_matches(winners)
        winner_pointers = {w.pointer for w in winners}
        unmatched = [
            c["pointer"] for c in conflicts if c["pointer"] not in winner_pointers
        ]

        plan = _build_rule_plan(row, winners, conflicts)
        applied = plan is not None
        if plan is not None:
            _commit_adjudication(conn, plan)
        # Nothing matched: there is no plan, so the committer is never
        # reached — no validation, no state change, no revision bump and no
        # audit event; the unmatched conflicts are reported back below.

    view = get_merge(merge_id)
    view["rule_matches"] = rule_matches
    view["unmatched_conflicts"] = unmatched
    view["applied"] = applied
    return view


# ---------------------------------------------------------------------------
# Historical resolution reuse
# ---------------------------------------------------------------------------


def reuse_resolutions(
    merge_id: str, expected_revision: int, source_merge_id: str
) -> dict:
    """Reuse a resolved merge's adjudications on the current open conflicts.

    Every open conflict is matched by JSON Pointer against the source
    merge's conflict snapshot; a match requires base/left/right and the
    missing semantics to be identical after canonicalization (an absent
    key never matches an explicit null). Each match takes its final value
    from the source's immutable result — a path missing there deletes the
    target key as well. Matching and validation run entirely in memory;
    the single UPDATE either applies every match (one revision bump) or
    nothing at all. Zero matches is a successful no-op: no write and no
    revision bump.

    Raises StaleRevision (409) / AlreadyResolved (409) for the target and
    SourceMergeInvalid (400) when the source is unknown, still pending or
    has no conflict snapshot.
    """
    with db.transaction() as conn:
        row = _load_pending_merge(conn, merge_id, expected_revision)

        source_row = conn.execute(
            "SELECT * FROM merges WHERE id = ?", (source_merge_id,)
        ).fetchone()
        if source_row is None:
            raise SourceMergeInvalid(
                f"source merge {source_merge_id!r} does not exist"
            )
        if source_row["status"] != STATUS_RESOLVED or source_row["result"] is None:
            raise SourceMergeInvalid(
                f"source merge {source_merge_id!r} is not resolved yet"
            )
        if source_row["source_conflicts"] is None:
            # Rows created before conflict snapshots existed cannot serve as
            # a reuse source: their original conflict candidates are gone.
            raise SourceMergeInvalid(
                f"source merge {source_merge_id!r} has no conflict snapshot"
            )
        if source_merge_id == merge_id:
            raise SourceMergeInvalid(
                "a merge cannot reuse resolutions from itself"
            )

        # Matching and value selection run entirely in memory; the plan is
        # None for a zero-match run, in which case the shared committer is
        # never reached (no UPDATE, no revision bump, no audit event).
        plan, matches, skipped = _build_reuse_plan(
            row, source_row, source_merge_id
        )
        applied = plan is not None
        if plan is not None:
            _commit_adjudication(conn, plan)

    view = get_merge(merge_id)
    view["applied"] = applied
    view["source_merge_id"] = source_merge_id
    view["applied_resolutions"] = [match_to_view(m) for m in matches]
    view["skipped"] = skipped
    view["applied_count"] = len(matches)
    view["skipped_count"] = len(skipped)
    return view


# ---------------------------------------------------------------------------
# Resolution drafts: a persistent, editable set of conflict resolutions that
# pins the merge revision at creation time and commits atomically.
# ---------------------------------------------------------------------------


def _draft_row_to_dict(row) -> dict:
    stored = json.loads(row["resolutions"])
    resolutions = []
    for pointer, entry in stored.items():
        item = {"pointer": pointer}
        if "choice" in entry:
            item["choice"] = entry["choice"]
        else:
            item["value"] = entry["value"]
        resolutions.append(item)
    return {
        "draft_id": row["id"],
        "merge_id": row["merge_id"],
        "status": row["status"],
        "base_revision": row["base_revision"],
        "resolutions": resolutions,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _stored_to_batch(stored: dict) -> list[dict]:
    """Convert the draft's stored resolutions into _apply_batch items.

    ``stored`` maps pointer -> {"choice": side} or {"value": value}; a
    ``value`` key is always explicit (it may hold a JSON null), mirroring
    the ``value_provided`` semantics of the resolution endpoints.
    """
    return [
        {
            "pointer": pointer,
            "choice": entry.get("choice"),
            "value": entry.get("value"),
            "value_provided": "value" in entry,
        }
        for pointer, entry in stored.items()
    ]


def _get_draft_rows(merge_id: str, draft_id: str):
    """Fetch (merge_row, draft_row) or raise MergeNotFound / DraftNotFound."""
    merge_row = db.query_one("SELECT * FROM merges WHERE id = ?", (merge_id,))
    if merge_row is None:
        raise MergeNotFound(merge_id)
    draft_row = db.query_one(
        "SELECT * FROM merge_drafts WHERE id = ?", (draft_id,)
    )
    if draft_row is None or draft_row["merge_id"] != merge_id:
        raise DraftNotFound(draft_id)
    return merge_row, draft_row


def create_draft(merge_id: str) -> dict:
    """Open a resolution draft pinned to the merge's current revision.

    The draft snapshots the working document and the open conflicts, so
    later edits and previews are stable even if the merge moves on. Only a
    pending merge accepts drafts; a resolved one yields AlreadyResolved.
    """
    row = db.query_one("SELECT * FROM merges WHERE id = ?", (merge_id,))
    if row is None:
        raise MergeNotFound(merge_id)
    if row["status"] == STATUS_RESOLVED:
        raise AlreadyResolved(merge_id)
    draft_id = uuid.uuid4().hex
    now = _now()
    db.execute(
        """
        INSERT INTO merge_drafts (
            id, merge_id, status, base_revision,
            document, conflicts, resolutions, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            draft_id,
            merge_id,
            DRAFT_OPEN,
            row["revision"],
            row["document"],
            row["conflicts"],
            "{}",
            now,
            now,
        ),
    )
    return get_draft(merge_id, draft_id)


def get_draft(merge_id: str, draft_id: str) -> dict:
    _, draft_row = _get_draft_rows(merge_id, draft_id)
    return _draft_row_to_dict(draft_row)


def update_draft(merge_id: str, draft_id: str, items: list[dict]) -> dict:
    """Append or overwrite conflict resolutions on the draft.

    Each item follows the same choice/value semantics as a resolution
    batch: exactly one of 'choice' / 'value' (an explicit null counts as
    'value'). A pointer already present in the draft is overwritten in
    place; a new one is appended. Only the draft row is written — the
    merge order is never touched by draft edits.
    """
    _, draft_row = _get_draft_rows(merge_id, draft_id)
    if draft_row["status"] != DRAFT_OPEN:
        raise DraftNotOpen(draft_id)
    open_pointers = {c["pointer"] for c in json.loads(draft_row["conflicts"])}
    stored = json.loads(draft_row["resolutions"])
    for item in items:
        pointer = item["pointer"]
        if pointer not in open_pointers:
            raise UnknownConflictPointer(pointer)
        if item.get("choice") is not None and item.get("value_provided"):
            raise InvalidResolution(
                f"resolution for {pointer!r} gives both 'choice' and 'value'"
            )
        if item.get("choice") is None and not item.get("value_provided"):
            raise InvalidResolution(
                f"resolution for {pointer!r} needs 'choice' or 'value'"
            )
        if item.get("choice") is not None:
            stored[pointer] = {"choice": item["choice"]}
        else:
            stored[pointer] = {"value": item["value"]}
    db.execute(
        "UPDATE merge_drafts SET resolutions = ?, updated_at = ? WHERE id = ?",
        (json.dumps(stored), _now(), draft_id),
    )
    return get_draft(merge_id, draft_id)


def preview_draft(merge_id: str, draft_id: str) -> dict:
    """Dry-run the draft's resolutions against its creation-time snapshot.

    Nothing is written. Returns the would-be document and the conflicts
    still open after the draft (pointers may be filled in over several
    edits, so a partial draft simply reports the rest); when the draft
    would resolve everything, the deterministic base-to-result patch and
    its SHA-256 are included, mirroring the resolution-preview endpoint.
    """
    merge_row, draft_row = _get_draft_rows(merge_id, draft_id)
    if draft_row["status"] != DRAFT_OPEN:
        raise DraftNotOpen(draft_id)
    document, remaining = _apply_batch(
        json.loads(draft_row["document"]),
        json.loads(draft_row["conflicts"]),
        _stored_to_batch(json.loads(draft_row["resolutions"])),
    )
    would_resolve = not remaining
    preview = {
        "merge_id": merge_id,
        "draft_id": draft_id,
        "base_revision": draft_row["base_revision"],
        "current_revision": merge_row["revision"],
        "document": document,
        "remaining_conflicts": remaining,
        "would_resolve": would_resolve,
    }
    if would_resolve:
        base = json.loads(merge_row["base_doc"])
        preview["patch"] = generate_patch(base, document)
        preview["result_sha256"] = sha256_hex(document)
    return preview


def commit_draft(merge_id: str, draft_id: str) -> dict:
    """Apply the whole draft to the merge in a single transaction.

    Inside the transaction the merge row is re-read and re-validated: the
    draft must still be open, the merge still pending, the revision still
    equal to the draft's pinned base_revision, every drafted pointer still
    an open conflict, and — when the draft resolves the last conflict —
    the candidate result must satisfy result_schema. Any changed condition
    aborts with 409 (a schema violation with 422) and neither the merge
    nor the draft is partially written. On success all resolutions are
    applied with exactly one revision bump and the draft is invalidated.
    """
    with db.transaction() as conn:
        merge_row = conn.execute(
            "SELECT * FROM merges WHERE id = ?", (merge_id,)
        ).fetchone()
        if merge_row is None:
            raise MergeNotFound(merge_id)
        draft_row = conn.execute(
            "SELECT * FROM merge_drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        if draft_row is None or draft_row["merge_id"] != merge_id:
            raise DraftNotFound(draft_id)
        if draft_row["status"] != DRAFT_OPEN:
            raise DraftNotOpen(draft_id)
        if merge_row["status"] == STATUS_RESOLVED:
            raise AlreadyResolved(merge_id)
        if merge_row["revision"] != draft_row["base_revision"]:
            raise StaleRevision(merge_row["revision"])

        # Pure in-memory validation/application (empty draft, conflict-set
        # coverage); the plan writes nothing.
        plan = _build_draft_plan(merge_row, draft_row)
        # The shared committer runs the schema gate, the single revision
        # bump, the draft invalidation and the audit append in this one
        # transaction. Any failure rolls all three back together.
        _commit_adjudication(conn, plan)
    view = get_merge(merge_id)
    view["draft_id"] = draft_id
    return view


# ---------------------------------------------------------------------------
# Undo of the last effective adjudication batch
# ---------------------------------------------------------------------------


def undo_last_resolution(merge_id: str, expected_revision: int) -> dict:
    """Undo exactly the most recent still-effective adjudication event.

    The merge must be pending and the supplied revision current. The
    document and open conflicts are never reverse-edited: they are rebuilt
    from the immutable initial conflict snapshot by replaying the audit
    history with the target event (and every already compensated event)
    skipped. On success the merge row is replaced with the rebuilt state,
    the revision is bumped by one, and a ``resolution-undo`` event is
    appended in the same transaction; its detail names the compensated
    event's sequence number and the restored pointers. A later undo skips
    compensated events automatically, and the hash chain stays verifiable.

    Raises StaleRevision (409) for an outdated revision, AlreadyResolved
    (409) for a resolved (immutable) merge and NothingToUndo (409) when no
    effective adjudication remains. Any failure rolls the whole request
    back: no state change and no audit event.
    """
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT * FROM merges WHERE id = ?", (merge_id,)
        ).fetchone()
        if row is None:
            raise MergeNotFound(merge_id)
        if row["status"] == STATUS_RESOLVED:
            raise AlreadyResolved(merge_id)
        if row["revision"] != expected_revision:
            raise StaleRevision(row["revision"])

        events = conn.execute(
            "SELECT * FROM merge_audit_events WHERE merge_id = ? ORDER BY seq ASC",
            (merge_id,),
        ).fetchall()
        # Fail closed: the rebuilt state is only as trustworthy as the
        # history it is replayed from. A tampered or gapped chain makes the
        # recorded decisions unverifiable, so no state is rewritten and no
        # undo event is appended; the exception rolls the transaction back.
        if not audit.verify_chain(conn, merge_id):
            raise AuditChainBroken(merge_id)
        target = undo.find_undo_target(events)
        if target is None:
            raise NothingToUndo(merge_id)

        restored_pointers = json.loads(target["pointers"])
        # Rebuild from the immutable snapshot, skipping the target batch.
        # Replay raises AuditReplayError on an inconsistent history; that
        # propagates and rolls the transaction back, so no state changes.
        document, open_conflicts = undo.rebuild_state(
            row, events, skip_seq=target["seq"]
        )

        conn.execute(
            """
            UPDATE merges
            SET revision = revision + 1, document = ?,
                conflicts = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(document),
                json.dumps(open_conflicts),
                _now(),
                merge_id,
            ),
        )
        # The undo itself is a chained event. It does not change status
        # (a resolved merge cannot be undone) and records exactly which
        # event it compensates, so a subsequent undo skips both.
        audit.record_event(
            conn,
            merge_id,
            audit.METHOD_RESOLUTION_UNDO,
            row["revision"],
            row["revision"] + 1,
            restored_pointers,
            {
                undo.DETAIL_UNDONE_SEQ: target["seq"],
                undo.DETAIL_RESTORED_POINTERS: restored_pointers,
                "undone_method": target["method"],
            },
        )

    view = get_merge(merge_id)
    view["undone_seq"] = target["seq"]
    view["undone_method"] = target["method"]
    view["restored_pointers"] = restored_pointers
    view["restored_count"] = len(restored_pointers)
    return view


# ---------------------------------------------------------------------------
# Read-only release-plan preview
# ---------------------------------------------------------------------------


def preview_release_plan(nodes: list[dict]) -> dict:
    """Preview stable release batches for a dependency graph of merge orders.

    Read-only: the nodes are validated as an acyclic graph whose edges
    point at other listed nodes, their current status/revision is fetched
    from SQLite, and release batches are computed entirely in memory.
    Nothing is written and no merge row is updated, so the preview never
    bumps a revision. Raises release_plan.InvalidReleasePlan (400) for
    duplicate nodes, self-dependencies, dangling edges, cycles or merges
    that do not exist.
    """
    merge_ids = [node["merge_id"] for node in nodes]
    # SQLite caps host parameters at 999 on older builds, so fetch the rows
    # in 500-id chunks. All chunks run inside one read-only snapshot
    # transaction: adjudications committing between two chunk queries must
    # not mix an old prerequisite state with a new successor state (which
    # would falsely report a publishable node as dependency_blocked). The
    # snapshot connection writes nothing and bumps no revision; the next
    # request opens a fresh snapshot and observes any committed state.
    state: dict[str, dict] = {}
    with db.read_snapshot() as conn:
        for start in range(0, len(merge_ids), 500):
            chunk = merge_ids[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            for row in conn.execute(
                f"SELECT id, status, revision FROM merges WHERE id IN ({placeholders})",
                tuple(chunk),
            ):
                state[row["id"]] = {
                    "status": row["status"],
                    "revision": row["revision"],
                }
    return build_preview(nodes, state)
