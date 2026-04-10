# rag_retrieval.py — minimal live RAG from rag_clean for /ask_final
import json
import logging
import math
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

from chat_engine import norm_q

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
TOP_K = 3

# Top-1 cosine must be high enough to beat unrelated chunks (tune with embed model).
_HIGH_CONFIDENCE_COSINE = 0.71
# Long numeric token in the top answer → likely a unique test / id / code (e.g. 91827).
_DISTINCTIVE_CODE = re.compile(r"\d{5,}")

# Verbose trace for RAG debugging (also enabled if augmented question contains this substring)
_RAG_DEBUG_SUBSTRING = "91827"


def _rag_debug_trace(augmented_question: str) -> bool:
    return os.environ.get("RAG_DEBUG") == "1" or _RAG_DEBUG_SUBSTRING in (
        augmented_question or ""
    )


def _questions_effectively_match(user_q: str, doc_q: str) -> bool:
    """Exact or near-exact match on normalized text (same helper style as chat)."""
    a = norm_q(user_q)
    b = norm_q(doc_q)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter in longer and len(shorter) >= 12 and len(shorter) / len(longer) >= 0.88:
        return True
    return False


def _answer_has_distinctive_code(text: str) -> bool:
    return bool(_DISTINCTIVE_CODE.search(text or ""))


def _verbatim_phrase_from_top_answer(answer: str) -> str:
    """Prefer full short answers; for long text, take the line that holds the code token."""
    t = (answer or "").strip()
    if not t:
        return ""
    if len(t) <= 900:
        return t
    m = _DISTINCTIVE_CODE.search(t)
    if not m:
        return t[:900]
    line_start = t.rfind("\n", 0, m.start())
    line_start = 0 if line_start < 0 else line_start + 1
    line_end = t.find("\n", m.end())
    line_end = len(t) if line_end < 0 else line_end
    segment = t[line_start:line_end].strip()
    return segment if segment else t[:900]


def _rag_clean_columns(con: sqlite3.Connection) -> Optional[List[str]]:
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='rag_clean'"
    )
    if cur.fetchone() is None:
        return None
    cur = con.execute('PRAGMA table_info("rag_clean")')
    return [row[1] for row in cur.fetchall()]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _parse_embedding_json(raw: Any) -> Optional[List[float]]:
    if raw is None or raw == "":
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list):
        return None
    out: List[float] = []
    for x in data:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            out.append(float(x))
        else:
            return None
    return out if out else None


def embed_augmented_question(
    text: str, openai_client: OpenAI, *, debug_trace: bool = False
) -> Optional[List[float]]:
    """
    Must match embed_db.py: combined = f"Q: {q}\\nA: {a}" then newline→space before API.
    At query time we only have the user question, so A is empty — still use Q:/A: layout.
    """
    qpart = (text or "").replace("\n", " ").strip()
    if not qpart:
        return None
    combined = f"Q: {qpart}\nA: "
    payload = combined.replace("\n", " ").strip()
    if debug_trace:
        logger.warning("RAG_DEBUG: embedding_input_for_api=%r", payload)
    try:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[payload])
        return list(resp.data[0].embedding)
    except Exception as exc:
        logger.warning("RAG: query embedding failed: %s", exc)
        return None


def build_rag_context_from_rag_clean(
    con: sqlite3.Connection,
    augmented_question: str,
    openai_client: OpenAI,
    db_path_abs: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Load rag_clean rows with non-empty embeddings, optionally filter is_active=1,
    embed the question, cosine top-3, return (context_for_llm, optional_verbatim_phrase).

    When retrieval is very confident and the top answer contains a distinctive code,
    the second value is the exact text the model must copy verbatim (strict grounding).
    """
    debug_trace = _rag_debug_trace(augmented_question)
    if debug_trace:
        try:
            pragma = list(con.execute("PRAGMA database_list"))
            logger.warning("RAG_DEBUG: PRAGMA database_list=%s", pragma)
        except Exception as exc:
            logger.warning("RAG_DEBUG: PRAGMA database_list failed: %s", exc)
        logger.warning(
            "RAG_DEBUG: db_path_abs=%r augmented_question=%r",
            db_path_abs,
            augmented_question,
        )

    col_names = _rag_clean_columns(con)
    if col_names is None:
        logger.warning("RAG: rag_clean table not found, skipping retrieval")
        return "", None

    col_set = set(col_names)
    for required in ("question", "answer", "embedding"):
        if required not in col_set:
            logger.warning("RAG: rag_clean missing column %r, skipping retrieval", required)
            return "", None

    select_cols = ["id", "question", "answer", "embedding"]
    for opt in ("source", "tags", "source_tier"):
        if opt in col_set:
            select_cols.append(opt)

    where_parts = ["(embedding IS NOT NULL AND TRIM(embedding) != '')"]
    if "is_active" in col_set:
        where_parts.append("(is_active = 1)")

    sql = 'SELECT {} FROM rag_clean WHERE {}'.format(
        ", ".join(select_cols),
        " AND ".join(where_parts),
    )

    cur = con.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    logger.info("RAG: loaded %d candidate rows from rag_clean", len(rows))

    query_vec = embed_augmented_question(
        augmented_question, openai_client, debug_trace=debug_trace
    )
    if not query_vec:
        return "", None

    qdim = len(query_vec)
    if debug_trace:
        logger.warning("RAG_DEBUG: query_embedding_dim=%d", qdim)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    dim_mismatch = 0
    parse_fail = 0
    for row in rows:
        vec = _parse_embedding_json(row.get("embedding"))
        if vec is None:
            parse_fail += 1
            continue
        if len(vec) != qdim:
            dim_mismatch += 1
            if debug_trace and row.get("id") == 6517:
                logger.warning(
                    "RAG_DEBUG: row 6517 dim=%d expected %d", len(vec), qdim
                )
            continue
        score = _cosine_similarity(query_vec, vec)
        scored.append((score, row))

    if debug_trace:
        logger.warning(
            "RAG_DEBUG: scored_rows=%d parse_fail=%d dim_mismatch=%d",
            len(scored),
            parse_fail,
            dim_mismatch,
        )
        try:
            cur7 = con.execute("SELECT * FROM rag_clean WHERE id = 6517")
            r7 = cur7.fetchone()
            if r7:
                cols = [d[0] for d in cur7.description]
                d7 = {cols[i]: r7[i] for i in range(len(cols))}
                emb = d7.get("embedding")
                logger.warning(
                    "RAG_DEBUG: row6517 id=%s is_active=%s embedding_chars=%s",
                    d7.get("id"),
                    d7.get("is_active"),
                    len(emb) if isinstance(emb, str) else None,
                )
            else:
                logger.warning("RAG_DEBUG: no row with id=6517 in rag_clean")
        except Exception as exc:
            logger.warning("RAG_DEBUG: row6517 lookup failed: %s", exc)

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:TOP_K]
    top10 = scored[:10]

    if debug_trace:
        top10_ids = [r.get("id") for _, r in top10]
        in_top10 = 6517 in top10_ids
        logger.warning("RAG_DEBUG: row_6517_in_top10=%s top10_ids=%s", in_top10, top10_ids)
        for rank, (sc, row) in enumerate(top10, start=1):
            ans = (row.get("answer") or "")[:200]
            qtxt = (row.get("question") or "")[:200]
            logger.warning(
                "RAG_DEBUG: top10 rank=%d id=%s score=%.6f question=%r answer_preview=%r",
                rank,
                row.get("id"),
                sc,
                qtxt,
                ans,
            )

    top_ids = [r.get("id") for _, r in top]
    top_scores = [round(s, 4) for s, _ in top]
    logger.info(
        "RAG: top %d similarity scores=%s row ids=%s",
        len(top),
        top_scores,
        top_ids,
    )

    verbatim_phrase: Optional[str] = None
    if scored:
        top_score, top_row = scored[0]
        doc_question = (top_row.get("question") or "").strip()
        doc_answer = (top_row.get("answer") or "").strip()
        if (
            top_score >= _HIGH_CONFIDENCE_COSINE
            and _questions_effectively_match(augmented_question, doc_question)
            and _answer_has_distinctive_code(doc_answer)
        ):
            verbatim_phrase = _verbatim_phrase_from_top_answer(doc_answer)
            if verbatim_phrase:
                logger.info(
                    "RAG: strict verbatim grounding (cosine=%.4f top_id=%s)",
                    top_score,
                    top_row.get("id"),
                )

    if not top:
        return "", verbatim_phrase

    blocks: List[str] = []
    for i, (_, row) in enumerate(top, start=1):
        rid = row.get("id")
        ans = (row.get("answer") or "").strip()
        lines = [f"[{i}] (id={rid})", f"תשובה: {ans}"]
        q = (row.get("question") or "").strip()
        if q:
            lines.append(f"שאלה: {q}")
        src = (row.get("source") or "").strip()
        if src:
            lines.append(f"מקור: {src}")
        tags = row.get("tags")
        if tags is not None and str(tags).strip():
            lines.append(f"תגיות: {tags}")
        blocks.append("\n".join(lines))

    context = "\n\n".join(blocks)
    if debug_trace:
        logger.warning("RAG_DEBUG: rag_context_sent_to_llm=%r", context)
        if verbatim_phrase:
            logger.warning("RAG_DEBUG: verbatim_phrase=%r", verbatim_phrase)
    return context, verbatim_phrase
