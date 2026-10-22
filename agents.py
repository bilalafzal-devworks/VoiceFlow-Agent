"""agents.py - the multi-agent layer (CrewAI agents running on Groq gpt-oss-120b).

Pipeline for one request
    Router     -> classify intent (JSON)
    Retriever  -> judge which knowledge-base passages are relevant / sufficient (JSON)
    Researcher -> (only if the knowledge base is not enough) extract facts from web results (JSON)
    Writer     -> write a cited answer using ONLY the evidence
    Reviewer   -> check groundedness / citations / format (JSON); Writer revises if rejected

Design note: the *tool calls* (vector search, DuckDuckGo search) are executed by plain
Python code, and the agents reason over their results. This is more reliable on free-tier
models than letting the LLM decide when to call tools, and it makes every step testable.

Each agent's system prompt lives in prompts/<version>/<agent>.md (role, goal, constraints,
output format, fallback behaviour) so prompts can be versioned and compared.
"""
from __future__ import annotations

import functools
import json
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

import config
import rag

INTENTS = ("knowledge_question", "web_lookup", "send_save")
AGENT_KEYS = ("router", "retriever", "researcher", "writer", "reviewer")
CITATION_RE = re.compile(r"\[S\d+\]")
DONT_KNOW = "I don't know based on the available sources."


class AgentError(Exception):
    """Raised when the language model could not be reached after retries."""


# ------------------------------------------------------------------ prompts
_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


@lru_cache(maxsize=None)
def load_prompt(agent_key, version):
    """Read prompts/<version>/<agent_key>.md and split it into CrewAI fields.

    The file has '## Role', '## Goal' and other sections (Constraints, Output format,
    Few-shot examples, Fallback behaviour). Role -> Agent.role, Goal -> Agent.goal and
    everything else is joined into Agent.backstory (the agent's system instructions).
    """
    path = config.PROMPTS_DIR / version / f"{agent_key}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    parts = _SECTION_RE.split(path.read_text(encoding="utf-8"))
    sections = {parts[i].strip().lower(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    for required in ("role", "goal"):
        if required not in sections:
            raise ValueError(f"{path} is missing a '## {required.title()}' section")
    backstory = "\n\n".join(
        f"{name.upper()}:\n{body}" for name, body in sections.items() if name not in ("role", "goal")
    )
    return {"role": sections["role"], "goal": sections["goal"], "backstory": backstory}



# ------------------------------------------------------ Groq compatibility
def _patch_litellm_cache_breakpoint():
    """Strip CrewAI's internal 'cache_breakpoint' key before messages reach Groq.

    Recent CrewAI versions tag messages with 'cache_breakpoint' (used for prompt caching
    on some providers). Groq rejects unknown message fields with a 400 error, so we remove
    the key from every request sent through LiteLLM. Safe to call many times.
    """
    try:
        import litellm
    except ImportError:
        return

    def clean(messages):
        if not isinstance(messages, list):
            return messages
        return [
            {k: v for k, v in m.items() if k != "cache_breakpoint"} if isinstance(m, dict) else m
            for m in messages
        ]

    def fix_args(args, kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = clean(kwargs["messages"])
        elif len(args) >= 2:
            args = (args[0], clean(args[1])) + tuple(args[2:])
        return args, kwargs

    original = getattr(litellm, "completion", None)
    if original is not None and not getattr(original, "_voiceflow_patched", False):

        @functools.wraps(original)
        def completion(*args, **kwargs):
            args, kwargs = fix_args(args, kwargs)
            return original(*args, **kwargs)

        completion._voiceflow_patched = True
        litellm.completion = completion

    original_async = getattr(litellm, "acompletion", None)
    if original_async is not None and not getattr(original_async, "_voiceflow_patched", False):

        @functools.wraps(original_async)
        async def acompletion(*args, **kwargs):
            args, kwargs = fix_args(args, kwargs)
            return await original_async(*args, **kwargs)

        acompletion._voiceflow_patched = True
        litellm.acompletion = acompletion

# --------------------------------------------------------------- LLM calls
def run_agent(agent_key, task_description, expected_output, version, temperature=0.2):
    """Run one CrewAI agent on one task and return its raw text output.

    Retries with a short back-off because the Groq free tier can rate-limit.
    """
    from crewai import LLM, Agent, Crew, Process, Task

    _patch_litellm_cache_breakpoint()

    api_key = config.groq_api_key()
    if not api_key:
        raise AgentError("GROQ_API_KEY is not set. Add it to .env (or Streamlit secrets).")

    prompt = load_prompt(agent_key, version)
    last_error = None
    for attempt in range(1, config.LLM_RETRIES + 1):
        try:
            llm = LLM(model=f"groq/{config.GROQ_MODEL}", api_key=api_key, temperature=temperature)
            agent = Agent(
                role=prompt["role"],
                goal=prompt["goal"],
                backstory=prompt["backstory"],
                llm=llm,
                allow_delegation=False,
                verbose=False,
            )
            task = Task(description=task_description, expected_output=expected_output, agent=agent)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            result = crew.kickoff()
            return str(getattr(result, "raw", result)).strip()
        except Exception as exc:  # rate limit, network, provider error...
            last_error = exc
            if attempt < config.LLM_RETRIES:
                time.sleep(2**attempt)
    raise AgentError(f"The language model call failed after {config.LLM_RETRIES} tries: {last_error}")


def extract_json(text):
    """Find the first complete JSON object in a model reply. Returns a dict or None."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


# ------------------------------------------------------------ data classes
@dataclass
class Evidence:
    """One numbered piece of evidence the Writer may cite."""

    id: str  # S1, S2, ...
    kind: str  # "kb" or "web"
    label: str  # "notes.pdf, p.3" or "Page title - https://..."
    text: str
    url: str = ""


@dataclass
class PipelineResult:
    query: str
    intent: str = "knowledge_question"
    route: dict = field(default_factory=dict)
    route_json_ok: bool = True
    answer: str = ""
    evidence: list = field(default_factory=list)
    review: dict = field(default_factory=dict)
    review_json_ok: bool = True
    approved: bool = False
    revisions: int = 0
    used_web: bool = False
    citations_ok: bool = False
    trace: list = field(default_factory=list)


def format_evidence(evidence):
    lines = []
    for ev in evidence:
        lines.append(f"[{ev.id}] ({ev.label})\n{ev.text}")
    return "\n\n".join(lines)


def _notify(progress, message):
    if progress:
        progress(message)


# ------------------------------------------------------------------ agents
def route(query, has_previous_answer, version):
    """Router Agent -> (route_dict, json_ok)."""
    raw = run_agent(
        "router",
        task_description=(
            f'Transcribed user request:\n"""{query}"""\n\n'
            f"A previous answer available to send: {'yes' if has_previous_answer else 'no'}.\n"
            "Classify the request and reply with the JSON object described in your instructions."
        ),
        expected_output='A JSON object with keys "intent", "confidence", "search_query", "reason".',
        version=version,
        temperature=0.0,
    )
    obj = extract_json(raw)
    ok = bool(obj) and obj.get("intent") in INTENTS
    if not ok:  # guardrail: never crash on a malformed reply, fall back to the knowledge base
        obj = {
            "intent": "knowledge_question",
            "confidence": 0.0,
            "search_query": query,
            "reason": "Fallback: router reply was not valid JSON.",
        }
    obj.setdefault("search_query", query)
    if not str(obj.get("search_query", "")).strip():
        obj["search_query"] = query
    return obj, ok


def retrieve(query, version):
    """Retriever Agent: vector search (code) + relevance judgement (agent).

    Returns (evidence_list, info_dict). info has 'sufficient' and 'notes'.
    """
    hits = rag.search(query)
    if not hits:
        return [], {"sufficient": False, "notes": "Knowledge base is empty or returned no passages.", "hits": 0}

    labelled = {f"K{i}": hit for i, hit in enumerate(hits, start=1)}
    passages = "\n\n".join(
        f"[{key}] ({hit['source']}, p.{hit['page']}, distance {hit['distance']})\n{hit['text']}"
        for key, hit in labelled.items()
    )
    raw = run_agent(
        "retriever",
        task_description=(
            f"Question: {query}\n\nPassages from the knowledge base:\n{passages}\n\n"
            "Judge the passages and reply with the JSON object described in your instructions."
        ),
        expected_output='A JSON object with keys "sufficient", "relevant_ids", "notes".',
        version=version,
        temperature=0.0,
    )
    obj = extract_json(raw)
    if obj is None:  # invalid JSON: keep every passage and let the Reviewer judge the final answer
        relevant, sufficient, notes = list(labelled), True, "Retriever reply was not valid JSON; kept all passages."
    else:
        relevant = [k for k in obj.get("relevant_ids", []) if k in labelled]
        sufficient = bool(obj.get("sufficient")) and bool(relevant)
        notes = str(obj.get("notes", ""))

    evidence = [
        Evidence(id="", kind="kb", label=f"{labelled[k]['source']}, p.{labelled[k]['page']}", text=labelled[k]["text"])
        for k in relevant
    ]
    return evidence, {"sufficient": sufficient, "notes": notes, "hits": len(hits)}


# Search engines tried in order. DuckDuckGo alone is often blocked on cloud servers,
# so we fall back to other engines supported by the `ddgs` package, then to Wikipedia.
WEB_BACKENDS = ("auto", "brave", "duckduckgo", "mojeek", "yahoo", "startpage")
last_web_error = ""  # human-readable reason when web search found nothing (shown in the trace)


def _wikipedia_search(query, max_results):
    """Last-resort fallback: the free Wikipedia search API (works from cloud servers)."""
    import requests

    resp = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "srlimit": max_results, "format": "json"},
        headers={"User-Agent": "VoiceFlowAgent/1.0 (course project)"},
        timeout=15,
    )
    resp.raise_for_status()
    out = []
    for item in resp.json().get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))  # remove <span> highlight tags
        out.append({"title": f"{title} (Wikipedia)", "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"), "snippet": snippet})
    return out


def web_search(query, max_results=None):
    """Web search with fallbacks. Returns [{title, url, snippet}] or [] (see last_web_error)."""
    global last_web_error
    max_results = max_results or config.WEB_MAX_RESULTS
    errors = []
    raw = []
    try:
        try:
            from ddgs import DDGS

            for backend in WEB_BACKENDS:
                try:
                    raw = DDGS().text(query, max_results=max_results, backend=backend) or []
                except Exception as exc:  # blocked, rate-limited, no results...
                    errors.append(f"{backend}: {str(exc)[:80]}")
                    raw = []
                if raw:
                    break
        except ImportError:  # older package name (no backend choice)
            from duckduckgo_search import DDGS

            raw = DDGS().text(query, max_results=max_results) or []
    except Exception as exc:
        errors.append(f"search library: {str(exc)[:80]}")

    results = [
        {"title": r.get("title", ""), "url": r.get("href") or r.get("url", ""), "snippet": r.get("body", "")}
        for r in raw
    ]
    if not results:  # fallback to Wikipedia
        try:
            results = _wikipedia_search(query, max_results)
        except Exception as exc:
            errors.append(f"wikipedia: {str(exc)[:80]}")
    last_web_error = "" if results else "; ".join(errors) or "no results returned"
    return results


def research(query, version):
    """Researcher Agent: DuckDuckGo search (code) + fact extraction (agent). Returns (evidence, notes)."""
    results = web_search(query)
    if not results:
        return [], f"Web search returned no results. Details: {last_web_error}"
    listing = "\n\n".join(
        f"[W{i}] {r['title']}\nurl: {r['url']}\n{r['snippet']}" for i, r in enumerate(results, start=1)
    )
    raw = run_agent(
        "researcher",
        task_description=(
            f"Question: {query}\n\nWeb search results:\n{listing}\n\n"
            "Extract the facts that answer the question and reply with the JSON object described in your instructions."
        ),
        expected_output='A JSON object with keys "findings" and "notes".',
        version=version,
        temperature=0.0,
    )
    obj = extract_json(raw)
    evidence = []
    if obj and isinstance(obj.get("findings"), list):
        for f in obj["findings"][:5]:
            if isinstance(f, dict) and f.get("fact"):
                url = str(f.get("url", ""))
                title = str(f.get("title", "")) or url
                evidence.append(Evidence(id="", kind="web", label=f"{title} - {url}", text=str(f["fact"]), url=url))
        return evidence, str(obj.get("notes", ""))
    # Invalid JSON: fall back to the raw search snippets so the user still gets something grounded.
    for r in results[:3]:
        if r["snippet"]:
            evidence.append(Evidence(id="", kind="web", label=f"{r['title']} - {r['url']}", text=r["snippet"], url=r["url"]))
    return evidence, "Researcher reply was not valid JSON; used raw search snippets."


def write(query, evidence, version, feedback=None, previous_draft=None):
    """Writer Agent: answer using ONLY the numbered evidence."""
    task = f"Question: {query}\n\nEvidence (the ONLY facts you may use):\n{format_evidence(evidence)}\n\n"
    if feedback and previous_draft:
        task += f"Your previous draft:\n{previous_draft}\n\nReviewer feedback to fix:\n{feedback}\n\n"
    task += "Write the final answer in the output format described in your instructions."
    return run_agent(
        "writer",
        task_description=task,
        expected_output="A markdown answer with Answer, Key points and Sources sections, with [S#] citations.",
        version=version,
        temperature=0.2,
    )


def review(query, evidence, draft, version):
    """Reviewer Agent -> (review_dict, json_ok)."""
    raw = run_agent(
        "reviewer",
        task_description=(
            f"Question: {query}\n\nEvidence:\n{format_evidence(evidence)}\n\n"
            f"Draft answer to audit:\n{draft}\n\n"
            "Audit the draft and reply with the JSON object described in your instructions."
        ),
        expected_output='A JSON object with keys "approved", "groundedness_score", "citations_ok", "format_ok", "issues", "revision_instructions".',
        version=version,
        temperature=0.0,
    )
    obj = extract_json(raw)
    ok = bool(obj) and "approved" in obj
    if not ok:
        obj = {"approved": False, "groundedness_score": 0, "issues": ["Reviewer reply was not valid JSON."], "revision_instructions": ""}
    return obj, ok


# ---------------------------------------------------------------- pipeline
def answer_question(query, has_previous_answer=False, prompt_version=None, progress: Optional[Callable] = None):
    """Run the whole multi-agent pipeline for one transcribed request."""
    version = prompt_version or config.DEFAULT_PROMPT_VERSION
    result = PipelineResult(query=query)

    # 1) Router
    _notify(progress, "Router Agent: understanding your request...")
    result.route, result.route_json_ok = route(query, has_previous_answer, version)
    result.intent = result.route["intent"]
    result.trace.append({"agent": "Router", "output": result.route})
    if result.intent == "send_save":
        return result  # the UI handles the send flow (it still asks for approval)

    search_query = str(result.route.get("search_query") or query)
    evidence = []

    # 2) Retriever (skipped when the router says the user wants the web)
    kb_sufficient = False
    if result.intent == "knowledge_question":
        _notify(progress, "Retriever Agent: searching the knowledge base...")
        kb_evidence, info = retrieve(search_query, version)
        evidence.extend(kb_evidence)
        kb_sufficient = info["sufficient"]
        result.trace.append({"agent": "Retriever", "output": info})

    # 3) Researcher (only when the knowledge base is not enough)
    if result.intent == "web_lookup" or not kb_sufficient:
        _notify(progress, "Researcher Agent: searching the web...")
        web_evidence, notes = research(search_query, version)
        evidence.extend(web_evidence)
        result.used_web = True
        result.trace.append({"agent": "Researcher", "output": {"findings": len(web_evidence), "notes": notes}})

    for i, ev in enumerate(evidence, start=1):  # number the evidence S1, S2, ...
        ev.id = f"S{i}"
    result.evidence = evidence

    # Guardrail: no evidence at all -> say "I don't know" without calling the Writer.
    if not evidence:
        result.answer = f"{DONT_KNOW} I found nothing relevant in the knowledge base or on the web."
        result.approved = True
        result.citations_ok = True
        result.review = {"approved": True, "groundedness_score": 5, "issues": [], "note": "No evidence; Writer skipped."}
        return result

    # 4) Writer <-> 5) Reviewer loop
    draft, feedback = None, None
    for revision in range(config.MAX_REVISIONS + 1):
        _notify(progress, "Writer Agent: drafting the answer..." if revision == 0 else f"Writer Agent: revising (round {revision})...")
        draft = write(query, evidence, version, feedback=feedback, previous_draft=draft if feedback else None)
        _notify(progress, "Reviewer Agent: checking facts and citations...")
        result.review, result.review_json_ok = review(query, evidence, draft, version)
        result.revisions = revision
        result.trace.append({"agent": "Reviewer", "output": result.review})
        if result.review.get("approved") or not result.review_json_ok:
            break
        feedback = str(result.review.get("revision_instructions") or "; ".join(map(str, result.review.get("issues", []))))

    result.answer = draft or ""
    result.approved = bool(result.review.get("approved"))
    result.citations_ok = bool(CITATION_RE.search(result.answer)) or DONT_KNOW in result.answer
    return result
