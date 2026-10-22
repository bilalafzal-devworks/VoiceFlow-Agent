"""n8n_client.py - sends an approved result to the n8n workflow (Google Docs + Gmail).

Safety rule: this function REFUSES to call the webhook unless approved=True. The n8n
workflow checks the same flag again, so approval is enforced on both sides.
"""
from __future__ import annotations

import re

import requests

import config

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class N8nError(Exception):
    """Raised when the workflow cannot be reached or reports a failure."""


def markdown_to_plain(md):
    """Light clean-up so the Google Doc does not show raw markdown symbols."""
    text = re.sub(r"^#{1,6}\s*", "", md, flags=re.M)  # headings
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
    text = re.sub(r"`([^`]+)`", r"\1", text)  # inline code
    return text.strip()


def send_to_google(title, content, recipient_email, approved, fmt="doc", webhook_url=None, timeout=90):
    """POST the content to n8n. Returns {'status': 'success', 'doc_url': '...'}."""
    if approved is not True:
        raise PermissionError("Refusing to send: the user has not approved this action.")
    if not content or not content.strip():
        raise N8nError("There is nothing to send (empty content).")
    if not recipient_email or not EMAIL_RE.match(recipient_email.strip()):
        raise N8nError("Please enter a valid recipient email address in the sidebar.")
    url = webhook_url or config.n8n_webhook_url()
    if not url:
        raise N8nError("N8N_WEBHOOK_URL is not set. Add the production webhook URL to .env (or Streamlit secrets).")

    payload = {
        "title": title.strip()[:150] or "VoiceFlow result",
        "content": markdown_to_plain(content),
        "recipient_email": recipient_email.strip(),
        "format": fmt,
        "approval": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        raise N8nError("The n8n workflow took too long to respond. Check that it is active.") from exc
    except requests.RequestException as exc:
        raise N8nError(f"Could not reach n8n: {exc}") from exc

    try:
        data = resp.json()
    except ValueError:
        data = {}

    if resp.status_code >= 400 or data.get("status") != "success":
        message = data.get("message") or f"HTTP {resp.status_code}: {resp.text[:200]}"
        raise N8nError(f"n8n reported a problem: {message}")
    return {"status": "success", "doc_url": data.get("doc_url", "")}
