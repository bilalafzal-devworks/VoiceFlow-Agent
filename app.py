"""app.py - the Streamlit web interface for VoiceFlow Agent.

Flow:  record voice -> Whisper transcript (editable) -> multi-agent pipeline -> answer
       -> "Send this to your Google account?" -> yes (voice or button) -> n8n webhook.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import sys

# ChromaDB needs SQLite >= 3.35. Streamlit Community Cloud ships an older one, so we
# swap in pysqlite3 when it is installed (it is listed in requirements.txt for Linux).
try:
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

import agents
import config
import n8n_client
import rag
import voice

st.set_page_config(page_title="VoiceFlow Agent", page_icon="🎙️", layout="wide")

LANGUAGES = {"Auto-detect": None, "English": "en", "Urdu": "ur", "Hindi": "hi", "Arabic": "ar"}


# ------------------------------------------------------------------ state
def init_state():
    defaults = {
        "transcript_box": "",
        "result": None,          # PipelineResult of the latest answer
        "last_answer": "",       # text that may be sent to Google
        "last_title": "",
        "send_state": "idle",    # idle | asking | sent | declined | error
        "doc_url": "",
        "error_msg": "",
        "seen_audio": set(),     # hashes of recordings we already processed
        "notice": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def is_new_audio(audio_file, tag):
    """True the first time we see a given recording (Streamlit reruns the script a lot)."""
    digest = tag + hashlib.md5(audio_file.getvalue()).hexdigest()
    if digest in st.session_state.seen_audio:
        return False
    st.session_state.seen_audio.add(digest)
    return True


# ------------------------------------------------------------- actions
def run_pipeline(query, version):
    """Run the agents and update session state."""
    st.session_state.send_state = "idle"
    st.session_state.doc_url = ""
    st.session_state.notice = ""
    with st.status("Agents are working...", expanded=True) as status:
        try:
            result = agents.answer_question(
                query,
                has_previous_answer=bool(st.session_state.last_answer),
                prompt_version=version,
                progress=lambda msg: status.write(msg),
            )
        except agents.AgentError as exc:
            status.update(label="Something went wrong", state="error")
            st.error(str(exc))
            return
        except Exception as exc:  # never show a raw traceback to the user
            status.update(label="Something went wrong", state="error")
            st.error(f"Unexpected error: {exc}")
            return
        status.update(label="Done", state="complete", expanded=False)

    if result.intent == "send_save":
        if st.session_state.last_answer:
            st.session_state.send_state = "asking"  # still needs an explicit yes
            st.session_state.notice = "You asked to send the last answer. Please confirm below."
        else:
            st.session_state.notice = "There is no answer to send yet. Ask a question first."
        return

    st.session_state.result = result
    if not result.evidence:  # nothing useful to send: skip the approval question
        st.session_state.last_answer = ""
        st.session_state.send_state = "idle"
        st.session_state.notice = (
            "No sources were found, so there is nothing worth sending. "
            "Open 'Agent trace' to see why, or try rephrasing the request."
        )
        return
    st.session_state.last_answer = result.answer
    st.session_state.last_title = f"VoiceFlow: {query[:90]}"
    st.session_state.send_state = "asking"  # human-in-the-loop: ask before sending


def send_now(recipient):
    """Call n8n. This is the ONLY place the webhook is triggered, and only after approval."""
    try:
        out = n8n_client.send_to_google(
            title=st.session_state.last_title,
            content=st.session_state.last_answer,
            recipient_email=recipient,
            approved=True,  # reached only from a Yes button or a spoken yes
        )
        st.session_state.doc_url = out.get("doc_url", "")
        st.session_state.send_state = "sent"
    except (n8n_client.N8nError, PermissionError) as exc:
        st.session_state.error_msg = str(exc)
        st.session_state.send_state = "error"


# ---------------------------------------------------------------- sidebar
def sidebar():
    st.sidebar.header("Settings")
    versions = config.available_prompt_versions() or ["v2"]
    default = versions.index(config.DEFAULT_PROMPT_VERSION) if config.DEFAULT_PROMPT_VERSION in versions else 0
    version = st.sidebar.selectbox("Prompt version", versions, index=default)
    language = st.sidebar.selectbox("Speech language", list(LANGUAGES))
    recipient = st.sidebar.text_input("Recipient email (Gmail)", value=config.default_recipient_email())

    st.sidebar.divider()
    st.sidebar.header("Knowledge base")
    uploads = st.sidebar.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads and st.sidebar.button("Add to knowledge base"):
        for f in uploads:
            try:
                with st.spinner(f"Embedding {f.name}..."):
                    n = rag.ingest_file(f.name, f.getvalue())
                st.sidebar.success(f"{f.name}: {n} chunks stored")
            except Exception as exc:
                st.sidebar.error(f"{f.name}: {exc}")

    try:
        sources = rag.list_sources()
    except Exception as exc:
        sources = {}
        st.sidebar.error(f"Knowledge base unavailable: {exc}")
    if sources:
        for name, chunks in sources.items():
            col1, col2 = st.sidebar.columns([4, 1])
            col1.caption(f"{name} ({chunks} chunks)")
            if col2.button("🗑", key=f"del_{name}"):
                rag.delete_source(name)
                st.rerun()
    else:
        st.sidebar.caption("Empty. Upload a document to get started.")

    st.sidebar.divider()
    st.sidebar.caption(
        "Keys found: "
        f"Groq {'✅' if config.groq_api_key() else '❌'} · "
        f"HF {'✅' if config.hf_token() or config.EMBEDDING_BACKEND == 'local' else '❌'} · "
        f"n8n {'✅' if config.n8n_webhook_url() else '❌'}"
    )
    return version, LANGUAGES[language], recipient


# ------------------------------------------------------------- main page
def show_result(result):
    if result.intent == "send_save":
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intent", result.intent.replace("_", " "))
    c2.metric("Web used", "yes" if result.used_web else "no")
    c3.metric("Reviewer", "approved" if result.approved else "not approved")
    c4.metric("Revisions", result.revisions)
    st.markdown(result.answer)
    if not result.approved:
        st.warning("The Reviewer did not fully approve this answer. Check the sources before sending.")
    with st.expander("Evidence used"):
        for ev in result.evidence:
            st.markdown(f"**[{ev.id}]** {ev.label}")
            st.caption(ev.text)
        if not result.evidence:
            st.caption("No evidence found.")
    with st.expander("Agent trace (for the course report)"):
        st.json(result.trace)


def approval_block(recipient):
    state = st.session_state.send_state
    if state == "asking":
        st.divider()
        st.subheader("Would you like me to send this to your Google account?")
        st.caption("Nothing is sent unless you say yes. Click a button or answer by voice (e.g. “yes, send it”).")
        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Yes, send it", type="primary", use_container_width=True):
            send_now(recipient)
            st.rerun()
        if col_no.button("❌ No, thanks", use_container_width=True):
            st.session_state.send_state = "declined"
            st.rerun()

        reply = st.audio_input("…or answer by voice", key="confirm_audio")
        if reply and is_new_audio(reply, "confirm"):
            try:
                said = voice.transcribe(reply.getvalue())
            except voice.VoiceError as exc:
                st.error(str(exc))
                return
            decision = voice.interpret_confirmation(said)
            st.write(f"You said: “{said}”")
            if decision == "yes":
                send_now(recipient)
                st.rerun()
            elif decision == "no":
                st.session_state.send_state = "declined"
                st.rerun()
            else:
                st.info("I did not catch a clear yes or no. Please say “yes, send it” or “no”.")
    elif state == "sent":
        st.success("Sent! Check your Gmail for the message.")
        if st.session_state.doc_url:
            st.markdown(f"📄 [Open the Google Doc]({st.session_state.doc_url})")
    elif state == "declined":
        st.info("Okay, I did not send anything.")
    elif state == "error":
        st.error(st.session_state.error_msg)
        if st.button("Try again"):
            st.session_state.send_state = "asking"
            st.rerun()


def main():
    init_state()
    version, language, recipient = sidebar()

    st.title("🎙️ VoiceFlow Agent")
    st.caption("Speak a request → multi-agent RAG answer → optional delivery to Google Docs + Gmail via n8n")

    # 1) voice -> text (done BEFORE the text box is drawn so we can fill it)
    audio = st.audio_input("Record your request")
    if audio and is_new_audio(audio, "request"):
        try:
            with st.spinner("Transcribing with Whisper..."):
                st.session_state.transcript_box = voice.transcribe(audio.getvalue(), language=language)
        except voice.VoiceError as exc:
            st.error(str(exc))

    # 2) editable transcript
    st.text_area("Transcript (edit if needed, or just type)", key="transcript_box", height=90)
    if st.button("Ask the agents", type="primary"):
        query = st.session_state.transcript_box.strip()
        if query:
            run_pipeline(query, version)
        else:
            st.warning("Record or type a request first.")

    if st.session_state.notice:
        st.info(st.session_state.notice)

    # 3) answer + approval
    if st.session_state.result:
        show_result(st.session_state.result)
    approval_block(recipient)


main()
