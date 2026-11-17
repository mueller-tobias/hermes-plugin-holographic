"""auto_extract stores only the person's own words (0.29.2+de.2): not a quoted reply, a notice the gateway injects as a
user turn, nor a turn that carries an attachment. The wrappers are the ones hermes-agent v0.23.1 builds
(gateway/run_inbound.py, gateway/run.py, tools/process_registry_notifications.py, the platform adapters)."""

import pytest

from hermes_plugin_holographic import HolographicMemoryProvider, _extract_words


@pytest.fixture
def provider(tmp_path):
    instance = HolographicMemoryProvider(config={
        "db_path": str(tmp_path / "facts.db"), "hrr_dim": 64, "auto_extract": True,
    })
    instance.initialize(session_id="own-words")
    yield instance
    instance.shutdown()


def _stored(provider):
    return sorted((fact["content"], fact["category"]) for fact in provider._store.list_facts(limit=20))


def _extract(provider, path, message, author=None):
    """The two ways a user turn reaches extraction: per turn (sync_turn) and from the transcript (session end)."""
    if path == "turn":
        provider.sync_turn(message, "Notiert.", session_id="s", turn_author=author)
    else:
        provider.on_session_end([{"role": "user", "content": message}])


PATHS = ["turn", "session_end"]

QUOTE = ('[Replying to: "Hier die Zusammenfassung von gestern.\nIch bevorzuge Python für alle Skripte.\n'
         'Wir haben beschlossen, SQLite zu behalten. Das Projekt nutzt FastAPI."]\n\n')
OWN_QUOTE = ('[Replying to your previous message: "Wir haben uns für Go entschieden. Ich trinke nie Kaffee nach '
             '16 Uhr."]\n\n')


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("message", [
    QUOTE + "Ok, danke dir",
    QUOTE + "Passt so.",
    OWN_QUOTE + "Gut zu wissen",
    # A quote of one line whose second sentence states a preference.
    '[Replying to: "Kurze Frage vorab. Ich mag keine langen Erklärungen."]\n\nStimmt das noch?',
])
def test_a_quoted_reply_is_not_stored(provider, path, message):
    _extract(provider, path, message)
    assert _stored(provider) == []


@pytest.mark.parametrize("path", PATHS)
def test_own_words_after_a_quote_are_stored_and_the_quote_is_not(provider, path):
    _extract(provider, path, QUOTE + "Ich trinke nie Kaffee nach 16 Uhr. Wir haben uns für PostgreSQL entschieden.")
    assert _stored(provider) == [("Ich trinke nie Kaffee nach 16 Uhr.", "user_pref"),
                                 ("Wir haben uns für PostgreSQL entschieden.", "project")]


def test_a_quote_and_the_authors_name_in_a_shared_chat_are_taken_off(provider):
    """In a group the gateway puts the author's name under the quote ("[Name] text"); it is not part of the words."""
    _extract(provider, "turn", QUOTE + "[Beispiel] Ich bevorzuge kurze Antworten.",
             author={"id": "1", "name": "Beispiel", "is_bot": False})
    assert _stored(provider) == [("Ich bevorzuge kurze Antworten.", "user_pref")]


DOCUMENT = ("[The user sent a text document: 'notizen.txt'. Its content has been included below. "
            "The file is also saved at: /cache/documents/notizen.txt]\n\n"
            "[Content of notizen.txt]:\nIch bevorzuge Python für alle Skripte.\n"
            "Wir haben beschlossen, SQLite zu behalten.\n\n")


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("message", [
    # A text document inlined by the adapter, with and without the person's caption: the whole turn is skipped.
    DOCUMENT + "Kannst du das zusammenfassen",
    DOCUMENT + "Ich nutze Linux auf meinem Laptop.",
    # Inlined without the gateway's note (an adapter that inlines on its own).
    "[Content of plan.md]:\nWir haben uns für Kubernetes entschieden.\nDas Projekt nutzt Helm.",
    # A binary document, an image and a sticker: the note says what the gateway made of it, not the person.
    "[The user sent a document: 'vertrag.pdf'. It is saved at: /cache/documents/vertrag.pdf. Its text is not "
    "inlined here (it's a binary format such as PDF or DOCX).]\n\nIch bevorzuge die zweite Variante mit Anhang.",
    "[The user sent an image~ Here's what I can see:\nEin Schild mit dem Text: Wir haben beschlossen, freitags "
    "zu schließen.]\n\nIch mag dieses Schild sehr gern.",
    '[The user sent a sticker 😀 from "Pack"~ It shows: "Ich liebe Montage"]',
    "[User sent a file: https://example.invalid/notizen.txt]\nIch verwende Markdown für alle Notizen.",
])
def test_a_turn_with_an_attachment_is_not_stored(provider, path, message):
    _extract(provider, path, message)
    assert _stored(provider) == []


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("message", [
    # tools/process_registry_notifications.py
    "[IMPORTANT: Background process proc_ab12 completed (exit code 0).\nCommand: ./deploy.sh\n"
    "Output:\nWir haben beschlossen, SQLite zu behalten.]",
    "[IMPORTANT: 2 background processes completed. Results below.]\nIch bevorzuge den Build mit Docker.",
    '[IMPORTANT: Background process proc_ab12 matched watch pattern "ERROR".\nWir haben uns für Go entschieden.]',
    "[IMPORTANT: Watch for proc_ab12 disabled after too many matches. Ich nutze das Log weiter.]",
    "[ASYNC DELEGATION COMPLETE — deleg_1]\nErgebnis: Wir haben uns für PostgreSQL entschieden.",
    "[ASYNC DELEGATION TASK FAILED — deleg_1, task 1/2]\nIch verwende den Fallback.",
    # gateway/run_notifications.py, gateway/session_automation.py, gateway/run_turn.py
    "[IMPORTANT: 3 background subagent delegations completed.]\nDas Projekt nutzt jetzt FastAPI.",
    "[IMPORTANT: 2 background process results arrived while this session was busy. ...]\nIch mag Python.",
    "[IMPORTANT: MCP servers have been reloaded. Added servers: notizen. 3 MCP tool(s) now available. "
    "The tool list for this conversation has been updated accordingly.]",
    # A system note with nothing of the person after it (an auto-continue turn).
    "[System note: Your previous turn was interrupted. Wir haben beschlossen, weiterzumachen.]",
])
def test_a_notice_turn_is_not_stored(provider, path, message):
    _extract(provider, path, message)
    assert _stored(provider) == []


@pytest.mark.parametrize("path", PATHS)
def test_own_words_after_a_system_note_are_stored(provider, path):
    message = ("[System note: A new message has arrived. The conversation history contains pending tool outputs "
               "from an interrupted turn. IGNORE those pending results. Address the user's NEW message below FIRST. "
               "Do NOT re-execute old tool calls from the history.]\n\nIch bevorzuge kurze Antworten.")
    _extract(provider, path, message)
    assert _stored(provider) == [("Ich bevorzuge kurze Antworten.", "user_pref")]


@pytest.mark.parametrize("path", PATHS)
def test_own_words_are_still_stored_with_trust_and_origin_unchanged(provider, path):
    _extract(provider, path, "Ich bevorzuge kurze und präzise Antworten.")
    facts = provider._store.list_facts(limit=10)
    assert [(f["content"], f["category"]) for f in facts] == [("Ich bevorzuge kurze und präzise Antworten.",
                                                                "user_pref")]
    assert facts[0]["trust_score"] == pytest.approx(0.4)


@pytest.mark.parametrize("message,words", [
    ("Ich mag Tee.", "Ich mag Tee."),
    (QUOTE + "Ich mag Tee.", "Ich mag Tee."),
    (DOCUMENT + "Ich mag Tee.", ""),
    ("[IMPORTANT: Background process proc_1 completed (exit code 0).]", ""),
    ("[System note: The user's previous session was stopped and suspended.]\n\nIch mag Tee.", "Ich mag Tee."),
    # A system note that cannot be taken off cleanly is the gateway's: skipped.
    ("[System note: see [docs] first]\n\nIch mag Tee.", ""),
])
def test_extract_words(message, words):
    assert _extract_words(message) == words
