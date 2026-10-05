"""
Impersona-Env — Shared Pipeline Logic
======================================

Pure-logic functions for iMessage export, contacts handling, conversation
scanning, and filtering.  No CLI I/O (no input(), no argparse, no sys.exit).

Used by run.py (CLI) and the worker scripts.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import traceback
from datetime import datetime, timedelta
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent

MAC_EXPORT_DIR = WORKSPACE / "imessage_export_mac"
IOS_EXPORT_DIR = WORKSPACE / "imessage_export_ios"
MERGED_EXPORT_DIR = WORKSPACE / "imessage_export_merged"
CONTACTS_VCF = WORKSPACE / "contacts.vcf"
SELECTED_CONVOS_FILE = WORKSPACE / "selected_conversations.json"
GENERATION_ERROR_LOG = WORKSPACE / "generation_errors.log"

APPLESCRIPT_EXPORT_CONTACTS = """\
tell application "Contacts"
    set output to ""
    repeat with p in every person
        set pName to name of p
        set phoneList to value of every phone of p
        if (count of phoneList) > 0 then
            set output to output & "BEGIN:VCARD" & linefeed
            set output to output & "VERSION:3.0" & linefeed
            set output to output & "FN:" & pName & linefeed
            repeat with ph in phoneList
                set output to output & "TEL:" & ph & linefeed
            end repeat
            set output to output & "END:VCARD" & linefeed
        end if
    end repeat
    return output
end tell
"""

# ───────────────────────────────────────────────
# Heuristic filters (mirrors FilterConfig defaults)
# ───────────────────────────────────────────────

_MIN_DATE = "2020-01-01"
_MIN_CONVERSATION_MESSAGES = 2

_BLOCKED_STRINGS = [
    "GamePigeon message:",
    "Library/Messages/",
    "Find My:",
    "Started sharing location",
    "Stopped sharing location",
    "is inviting you to a scheduled Zoom meeting",
]

_ATTACHMENT_EXTS = (
    r"heic|jpg|jpe|jpeg|png|gif|bmp|tiff|webp|svg"
    r"|mov|mp4|avi|mkv|m4v"
    r"|mp3|aac|m4a|wav|caf|ogg"
    r"|pdf|docx?|xlsx?|pptx?|csv|txt|zip|rar"
    r"|vcf|pkpass"
)
_ATTACHMENT_RE = re.compile(
    r"^(<reply:\d+>)?\s*.+\." rf"({_ATTACHMENT_EXTS})" r"(\.\w+)?\s*$",
    re.IGNORECASE,
)
_ATTACHMENT_PREFIXED_RE = re.compile(
    r"^(<reply:\d+>)?\s*"
    r"(?:PNG|JPEG|JPG|GIF|TIFF|BMP|WEBP|MOV|MP4|PDF|HEIC)\s+"
    r".+\." rf"({_ATTACHMENT_EXTS})" r"(\.\w+)?\s*$",
    re.IGNORECASE,
)


def _is_attachment(text: str) -> bool:
    return bool(_ATTACHMENT_RE.match(text) or _ATTACHMENT_PREFIXED_RE.match(text))


def _contains_url(text: str) -> bool:
    patterns = ["http://", "https://", "www.", ".com/", ".org/", ".edu/"]
    return any(p in text.lower() for p in patterns)


def _is_non_text(content: str) -> bool:
    """Fast check: is this message non-textual content the model can't learn from?"""
    text = content.strip()
    if not text:
        return True
    if _is_attachment(text):
        return True
    for bs in _BLOCKED_STRINGS:
        if bs.lower() in text.lower():
            return True
    if _contains_url(text) and len(text.split()) <= 3:
        return True
    return False


def _has_duplicate_parts(text: str) -> bool:
    """Check for consecutive identical <msg> parts (iMessage sync artifacts)."""
    if "<msg>" not in text:
        return False
    parts = [p.strip() for p in text.split("<msg>")]
    return any(parts[i] and parts[i] == parts[i - 1] for i in range(1, len(parts)))


def _resolve_speaker(sender: str, contacts: dict[str, str] | None) -> str:
    """Resolve a phone number speaker to a name using contacts lookup."""
    if not contacts or not sender.startswith("+"):
        return sender
    digits = re.sub(r"[^\d]", "", sender)
    for key in [digits, digits.lstrip("1"), "1" + digits.lstrip("1")]:
        if key in contacts:
            return contacts[key]
    return sender


# ───────────────────────────────────────────────
# Export helpers
# ───────────────────────────────────────────────

def is_macos() -> bool:
    return platform.system() == "Darwin"


def has_imessage_exporter() -> bool:
    return shutil.which("imessage-exporter") is not None


def install_imessage_exporter() -> tuple[bool, str]:
    """Attempt install. Returns (success, message)."""
    if shutil.which("brew"):
        r = subprocess.run(["brew", "install", "imessage-exporter"], capture_output=True, text=True)
        if r.returncode == 0:
            return True, "Installed via Homebrew"
        return False, f"brew install failed: {r.stderr[:200]}"
    if shutil.which("cargo"):
        r = subprocess.run(["cargo", "install", "imessage-exporter"], capture_output=True, text=True)
        if r.returncode == 0:
            return True, "Installed via cargo"
        return False, f"cargo install failed: {r.stderr[:200]}"
    return False, "Neither brew nor cargo found. Install imessage-exporter manually."


def run_imessage_export(export_dir: Path, db_path: Path | None = None) -> tuple[bool, str]:
    """Run imessage-exporter. Returns (success, message)."""
    export_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["imessage-exporter", "--format", "txt", "--export-path", str(export_dir)]
    if db_path:
        cmd += ["--db-path", str(db_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)

    combined = (r.stdout or "") + (r.stderr or "")

    if "full disk access" in combined.lower() or "unable to read from chat database" in combined.lower():
        return False, (
            "Full Disk Access required. Go to System Settings > Privacy & Security > "
            "Full Disk Access and enable it for your terminal app (Terminal, iTerm2, etc.), "
            "then restart the terminal and try again."
        )

    if r.returncode != 0:
        return False, f"Export failed: {combined[:300]}"

    n = len(list(export_dir.glob("*.txt")))
    if n == 0:
        return False, f"Export ran but produced 0 files. Output: {combined[:300]}"

    return True, f"Exported {n} conversation files"


def find_iphone_backups() -> list[Path]:
    """Return paths to iPhone backups, most recent last."""
    backup_root = Path.home() / "Library" / "Application Support" / "MobileSync" / "Backup"
    if not backup_root.exists():
        return []
    backups = sorted(backup_root.iterdir(), key=lambda p: p.stat().st_mtime)
    return [b for b in backups if b.is_dir()]


def merge_exports(mac_dir: Path, ios_dir: Path, out_dir: Path) -> int:
    """Merge exports by keeping the longer file. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mac_files = {f.name: f for f in mac_dir.glob("*.txt")}
    ios_files = {f.name: f for f in ios_dir.glob("*.txt")}
    all_names = set(mac_files) | set(ios_files)
    for name in all_names:
        if name in mac_files and name in ios_files:
            src = mac_files[name] if mac_files[name].stat().st_size >= ios_files[name].stat().st_size else ios_files[name]
        elif name in mac_files:
            src = mac_files[name]
        else:
            src = ios_files[name]
        shutil.copy2(src, out_dir / name)
    return len(all_names)


# ───────────────────────────────────────────────
# Contacts
# ───────────────────────────────────────────────

def find_existing_vcf() -> Path | None:
    """Look for an existing VCF file in common locations."""
    candidates = [
        CONTACTS_VCF,
        WORKSPACE / "experiment_output" / "All Contacts.vcf",
        WORKSPACE / "data" / "contacts.vcf",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def export_contacts_via_applescript(dest: Path) -> tuple[int, str]:
    """Export contacts via AppleScript. Returns (count, message)."""
    import time as _time

    # Contacts app must be running before AppleScript can query it
    subprocess.run(["open", "-a", "Contacts"], capture_output=True)
    _time.sleep(3)

    try:
        result = subprocess.run(
            ["osascript", "-e", APPLESCRIPT_EXPORT_CONTACTS],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return 0, "AppleScript timed out"
    if result.returncode != 0:
        return 0, f"AppleScript failed: {result.stderr.strip()[:200]}"
    vcf_text = result.stdout
    if not vcf_text.strip():
        return 0, "No contacts with phone numbers found"
    dest.write_text(vcf_text, encoding="utf-8")
    n = vcf_text.count("BEGIN:VCARD")
    return n, f"Exported {n} contacts"


def build_contacts_lookup(vcf_path: Path | None) -> dict[str, str]:
    """Lightweight VCF parser -- returns phone->name dict."""
    if not vcf_path or not vcf_path.exists():
        return {}
    lookup: dict[str, str] = {}
    current_name = None
    for line in vcf_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("FN:"):
            current_name = line[3:].strip()
        elif (line.startswith("TEL") or line.startswith("item")) and ":" in line:
            phone = line.split(":")[-1].strip()
            digits = re.sub(r"[^\d]", "", phone)
            if current_name and len(digits) >= 10:
                lookup[digits] = current_name
                if len(digits) == 10:
                    lookup["1" + digits] = current_name
                elif digits.startswith("1") and len(digits) == 11:
                    lookup[digits[1:]] = current_name
        elif line == "END:VCARD":
            current_name = None
    return lookup


# ───────────────────────────────────────────────
# Status helpers
# ───────────────────────────────────────────────

def pick_best_export_dir() -> Path:
    """Return the merged dir if it exists and has files, otherwise Mac dir."""
    if MERGED_EXPORT_DIR.exists() and any(MERGED_EXPORT_DIR.glob("*.txt")):
        return MERGED_EXPORT_DIR
    return MAC_EXPORT_DIR


# ───────────────────────────────────────────────
# Conversation scanning
# ───────────────────────────────────────────────

def fast_parse_and_filter(file_path: Path, contacts: dict[str, str] | None) -> dict | None:
    """
    Fast parse of a single iMessage export with heuristic filtering.
    Returns None if excluded (group chat, no usable data).
    """
    ts_pattern = r"([A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M)"

    try:
        lines = file_path.read_text(errors="replace").splitlines()
    except Exception:
        return None

    messages: list[dict] = []
    speakers_seen: set[str] = set()
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        if raw_line.startswith("    ") or raw_line.startswith("\t"):
            i += 1
            continue
        line = raw_line.strip()
        ts_match = re.match(ts_pattern, line)
        if not ts_match:
            i += 1
            continue

        ts_end = ts_match.end(1)
        remainder = line[ts_end:].strip()
        if remainder and not remainder.startswith("("):
            i += 1
            continue

        ts_str = ts_match.group(1)
        i += 1
        if i >= len(lines):
            break
        sender = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        content = lines[i].strip()
        i += 1

        if not sender or sender.startswith("Edited "):
            continue
        if (content.startswith("This message")
                or content.startswith("Tapbacks:")
                or content.startswith("/Users/")
                or content.startswith("Edited to")
                or content.startswith("Edited ")):
            continue

        if sender != "Me":
            resolved = _resolve_speaker(sender, contacts)
            speakers_seen.add(resolved)

        messages.append({
            "speaker": sender if sender == "Me" else "Other",
            "content": content,
            "timestamp": ts_str,
        })

    if len(speakers_seen) > 1:
        return None
    if not messages:
        return None

    stem = file_path.stem
    display_name = stem
    if contacts and stem.startswith("+"):
        digits = re.sub(r"[^\d]", "", stem)
        for key in [digits, digits.lstrip("1"), "1" + digits.lstrip("1")]:
            if key in contacts:
                display_name = contacts[key]
                break
    elif len(speakers_seen) == 1:
        display_name = next(iter(speakers_seen))

    max_gap = timedelta(hours=48)
    fmt = "%b %d, %Y %I:%M:%S %p"

    conversations: list[list[dict]] = []
    current: list[dict] = []

    for msg in messages:
        try:
            ts = datetime.strptime(msg["timestamp"], fmt)
        except ValueError:
            continue
        msg["_ts"] = ts
        if not current:
            current.append(msg)
        elif ts - current[-1]["_ts"] > max_gap:
            conversations.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        conversations.append(current)

    total_raw = len(messages)
    training_examples = 0
    usable_conversations = 0
    last_msg_date = None

    # Track the latest timestamp across all messages
    for msg in messages:
        ts = msg.get("_ts")
        if ts and (last_msg_date is None or ts > last_msg_date):
            last_msg_date = ts

    for conv in conversations:
        if conv[0]["_ts"].strftime("%Y-%m-%d") < _MIN_DATE:
            continue
        text_msgs = [m for m in conv if not _is_non_text(m["content"])]
        if len(text_msgs) < _MIN_CONVERSATION_MESSAGES:
            continue
        usable_conversations += 1

        msg_idx = 0
        while msg_idx < len(conv):
            if conv[msg_idx]["speaker"] != "Me":
                msg_idx += 1
                continue
            me_parts: list[str] = []
            has_blocked = False
            while msg_idx < len(conv) and conv[msg_idx]["speaker"] == "Me":
                content = conv[msg_idx]["content"].strip()
                if content:
                    me_parts.append(content)
                    if _is_attachment(content):
                        has_blocked = True
                msg_idx += 1
            if not me_parts or has_blocked:
                continue
            target = "<msg>".join(me_parts)
            target_lower = target.lower()
            if any(bs.lower() in target_lower for bs in _BLOCKED_STRINGS):
                continue
            if _contains_url(target):
                continue
            if _is_attachment(target):
                continue
            if _has_duplicate_parts(target):
                continue
            training_examples += 1

    return {
        "file": str(file_path),
        "filename": file_path.name,
        "display_name": display_name,
        "raw_messages": total_raw,
        "training_examples": training_examples,
        "usable_conversations": usable_conversations,
        "total_conversations": len(conversations),
        "size_kb": round(file_path.stat().st_size / 1024, 1),
        "last_message_date": last_msg_date.strftime("%Y-%m-%d") if last_msg_date else None,
    }


def scan_all_conversations(export_dir: Path, contacts: dict[str, str] | None) -> tuple[list[dict], int, int]:
    """
    Scan all .txt files in export_dir with filtering.
    Returns (results, skipped_group, skipped_empty).
    """
    files = list(export_dir.glob("*.txt"))
    results: list[dict] = []
    skipped_group = 0
    skipped_empty = 0

    for f in files:
        info = fast_parse_and_filter(f, contacts)
        if info is None:
            stem = f.stem
            if "," in stem or " and " in stem:
                skipped_group += 1
            else:
                skipped_empty += 1
            continue
        if info["training_examples"] == 0:
            skipped_empty += 1
            continue
        results.append(info)

    results.sort(key=lambda x: -x["training_examples"])
    return results, skipped_group, skipped_empty


PERSONA_CONFIG_FILE = WORKSPACE / "persona_config.json"
OUTPUT_DIR = WORKSPACE / "experiment_output"
DEFAULT_DATASET_VERSION = "v4"


def _load_workspace_env() -> None:
    """Load repo-local .env values when available."""
    try:
        from dotenv import load_dotenv
        load_dotenv(WORKSPACE / ".env", override=False)
    except Exception:
        pass


def _user_model_base() -> str:
    from IMPersona2.model_config import model_for
    return model_for("user_model", "base")


def get_annotation_model() -> str:
    """Return the annotation model (configs/models.yaml, or ANNOTATION_MODEL in .env)."""
    from IMPersona2.model_config import model_for
    return model_for("annotation")


def annotation_provider_for_model(model: str) -> str:
    """Infer the annotation provider from a model name."""
    name = model.lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gpt"):
        return "openai"
    return "unknown"


def _has_configured_api_key(env_var: str) -> bool:
    value = os.environ.get(env_var, "").strip()
    if not value:
        return False
    return not value.startswith("your-") and not value.endswith("-here")


def get_annotation_key_status(model: str | None = None) -> dict:
    """Return the provider/key requirement for the selected annotation model."""
    _load_workspace_env()
    selected_model = model or get_annotation_model()
    from IMPersona2.model_config import model_for
    base_url = model_for("annotation", "base_url")
    if base_url:
        # OpenAI-compatible endpoint (see LLMAnnotator): OpenRouter needs
        # OPENROUTER_API_KEY; other servers (vLLM, ...) don't need a key.
        env_var = "OPENROUTER_API_KEY" if "openrouter.ai" in base_url else ""
        return {
            "model": selected_model,
            "provider": "openai-compatible",
            "env_var": env_var,
            "has_api_key": _has_configured_api_key(env_var) if env_var else True,
        }
    provider = annotation_provider_for_model(selected_model)
    if provider == "anthropic":
        env_var = "ANTHROPIC_API_KEY"
    elif provider == "openai":
        env_var = "OPENAI_API_KEY"
    else:
        return {
            "model": selected_model,
            "provider": provider,
            "env_var": "",
            "has_api_key": False,
            "error": (
                f"Unsupported annotation model '{selected_model}'. Use a claude* "
                "model with ANTHROPIC_API_KEY or a gpt* model with OPENAI_API_KEY."
            ),
        }
    return {
        "model": selected_model,
        "provider": provider,
        "env_var": env_var,
        "has_api_key": _has_configured_api_key(env_var),
    }


# ───────────────────────────────────────────────
# Persona config
# ───────────────────────────────────────────────

def load_persona() -> dict | None:
    if not PERSONA_CONFIG_FILE.exists():
        return None
    try:
        return json.loads(PERSONA_CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_persona(persona: dict) -> None:
    PERSONA_CONFIG_FILE.write_text(
        json.dumps(persona, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _modal_app_name() -> str:
    """Name of the deployed Modal app (matches modal_app.MODAL_APP_NAME)."""
    return "impersona"


# ───────────────────────────────────────────────
# Dataset generation
# ───────────────────────────────────────────────

# Thread-safe progress tracking for the generation step
_generation_progress: dict = {}
_generation_lock = __import__("threading").Lock()


def get_generation_status() -> list[dict]:
    """Return per-contact generation status."""
    selection = load_selection()
    if not selection:
        return []

    persona = load_persona()
    model_name = get_annotation_model()
    if DEFAULT_DATASET_VERSION:
        model_name = f"{model_name}_{DEFAULT_DATASET_VERSION}"

    statuses = []
    for entry in selection:
        name = entry.get("display_name", "?")
        files = entry.get("files", [entry.get("file", "")])

        # Check if all files for this contact have dataset.json
        all_done = True
        total_examples = 0
        for f in files:
            conv_id = Path(f).stem
            ds_path = OUTPUT_DIR / conv_id / model_name / "dataset.json"
            if ds_path.exists():
                try:
                    bundle = json.loads(ds_path.read_text())
                    total_examples += bundle.get("metadata", {}).get("num_examples", 0)
                except (json.JSONDecodeError, OSError):
                    all_done = False
            else:
                all_done = False

        # Check in-progress status
        with _generation_lock:
            progress = _generation_progress.get(name)

        if progress:
            status = progress.get("status", "pending")
            detail = progress.get("detail", "")
        elif all_done and total_examples > 0:
            status = "done"
            detail = f"{total_examples} examples"
        else:
            status = "pending"
            detail = ""
            # Check llm_cache.json for cached progress
            cached_counts = []
            for f in files:
                conv_id = Path(f).stem
                cache_path = OUTPUT_DIR / conv_id / model_name / "llm_cache.json"
                if cache_path.exists():
                    try:
                        cache = json.loads(cache_path.read_text())
                        n_rel = 1 if cache.get("relationship_summary") else 0
                        n_goals = len(cache.get("conversation_goals_by_hash", {}))
                        n_cot = len(cache.get("chain_of_thought", {}))
                        cached_counts.append((n_rel, n_goals, n_cot))
                    except (json.JSONDecodeError, OSError):
                        pass
            if cached_counts:
                total_rel = sum(c[0] for c in cached_counts)
                total_goals = sum(c[1] for c in cached_counts)
                total_cot = sum(c[2] for c in cached_counts)
                parts = []
                if total_rel:
                    parts.append("relationship")
                if total_goals:
                    parts.append(f"{total_goals} goals")
                if total_cot:
                    parts.append(f"{total_cot} CoTs")
                detail = f"Cached: {', '.join(parts)}"

        # Check if visualizer exists for this contact
        has_visualizer = False
        viz_conv_id = None
        for f in files:
            conv_id = Path(f).stem
            viz = OUTPUT_DIR / conv_id / model_name / "visualizer.html"
            if viz.exists():
                has_visualizer = True
                viz_conv_id = conv_id
                break

        statuses.append({
            "display_name": name,
            "files": files,
            "status": status,
            "detail": detail,
            "examples": total_examples,
            "has_visualizer": has_visualizer,
            "visualizer_path": f"/view/{viz_conv_id}" if has_visualizer else None,
        })

    return statuses


def _inject_renames(contacts: dict[str, str], selection: list[dict]) -> None:
    """
    Inject display_name overrides (renames from the select step) into the
    contacts lookup dict.

    For each selected contact, scans the raw export file to find the actual
    speaker labels used, then maps them to the user's chosen display_name.
    This ensures the training data uses the renamed name everywhere.

    The contacts dict is modified in-place: keys are added for speaker labels
    (as literal strings, not just phone digits) so that resolve_contact_name
    can find them.
    """
    ts_pattern = re.compile(r"[A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M")

    for entry in selection:
        display_name = entry.get("display_name", "")
        if not display_name:
            continue
        files = entry.get("files", [entry.get("file", "")])

        for f in files:
            fp = Path(f)
            if not fp.exists():
                continue

            # Quick scan to find non-Me speaker labels in this file
            try:
                lines = fp.read_text(errors="replace").splitlines()
            except OSError:
                continue

            speakers_found: set[str] = set()
            i = 0
            while i < len(lines):
                raw = lines[i]
                if raw.startswith("    ") or raw.startswith("\t"):
                    i += 1
                    continue
                line = raw.strip()
                if ts_pattern.match(line):
                    i += 1
                    if i < len(lines):
                        sender = lines[i].strip()
                        if (sender and sender != "Me"
                                and not sender.startswith("Edited ")):
                            speakers_found.add(sender)
                    i += 1
                    continue
                i += 1

            # Map each found speaker label to the display_name
            for speaker in speakers_found:
                if speaker == display_name:
                    continue
                # Add as literal key so resolve_contact_name can match it.
                # Also add the digits-only form for phone-style labels.
                contacts[speaker] = display_name
                digits = re.sub(r"[^\d]", "", speaker)
                if digits and len(digits) >= 10:
                    contacts[digits] = display_name
                    if len(digits) == 10:
                        contacts["1" + digits] = display_name
                    elif digits.startswith("1") and len(digits) == 11:
                        contacts[digits[1:]] = display_name


def run_generation(
    force: bool = False,
    on_progress=None,
    contacts_filter: list[str] | None = None,
) -> dict:
    """
    Run dataset generation for selected contacts.

    Args:
        force: Re-generate even if dataset.json exists.
        on_progress: Callback(name, status, detail) for progress updates.
        contacts_filter: If provided, only process contacts whose display_name
                         is in this list. If None, process all selected contacts.

    Returns a summary dict with counts and cost info.
    """
    import sys

    selection = load_selection()
    if not selection:
        return {"error": "No conversations selected"}

    if contacts_filter:
        filter_set = set(contacts_filter)
        selection = [s for s in selection if s.get("display_name") in filter_set]
        if not selection:
            return {"error": "No matching contacts found in selection"}

    persona = load_persona()
    if not persona:
        return {"error": "No persona configured"}

    key_status = get_annotation_key_status()
    if key_status.get("error"):
        return {"error": key_status["error"]}
    if not key_status["has_api_key"]:
        return {
            "error": (
                f"{key_status['env_var']} is not configured in .env for annotation "
                f"model {key_status['model']}"
            )
        }

    from IMPersona2.experiment_config import PipelineConfig
    from IMPersona2.generate_dataset import run_pipeline, _build_contacts_lookup

    config = PipelineConfig()

    # Apply persona
    config.persona.name = persona.get("name", "User")
    config.persona.full_name = persona.get("full_name", persona.get("name", "User"))
    config.persona.nicknames = persona.get("nicknames", [])

    # Apply model settings
    config.llm.model = key_status["model"]
    config.dataset_version = DEFAULT_DATASET_VERSION
    config.filtering.max_input_turns = 30

    # Contacts lookup
    vcf = find_existing_vcf()
    contacts = _build_contacts_lookup(str(vcf)) if vcf else {}

    # Inject renames from the select step: scan each file's actual speaker
    # labels and map them to the user's chosen display_name so the training data uses
    # the correct name everywhere (system prompt, history, metadata).
    _inject_renames(contacts, selection)

    model_name = config.llm.model
    if config.dataset_version:
        model_name = f"{model_name}_{config.dataset_version}"

    # Build list of (display_name, input_file, output_file, cache_file)
    jobs: list[tuple[str, str, str, str]] = []
    skipped = 0

    for entry in selection:
        name = entry.get("display_name", "?")
        files = entry.get("files", [entry.get("file", "")])

        for f in files:
            conv_id = Path(f).stem
            conv_dir = OUTPUT_DIR / conv_id / model_name
            output_file = str(conv_dir / "dataset.json")
            cache_file = str(conv_dir / "llm_cache.json")

            if os.path.exists(output_file) and not force:
                try:
                    bundle = json.loads(Path(output_file).read_text())
                    n = bundle.get("metadata", {}).get("num_examples", 0)
                    skipped += 1
                    _update_progress(name, "done", f"{n} examples (cached)")
                    if on_progress:
                        on_progress(name, "done", f"{n} examples (cached)")
                    continue
                except (json.JSONDecodeError, OSError):
                    pass

            if not os.path.exists(f):
                _update_progress(name, "error", f"File not found: {f}")
                if on_progress:
                    on_progress(name, "error", f"File not found: {f}")
                continue

            conv_dir.mkdir(parents=True, exist_ok=True)
            jobs.append((name, f, output_file, cache_file))

    if not jobs:
        return {"processed": 0, "skipped": skipped, "total_cost": 0}

    total_cost = 0.0
    processed = 0
    errors = 0

    _STEP_LABELS = {
        "1": "Parsing messages",
        "2": "Analyzing relationship",
        "3": "Generating goals",
        "4": "Building training examples",
        "5": "Generating chain-of-thought",
        "6": "Computing stats & saving",
    }

    for display_name, input_file, output_file, cache_file in jobs:
        _update_progress(display_name, "running", "Starting...")
        if on_progress:
            on_progress(display_name, "running", "Starting...")

        # Intercept stdout to capture [N/6] step markers from run_pipeline
        _real_stdout = sys.stdout
        _capture_name = display_name

        class _StepCapture:
            def __init__(self, real, name):
                self._real = real
                self._name = name
                self._step_re = re.compile(r"\[(\d)/6\]")
                self._cot_re = re.compile(r"Generating CoT \[(\d+)/(\d+)\] \((\d+)%\)")
                self._goals_re = re.compile(r"goals?.+?(\d+)/(\d+)", re.IGNORECASE)
                self._notifying = False

            def _notify(self, detail):
                _update_progress(self._name, "running", detail)
                if on_progress:
                    self._notifying = True
                    try:
                        on_progress(self._name, "running", detail)
                    finally:
                        self._notifying = False

            def write(self, text):
                self._real.write(text)
                # Skip parsing while a callback is printing, to avoid recursion.
                if self._notifying:
                    return

                # CoT progress: "Generating CoT [N/M] (X%)"
                cot_m = self._cot_re.search(text)
                if cot_m:
                    done, total, pct = cot_m.group(1), cot_m.group(2), cot_m.group(3)
                    self._notify(f"[5/6] Chain-of-thought: {done}/{total} ({pct}%)")
                    return

                # Step markers: "[N/6]"
                m = self._step_re.search(text)
                if m:
                    step_num = m.group(1)
                    label = _STEP_LABELS.get(step_num, f"Step {step_num}/6")
                    # Enrich with counts from the line itself
                    extra = ""
                    if step_num == "5":
                        new_m = re.search(r"\((\d+) new, (\d+) cached", text)
                        if new_m:
                            extra = f" ({new_m.group(1)} new, {new_m.group(2)} cached)"
                    elif step_num == "3":
                        conv_m = re.search(r"\((\d+) conversations\)", text)
                        if conv_m:
                            extra = f" ({conv_m.group(1)} conversations)"
                    self._notify(f"[{step_num}/6] {label}{extra}")

            def flush(self):
                self._real.flush()

            def __getattr__(self, attr):
                return getattr(self._real, attr)

        sys.stdout = _StepCapture(_real_stdout, _capture_name)
        try:
            examples, filter_stats, dataset_stats, llm_usage = run_pipeline(
                config,
                input_file=input_file,
                output_file=output_file,
                llm_cache_file=cache_file,
                contacts=contacts,
            )
            n = len(examples)
            cost = llm_usage.get("total_cost_usd", 0) if llm_usage else 0
            total_cost += cost
            processed += 1

            # Generate visualizer HTML alongside the dataset
            _generate_visualizer(output_file)

            _update_progress(display_name, "done", f"{n} examples (${cost:.2f})")
            if on_progress:
                on_progress(display_name, "done", f"{n} examples (${cost:.2f})")
        except Exception as e:
            errors += 1
            tb = traceback.format_exc()
            sys.stdout = _real_stdout
            print(f"\n  [ERR] {display_name}: {e}\n{tb}")
            with open(GENERATION_ERROR_LOG, "a") as _log:
                _log.write(f"\n[{datetime.now().isoformat()}] {display_name}\n{tb}")
            _update_progress(display_name, "error", str(e)[:200])
            if on_progress:
                on_progress(display_name, "error", str(e)[:200])
        finally:
            sys.stdout = _real_stdout

    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "total_cost": round(total_cost, 2),
    }


def _generate_visualizer(dataset_path: str) -> None:
    """Generate a visualizer.html alongside the dataset.json."""
    try:
        from IMPersona2.dataset_visualizer import load_dataset_bundle, build_html
        examples, metadata = load_dataset_bundle(dataset_path)
        html = build_html(examples, metadata, dataset_path)
        viz_path = str(Path(dataset_path).parent / "visualizer.html")
        Path(viz_path).write_text(html, encoding="utf-8")
    except Exception:
        pass


def _update_progress(name: str, status: str, detail: str) -> None:
    with _generation_lock:
        _generation_progress[name] = {"status": status, "detail": detail}


def load_selection() -> list[dict] | None:
    """Load saved selection, or None if not present."""
    if not SELECTED_CONVOS_FILE.exists():
        return None
    try:
        return json.loads(SELECTED_CONVOS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_selection(selected: list[dict]) -> None:
    """Save selection to disk."""
    SELECTED_CONVOS_FILE.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ───────────────────────────────────────────────
# Training (Modal)
# ───────────────────────────────────────────────

TRAINING_DATA_FILE = WORKSPACE / "training_data.json"
TRAIN_RUNS_DIR = WORKSPACE / "train_output"


def combine_datasets_to_jsonl(output_path: Path | None = None) -> dict:
    """
    Collect all completed dataset.json files and combine into
    Alpaca format (list of instruction/input/output dicts).

    Returns {"path": str, "examples": int, "contacts": int} or {"error": str}.
    """
    output_path = output_path or TRAINING_DATA_FILE
    selection = load_selection()
    if not selection:
        return {"error": "No conversations selected"}

    model_name = get_annotation_model()
    if DEFAULT_DATASET_VERSION:
        model_name = f"{model_name}_{DEFAULT_DATASET_VERSION}"

    all_examples = []
    contacts_included = 0

    for entry in selection:
        files = entry.get("files", [entry.get("file", "")])
        for f in files:
            conv_id = Path(f).stem
            ds_path = OUTPUT_DIR / conv_id / model_name / "dataset.json"
            if not ds_path.exists():
                continue
            try:
                bundle = json.loads(ds_path.read_text())
                examples = bundle.get("examples", [])
                all_examples.extend(examples)
                if examples:
                    contacts_included += 1
            except (json.JSONDecodeError, OSError):
                continue

    if not all_examples:
        return {"error": "No generated datasets found"}

    alpaca_data = [
        {
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "output": ex.get("output", ""),
        }
        for ex in all_examples
    ]
    output_path.write_text(
        json.dumps(alpaca_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "path": str(output_path),
        "examples": len(alpaca_data),
        "contacts": contacts_included,
    }


# ───────────────────────────────────────────────
# Memory corpus (arena memory-baseline data layer)
# ───────────────────────────────────────────────

MEMORY_CORPUS_DIR = WORKSPACE / ".memory_corpus"
MEMORY_CORPUS_FILE = MEMORY_CORPUS_DIR / "conversations.json"


def resolve_memory_corpus_inputs() -> tuple[list[str], list[str], str | None]:
    """Resolve which raw iMessage files feed into the memory corpus.

    A file is *included* only when BOTH:
      1. It appears in selected_conversations.json (the user selected it)
      2. The corresponding dataset.json exists under experiment_output/
         (the user actually ran the training-data generation pipeline on it)

    This ensures the memory baseline operates over exactly the contacts the
    IMPersona model was trained on — no extra context the model would not
    have learned from.

    Returns:
        (included_files, excluded_files, contacts_vcf_path)
        ``contacts_vcf_path`` is ``None`` if no vCard could be found (the
        builder falls back to whatever speaker labels the exports already
        carry).
    """
    selection = load_selection() or []

    model_name = get_annotation_model()
    if DEFAULT_DATASET_VERSION:
        model_name = f"{model_name}_{DEFAULT_DATASET_VERSION}"

    included: list[str] = []
    excluded: list[str] = []
    for entry in selection:
        files = entry.get("files", [entry.get("file", "")])
        for f in files:
            if not f:
                continue
            conv_id = Path(f).stem
            ds_path = OUTPUT_DIR / conv_id / model_name / "dataset.json"
            if ds_path.exists():
                included.append(f)
            else:
                excluded.append(f)

    # Prefer the pipeline's standard vCard location; fall back to a
    # repo-root contacts.vcf.
    vcf_candidates = [
        WORKSPACE / "experiment_output" / "All Contacts.vcf",
        WORKSPACE / "contacts.vcf",
    ]
    contacts_vcf = next((str(p) for p in vcf_candidates if p.exists()), None)

    return included, excluded, contacts_vcf


def upload_training_file() -> dict:
    """Prepare training data for Modal (just validates the file exists)."""
    if not TRAINING_DATA_FILE.exists():
        return {"error": "Training data file not found. Combine datasets first."}

    try:
        data = json.loads(TRAINING_DATA_FILE.read_text())
        size_mb = TRAINING_DATA_FILE.stat().st_size / 1024 / 1024
        _save_training_state({"uploaded": True})
        return {"ok": True, "examples": len(data), "size_mb": round(size_mb, 1)}
    except Exception as e:
        return {"error": str(e)[:300]}


def submit_training_job(
    run_label: str | None = None,
    model_name: str | None = None,
    adapter_name: str | None = None,
    dataset_path: str | None = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 64,
    lora_alpha: int = 128,
) -> dict:
    """Start a training job on Modal as a detached subprocess.

    Spawns run_sft_worker.py with start_new_session=True so it survives
    the CLI exiting. State is tracked via state.json in the
    run directory.
    """
    import sys

    data_path = Path(dataset_path) if dataset_path else TRAINING_DATA_FILE
    if not data_path.exists():
        return {"error": f"Training data not found: {data_path}"}

    label = run_label or (model_name.split("/")[-1] if model_name else "user-model")
    timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = TRAIN_RUNS_DIR / f"{label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_adapter = adapter_name or f"adapter-{label}"

    try:
        num_examples = len(json.loads(data_path.read_text()))
    except Exception:
        num_examples = None

    from IMPersona2.model_config import template_for
    effective_model = model_name or _user_model_base()
    template = template_for("user_model", effective_model)

    config = {
        "modal_app_name": _modal_app_name(),
        "model_name": effective_model,
        "adapter_name": effective_adapter,
        "template": template,
        "training_data": str(data_path),
        "num_examples": num_examples,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.05,
        "lora_target": "all",
        "cutoff_len": 4096,
        "bf16": True,
        "gpu": "H100",
        "timestamp": timestamp,
        "label": label,
        "engine": "llamafactory",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log_file = open(run_dir / "worker.log", "w")
    subprocess.Popen(
        [sys.executable, str(WORKSPACE / "run_sft_worker.py"), "--run-dir", str(run_dir)],
        cwd=str(WORKSPACE),
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    return {"ok": True, "run_dir": run_dir.name, "message": f"Training submitted ({effective_model})"}


def _get_active_train_run_dir() -> Path | None:
    """Find the most recent training run directory."""
    if not TRAIN_RUNS_DIR.exists():
        return None
    dirs = sorted(TRAIN_RUNS_DIR.iterdir(), reverse=True)
    return dirs[0] if dirs else None


def list_train_runs() -> list[dict]:
    """List all training runs with their config and state."""
    if not TRAIN_RUNS_DIR.exists():
        return []
    runs = []
    for run_dir in sorted(TRAIN_RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        state, config = {}, {}
        state_file = run_dir / "state.json"
        config_file = run_dir / "config.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        runs.append({
            "run_id": run_dir.name,
            "status": state.get("status", "unknown"),
            "detail": state.get("detail"),
            "model_name": config.get("model_name") or _user_model_base(),
            "adapter_name": config.get("adapter_name", "adapter-user-model"),
            "template": config.get("template") or "",
            "label": config.get("label", run_dir.name),
            "gpu": config.get("gpu", ""),
            "timestamp": config.get("timestamp", ""),
            "has_lf_config": (run_dir / "llamafactory_config.json").exists(),
        })
    return runs


def get_training_status() -> dict:
    """Check the status of the most recent training run.

    Reflects the newest run directory only, so a freshly submitted run (still
    running, or failed immediately) is never masked by an older completed one.
    """
    import time as _time

    state = load_training_state()

    data_exists = TRAINING_DATA_FILE.exists()
    data_examples = 0
    if data_exists:
        try:
            data_examples = len(json.loads(TRAINING_DATA_FILE.read_text()))
        except Exception:
            pass

    result = {
        "jsonl_exists": data_exists,
        "jsonl_examples": data_examples,
        "stage": "none",
        "status": state.get("status"),
        "error": state.get("error"),
        "detail": state.get("detail"),
        "elapsed": None,
        "log_tail": None,
    }

    started_at = state.get("started_at")
    if started_at and state.get("status") == "running":
        result["elapsed"] = round(_time.time() - started_at)
        pid = state.get("pid")
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                # The worker is gone, but if it spawned a detached Modal call
                # the GPU job keeps running independently. Don't report a hard
                # error — surface that it's reattachable instead.
                fc_id = state.get("function_call_id")
                if fc_id:
                    result["stage"] = "training"
                    result["status"] = (
                        f"Worker detached; training continues on Modal (call {fc_id}). "
                        f"Reattach: python run_sft_worker.py --run-dir <run_dir>"
                    )
                    return result
                result["stage"] = "error"
                result["error"] = "Training worker process died. Check worker.log in the run directory."
                return result

    if not data_exists:
        result["stage"] = "none"
        return result

    if not state.get("uploaded"):
        result["stage"] = "jsonl_ready"
        return result

    status = state.get("status", "")
    if status == "running":
        result["stage"] = "training"
        result["status"] = state.get("detail", "Training on Modal (GPU)...")
    elif status == "completed":
        result["stage"] = "completed"
        result["status"] = "completed"
    elif status == "error":
        result["stage"] = "error"
        result["error"] = state.get("error", "Training failed")
    else:
        result["stage"] = "uploaded"

    return result


def load_training_state() -> dict:
    """Load training state from the most recent run directory."""
    run_dir = _get_active_train_run_dir()
    if run_dir:
        state_file = run_dir / "state.json"
        if state_file.exists():
            try:
                return json.loads(state_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def _save_training_state(state: dict) -> None:
    run_dir = _get_active_train_run_dir()
    if not run_dir:
        run_dir = TRAIN_RUNS_DIR / "default"
        run_dir.mkdir(parents=True, exist_ok=True)
    state_file = run_dir / "state.json"
    existing = {}
    if state_file.exists():
        try:
            existing = json.loads(state_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    existing.update(state)
    state_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")


# ───────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────

EVAL_RUNS_DIR = WORKSPACE / "eval_output"
EVAL_SCENARIOS_FILE = WORKSPACE / "IMPersona2" / "eval_scenarios.json"


def get_eval_scenarios() -> list[dict]:
    """Return list of available eval scenarios."""
    if not EVAL_SCENARIOS_FILE.exists():
        return []
    try:
        return json.loads(EVAL_SCENARIOS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _list_eval_runs() -> list[dict]:
    """List all eval runs sorted by most recent first."""
    if not EVAL_RUNS_DIR.exists():
        return []
    runs = []
    for run_dir in sorted(EVAL_RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        state_file = run_dir / "state.json"
        config_file = run_dir / "config.json"
        state = {}
        config = {}
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        results_file = run_dir / "results.json"
        evaluations = {}
        num_conversations = 0
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                evaluations = data.get("evaluations", {})
                num_conversations = len(data.get("conversations", []))
            except (json.JSONDecodeError, OSError):
                pass

        runs.append({
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": state.get("status", "unknown"),
            "detail": state.get("detail"),
            "progress": state.get("progress"),
            "total": state.get("total"),
            "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at"),
            "error": state.get("error"),
            "pid": state.get("pid"),
            "ai_partner_model": config.get("ai_partner_modal_model") or config.get("ai_partner_model", "?"),
            "num_scenarios": config.get("num_runs", 1) * len(get_eval_scenarios()) if config.get("quick") is False and config.get("category") is None else None,
            "run_label": config.get("run_label", ""),
            "evaluations": evaluations,
            "num_conversations": num_conversations,
            "has_viewer": (run_dir / "viewer.html").exists(),
        })
    return runs


def get_eval_status(run_id: str | None = None) -> dict:
    """Check evaluation status for a specific run or the most recent one."""
    import time as _time

    if run_id:
        run_dir = EVAL_RUNS_DIR / run_id
    else:
        runs = _list_eval_runs()
        if not runs:
            return {"stage": "idle", "status": None, "error": None,
                    "progress": None, "total": None, "elapsed": None, "results_dir": None}
        run_dir = Path(runs[0]["run_dir"])

    state_file = run_dir / "state.json"
    if not state_file.exists():
        return {"stage": "idle", "status": None, "error": None,
                "progress": None, "total": None, "elapsed": None, "results_dir": None}

    try:
        state = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return {"stage": "idle", "status": None, "error": None,
                "progress": None, "total": None, "elapsed": None, "results_dir": None}

    result = {
        "stage": state.get("status", "idle"),
        "status": state.get("detail"),
        "error": state.get("error"),
        "progress": state.get("progress"),
        "total": state.get("total"),
        "elapsed": None,
        "results_dir": str(run_dir),
        "run_id": run_dir.name,
    }

    started_at = state.get("started_at")
    if started_at and state.get("status") == "running":
        result["elapsed"] = round(_time.time() - started_at)

        pid = state.get("pid")
        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                result["stage"] = "error"
                result["error"] = "Eval worker process died. Check worker.log in the run directory."

    return result


def list_eval_runs() -> list[dict]:
    """Public API for listing eval runs."""
    return _list_eval_runs()


def submit_eval_job(
    ai_partner_model: str | None = None,
    ai_partner_modal_model: str | None = None,
    ai_partner_modal_adapter: str | None = None,
    ai_partner_inference_class: str | None = None,
    category: str | None = None,
    quick: bool = False,
    num_runs: int = 1,
    max_messages: int = 20,
    run_label: str | None = None,
    impersona_model_name: str | None = None,
    impersona_adapter_name: str | None = None,
    impersona_template: str | None = None,
    parallel_conversations: int = 50,
    inference_class: str = "VLLMInference",
    ai_partner_enable_thinking: bool | None = None,
    critic_mode: bool = False,
    include_prior_cot: bool = False,
    resume_run_id: str | None = None,
) -> dict:
    """Start an evaluation job as a detached subprocess (survives CLI shutdown).

    When ``resume_run_id`` is given, the existing run directory is reused as-is
    (its config and any ``results_partial.json`` are kept) and the worker picks
    up where it left off instead of creating a fresh run. All other arguments
    are ignored in that case since the original config governs the resumed run.
    """
    import subprocess
    import sys

    persona = load_persona()
    if not persona:
        return {"error": "No persona configured"}

    if resume_run_id:
        run_dir = EVAL_RUNS_DIR / resume_run_id
        if not run_dir.exists() or not (run_dir / "config.json").exists():
            return {"error": f"Cannot resume '{resume_run_id}': run directory or config.json not found"}
        log_mode = "a"
    else:
        if not impersona_model_name or not impersona_adapter_name:
            runs = list_train_runs()
            completed = [r for r in runs if r["status"] == "completed"]
            if completed:
                latest = completed[0]
                impersona_model_name = impersona_model_name or latest["model_name"]
                impersona_adapter_name = impersona_adapter_name or latest["adapter_name"]
                impersona_template = impersona_template or latest["template"]

        label = run_label or "eval"
        timestamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = EVAL_RUNS_DIR / f"{label}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "modal_app_name": _modal_app_name(),
            "persona_name": persona.get("name", "User"),
            "ai_partner_model": ai_partner_model,
            "ai_partner_modal_model": ai_partner_modal_model,
            "ai_partner_modal_adapter": ai_partner_modal_adapter or "",
            "impersona_model_name": impersona_model_name or _user_model_base(),
            "impersona_adapter_name": impersona_adapter_name or "adapter-user-model",
            "impersona_template": impersona_template or "",  # "" = detected from the model name
            "parallel_conversations": parallel_conversations,
            "inference_class": inference_class,
            "ai_partner_inference_class": ai_partner_inference_class or "VLLMPartner",
            "ai_partner_enable_thinking": ai_partner_enable_thinking,
            "category": category,
            "quick": quick,
            "num_runs": num_runs,
            "max_messages": max_messages,
            "run_label": label,
            "critic_mode": critic_mode,
            "include_prior_cot": include_prior_cot,
        }

        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        log_mode = "w"

    log_file = open(run_dir / "worker.log", log_mode)
    subprocess.Popen(
        [sys.executable, str(WORKSPACE / "run_eval_worker.py"), "--run-dir", str(run_dir)],
        cwd=str(WORKSPACE),
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )

    verb = "resumed" if resume_run_id else "started"
    return {"ok": True, "message": f"Evaluation {verb} (runs in background)", "run_id": run_dir.name}
