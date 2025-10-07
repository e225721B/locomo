#!/usr/bin/env python3
"""
Simple converter: out/<session_dir>/agent_a.json -> out/agent_a_transcript.md
Usage: python3 scripts/export_transcript.py <input_json_path> <output_md_path>
"""
import sys
import json
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def escape_md(s: str) -> str:
    return s.replace('`', '\`')


def session_to_md(name, persona_summary, data):
    lines = []
    header = f"# Transcript for {name}\n"
    lines.append(header)
    if persona_summary:
        lines.append(f"**Persona:** {persona_summary}\n")

    # find sessions by keys like session_1_date_time, session_1, session_1_summary
    session_nums = []
    for k in data.keys():
        if k.startswith('session_') and k.endswith('_date_time'):
            try:
                n = int(k.split('_')[1])
                session_nums.append(n)
            except Exception:
                pass
    session_nums = sorted(set(session_nums))

    for n in session_nums:
        date_key = f'session_{n}_date_time'
        sess_key = f'session_{n}'
        summary_key = f'session_{n}_summary'
        date = data.get(date_key, '')
        lines.append(f"\n## Session {n} — {date}\n")
        # transcript
        items = data.get(sess_key, [])
        for it in items:
            sp = it.get('speaker') or it.get('speaker_name') or 'Unknown'
            text = it.get('clean_text') or it.get('text') or it.get('raw_text') or ''
            text = text.strip()
            # remove trailing [END] markers in display
            if text.endswith('[END]'):
                text = text[:-5].strip()
            lines.append(f"- **{sp}**: {escape_md(text)}")
        # facts
        facts = data.get(f'session_{n}_facts', {})
        if facts:
            lines.append('\n**Extracted facts:**')
            for who, arr in facts.items():
                lines.append(f"- {who}:")
                if arr:
                    for f in arr:
                        lines.append(f"  - {escape_md(str(f))}")
                else:
                    lines.append("  - (none)")
        # relationships (if present)
        rel = data.get(f'session_{n}_relationships')
        if rel and isinstance(rel, dict):
            lines.append('\n**Relationship scores:**')
            def fmt(d):
                return f"intimacy={d.get('intimacy','-')}, power={d.get('power','-')}, social_distance={d.get('social_distance','-')}, trust={d.get('trust','-')}"
            # Prefer name-aware display if available
            by = rel.get('by_speaker')
            if isinstance(by, dict) and by:
                for speaker_name, obj in by.items():
                    toward = obj.get('toward', '?')
                    scores = obj.get('scores', {})
                    lines.append(f"- {speaker_name} → {toward}: {fmt(scores)}")
            else:
                a2b = rel.get('a_to_b') or {}
                b2a = rel.get('b_to_a') or {}
                lines.append(f"- A→B: {fmt(a2b)}")
                lines.append(f"- B→A: {fmt(b2a)}")

        # summary
        summary = data.get(summary_key)
        if summary:
            lines.append('\n**Session summary:**')
            lines.append(summary.strip())

    return '\n'.join(lines)


def find_agent_jsons(root: Path):
    return list(root.glob('**/agent_*.json'))


def process_file(inp: Path, outp: Path):
    data = load_json(str(inp))
    name = data.get('name', inp.stem)
    persona = data.get('persona_summary', '')
    md = session_to_md(name, persona, data)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(md, encoding='utf-8')
    print(f'Wrote {outp}')


def main():
    # No args: discover all out/**/agent_*.json and write transcripts next to them
    if len(sys.argv) == 1:
        root = Path('.').resolve()
        agent_files = find_agent_jsons(root / 'out')
        if not agent_files:
            print('No agent_*.json files found under out/.')
            sys.exit(0)
        for f in agent_files:
            outp = f.with_name(f.stem + '_transcript.md')
            process_file(f, outp)
        return

    # Two args: explicit input and output
    if len(sys.argv) < 3:
        print('Usage: export_transcript.py <input_json> <output_md> OR run with no args to process out/**/agent_*.json')
        sys.exit(2)
    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    if not inp.exists():
        print(f'Input not found: {inp}')
        sys.exit(1)
    process_file(inp, outp)


if __name__ == '__main__':
    main()
