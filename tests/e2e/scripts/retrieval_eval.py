# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Recall dei capitoli di evidenza (Pinocchio, italiano): metrica di retrieval deterministica, a costo zero.

Uso:
    GRAPHRAG_DUMP_CONTEXT=/tmp/ctx.jsonl cargo run -p graphrag-cli -- bench ...   # produce il dump
    uv run tests/e2e/scripts/retrieval_eval.py --dump /tmp/ctx.jsonl --gold tests/e2e/data/it/gold.json \
        [--book tests/e2e/data/it/pinocchio.txt] [--out tests/e2e/results/retrieval_eval/<nome>.json]

Ogni riga del dump e' {"query", "context", "prompt"} (vedi dump_answer_context in graphrag-core/src/graphrag/ask.rs).
Il contesto e' diviso in parti ("\\n\\n---\\n\\n"); si tengono solo le parti-chunk ("[Chunk | ...]" e "[Entity: X | ...]",
quest'ultima porta il testo di un chunk), si toglie la riga d'intestazione e si cerca il testo nel libro (spazi
normalizzati, finestre di 100/80/60 caratteri). Un chunk a cavallo di due capitoli conta per entrambi.

Per domanda: recall = |evidence_chapters ∩ capitoli recuperati| / |evidence_chapters|.
"Almeno uno" = domande con almeno un capitolo di evidenza nel contesto: e' la misura giusta per le domande a
risposta singola le cui note citano un intervallo di capitoli (es. "cap. XII-XVIII").
Restano fuori dalla media (e vengono stampate): domande gold senza riga nel dump e domande senza
`evidence_chapters` (lista vuota). Se una query compare piu' volte nel dump vale l'ultima riga.
I chunk non trovati nel libro sono contati e riportati, mai scartati in silenzio.
I chunk-riassunto (testo che inizia con "[Riassunto", scritti da `chapter_summaries`) non sono testo del libro:
si contano a parte ("riassunti"), non sono "non trovati" e non contano come evidenza di un capitolo.
"""
import argparse
import bisect
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PART_SEP = "\n\n---\n\n"
HEADER_RE = re.compile(r"^\[([^\]\n]*)\]\n")
CHAPTER_RE = re.compile(r"(?m)^([IVXL]+)\.[ \t]*$")
WINDOWS = (100, 80, 60)
ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50}


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).split())


def roman(s: str) -> int:
    total = 0
    for i, c in enumerate(s):
        nxt = ROMAN[s[i + 1]] if i + 1 < len(s) else 0
        total += -ROMAN[c] if ROMAN[c] < nxt else ROMAN[c]
    return total


class Book:
    """Libro normalizzato (spazi) con l'inizio di ogni capitolo nello spazio normalizzato."""

    def __init__(self, text: str):
        marks = list(CHAPTER_RE.finditer(text))
        nums = [roman(m.group(1)) for m in marks]
        if nums != list(range(1, len(nums) + 1)) or not nums:
            raise SystemExit(f"intestazioni dei capitoli inattese nel libro: {nums}")
        bounds = [0] + [m.start() for m in marks] + [len(text)]
        pieces = [norm(text[bounds[i]:bounds[i + 1]]) for i in range(len(bounds) - 1)]
        self.starts: list[int] = []  # inizio normalizzato del capitolo i (indice 0 = prima del cap. I)
        pos = 0
        for p in pieces:
            self.starts.append(pos)
            pos += len(p) + 1
        self.text = " ".join(pieces)
        self.n_chapters = len(nums)

    def chapter_at(self, pos: int) -> int:
        return bisect.bisect_right(self.starts, pos) - 1  # 0 = frontespizio

    def locate(self, chunk: str) -> set[int] | None:
        """Capitoli toccati dal chunk, o None se il testo non si trova nel libro."""
        c = norm(chunk)
        if not c:
            return None
        for width in WINDOWS:
            if len(c) <= width:
                offs = [0]
                width = len(c)
            else:
                last = len(c) - width
                offs = sorted({0, last // 4, last // 2, 3 * last // 4, last})
            spans = []
            for o in offs:
                p = self.text.find(c[o:o + width])
                if p >= 0:
                    spans.append((p - o, p + width))
            if spans:
                lo = max(min(sp[0] for sp in spans), 0)
                hi = max(sp[1] for sp in spans)
                first, last_ = self.chapter_at(lo), self.chapter_at(max(hi - 1, lo))
                return {ch for ch in range(first, last_ + 1) if ch >= 1}
        return None


def split_chunks(context: str) -> tuple[list[str], int, int]:
    """(testi dei chunk, numero di parti non-chunk saltate, numero di chunk-riassunto)."""
    chunks, skipped, summaries = [], 0, 0
    for part in context.split(PART_SEP):
        m = HEADER_RE.match(part)
        if m and (m.group(1).startswith("Chunk") or m.group(1).startswith("Entity:")):
            text = part[m.end():]
            if text.startswith("[Riassunto"):
                summaries += 1
            else:
                chunks.append(text)
        else:
            skipped += 1
    return chunks, skipped, summaries


def evaluate(dump_path: Path, gold_path: Path, book: Book) -> dict:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))["questions"]
    dump: dict[str, str] = {}
    for ln, line in enumerate(dump_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            dump[row["query"]] = row["context"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"riga {ln} del dump ignorata: {e}", file=sys.stderr)
    rows, no_dump, no_evidence = [], [], []
    for q in gold:
        evidence = sorted(set(q.get("evidence_chapters") or []))
        if q["question"] not in dump:
            no_dump.append(q["id"])
            continue
        if not evidence:
            no_evidence.append(q["id"])
            continue
        chunks, skipped, summaries = split_chunks(dump[q["question"]])
        retrieved: set[int] = set()
        unmatched = 0
        for ch in chunks:
            found = book.locate(ch)
            if found is None:
                unmatched += 1
            else:
                retrieved |= found
        hit = [c for c in evidence if c in retrieved]
        rows.append({
            "id": q["id"], "evidence": evidence, "recall": len(hit) / len(evidence),
            "missing": [c for c in evidence if c not in retrieved],
            "retrieved_chapters": sorted(retrieved), "chunks": len(chunks),
            "unmatched_chunks": unmatched, "non_chunk_parts": skipped, "summaries": summaries,
        })
    mean = sum(r["recall"] for r in rows) / len(rows) if rows else None
    return {
        "dump": str(dump_path), "gold": str(gold_path), "questions": rows,
        "mean_recall": mean, "evaluated": len(rows),
        "full_recall": sum(1 for r in rows if r["recall"] == 1.0),
        "any_hit": sum(1 for r in rows if r["recall"] > 0),
        "gold_without_dump": no_dump, "gold_without_evidence": no_evidence,
        "unmatched_chunks": sum(r["unmatched_chunks"] for r in rows),
        "total_chunks": sum(r["chunks"] for r in rows),
        "total_summaries": sum(r["summaries"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True, type=Path, help="JSONL di GRAPHRAG_DUMP_CONTEXT")
    ap.add_argument("--gold", required=True, type=Path, help="gold.json o gold_graph.json (con evidence_chapters)")
    ap.add_argument("--book", type=Path, default=ROOT / "tests/e2e/data/it/pinocchio.txt")
    ap.add_argument("--out", type=Path, help="percorso del JSON di output (es. tests/e2e/results/retrieval_eval/x.json)")
    a = ap.parse_args()
    res = evaluate(a.dump, a.gold, Book(a.book.read_text(encoding="utf-8")))
    print("| Id | Recall | Mancanti | Chunk | Non trovati | Riassunti |")
    print("|---|---|---|---|---|---|")
    for r in res["questions"]:
        miss = ",".join(map(str, r["missing"])) or "-"
        print(f"| {r['id']} | {r['recall']*100:.0f}% ({len(r['evidence']) - len(r['missing'])}/{len(r['evidence'])}) "
              f"| {miss} | {r['chunks']} | {r['unmatched_chunks']} | {r['summaries']} |")
    if res["mean_recall"] is None:
        print("\nNessuna domanda valutata.")
    else:
        print(f"\nRecall media: {res['mean_recall']*100:.1f}% su {res['evaluated']} domande; "
              f"recall 1.0 in {res['full_recall']}/{res['evaluated']}; "
              f"almeno un capitolo in {res['any_hit']}/{res['evaluated']}.")
    print(f"Chunk non trovati nel libro: {res['unmatched_chunks']}/{res['total_chunks']}.")
    print(f"Chunk-riassunto nel contesto (esclusi dal recall): {res['total_summaries']}.")
    if res["gold_without_dump"]:
        print(f"ATTENZIONE: {len(res['gold_without_dump'])} domande gold senza riga nel dump, ESCLUSE dalla media: "
              + ", ".join(res["gold_without_dump"]))
    if res["gold_without_evidence"]:
        print(f"Domande senza evidence_chapters, ESCLUSE dalla media: " + ", ".join(res["gold_without_evidence"]))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"JSON scritto in {a.out}")


if __name__ == "__main__":
    main()
