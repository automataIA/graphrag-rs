# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Sweep offline delle soglie Jev (`gliner.jev`) sul log grezzo `jev_raw.jsonl`, senza richiamare l'API.

Uso:
    uv run tests/e2e/scripts/jev_sweep.py <workspace|path/jev_raw.jsonl> [--top 25] [--all] [--by-kind]

Per ogni combinazione (min_link, min_evidence, min_type_prob) la regola di accettazione e' quella del crate:
scelta != NONE, p(scelta) >= min_type_prob, link >= min_link, evidence >= min_evidence (evidence presente solo per
le coppie con link >= candidate_floor al momento della run). Metriche:
  - relazioni accettate;
  - pair / typed recall sulle triple gold (tests/e2e/data/it/gold_triples.json) con la stessa logica di graph_eval.py
    (alias a inizio parola, famiglie da relation_type + descrizione), su un grafo ricostruito dalle sole relazioni accettate;
  - precisione/recall/F1 sull'insieme di coppie etichettate tests/e2e/data/it/jev_pairs_gold.json
    (se manca, la parte viene saltata): un'accettazione e' corretta se il tipo coincide con l'etichetta;
    label NONE => qualsiasi accettazione e' un falso positivo. Una coppia gold e' abbinata ai record il cui testo
    contiene `snippet` e che contengono la coppia (nomi confrontati come in graph_eval).
"""
import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_eval as ge  # noqa: E402  (riuso norm/name_matches/families_of; i suoi risultati non cambiano)

ROOT = ge.ROOT
PAIRS_GOLD = ROOT / "tests/e2e/data/it/jev_pairs_gold.json"
GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TYPE_GRID = [0.0, 0.3, 0.5, 0.7, 0.9]


def load_records(arg: str) -> list[dict]:
    p = Path(arg)
    if not p.is_file():
        p = ge.WS_DIR / arg / "jev_raw.jsonl"
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{len(recs)} record da {p}", file=sys.stderr)
    return recs


def accepted(pair: dict, ml: float, me: float, mt: float) -> bool:
    c = pair.get("rel_choice")
    ev = pair.get("evidence")
    if not c or c == "NONE" or ev is None:
        return False
    return (pair["rel_probs"].get(c, 0.0) >= mt and pair["link"] >= ml and ev >= me)


def description(rec: dict, pair: dict) -> str:
    sp = pair.get("span_choice") or ""
    if sp.startswith("s") and sp[1:].isdigit():
        i = int(sp[1:]) - 1
        if 0 <= i < len(rec["sentences"]):
            return rec["sentences"][i]
    return ""


def gold_triples_eval(records: list[dict], ml: float, me: float, mt: float, triples: list[dict]) -> tuple[int, int, int]:
    """(n_relazioni, pair_hits, typed_hits) su un grafo ricostruito dalle relazioni accettate."""
    names: dict[str, str] = {}  # norm -> nome
    rels = []
    for rec in records:
        for e in rec.get("entities", []):
            names[ge.norm(e["name"])] = e["name"]
        for p in rec.get("pairs", []):
            if accepted(p, ml, me, mt):
                rels.append({"s": ge.norm(p["head"]), "t": ge.norm(p["tail"]), "relation_type": p["rel_choice"],
                             "description": description(rec, p)})
                names[ge.norm(p["head"])] = p["head"]
                names[ge.norm(p["tail"])] = p["tail"]
    adj: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rels:
        adj[(r["s"], r["t"])].append(r)

    def ids_for(label, aliases):
        al = [ge.norm(label)] + [ge.norm(a) for a in aliases]
        return {n for n in names if any(ge.name_matches(a, n) for a in al)}

    pair_hits = typed_hits = 0
    for t in triples:
        hs, ts = ids_for(t["head"], t.get("head_aliases", [])), ids_for(t["tail"], t.get("tail_aliases", []))
        conn = [r for h in hs for x in ts if h != x for r in adj.get((h, x), []) + adj.get((x, h), [])]
        pair_hits += bool(conn)
        typed_hits += any(t["family"] in ge.families_of(r) for r in conn)
    return len(rels), pair_hits, typed_hits


def match_gold_pairs(records: list[dict], gold: list[dict]) -> list[dict]:
    """Per ogni coppia gold: i (record, coppia) abbinati."""
    out = []
    for g in gold:
        sn = ge.norm(g["snippet"])
        hn, tn = ge.norm(g["head"]), ge.norm(g["tail"])
        hits = []
        for rec in records:
            if sn not in ge.norm(rec["text"]):
                continue
            for p in rec.get("pairs", []):
                if ge.name_matches(hn, ge.norm(p["head"])) and ge.name_matches(tn, ge.norm(p["tail"])):
                    hits.append((rec, p))
        out.append({"gold": g, "hits": hits})
    return out


def pairs_eval(matched: list[dict], ml: float, me: float, mt: float) -> dict:
    tp = fp = fn = 0
    kinds: dict[str, dict] = defaultdict(lambda: {"n": 0, "found": 0, "accepted": 0, "correct": 0, "fp": 0})
    for m in matched:
        g, hits = m["gold"], m["hits"]
        k = kinds[g.get("kind", "?")]
        k["n"] += 1
        k["found"] += bool(hits)
        acc_types = {p["rel_choice"] for _, p in hits if accepted(p, ml, me, mt)}
        k["accepted"] += bool(acc_types)
        if g["label"] == "NONE":
            if acc_types:
                fp += 1
                k["fp"] += 1
        else:
            if g["label"] in acc_types:
                tp += 1
                k["correct"] += 1
                if acc_types - {g["label"]}:
                    fp += 1
                    k["fp"] += 1
            else:
                fn += 1
                if acc_types:
                    fp += 1
                    k["fp"] += 1
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "p": pr, "r": rc, "f1": f1, "kinds": dict(kinds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="workspace (cerca <WS_DIR>/<ws>/jev_raw.jsonl) o percorso del jsonl")
    ap.add_argument("--top", type=int, default=25, help="righe da mostrare, ordinate per F1 (se c'e' il gold) o typed recall")
    ap.add_argument("--all", action="store_true", help="mostra tutte le combinazioni")
    ap.add_argument("--by-kind", action="store_true", help="dettaglio per kind della migliore combinazione (e di quella di default)")
    a = ap.parse_args()

    records = load_records(a.raw)
    triples = json.loads(ge.GOLD.read_text())["triples"]
    gold = None
    if PAIRS_GOLD.is_file():
        raw = json.loads(PAIRS_GOLD.read_text(encoding="utf-8"))
        gold = raw["pairs"] if isinstance(raw, dict) else raw
    else:
        print(f"NOTA: {PAIRS_GOLD} mancante: salto precisione/recall sulle coppie etichettate.", file=sys.stderr)
    matched = match_gold_pairs(records, gold) if gold else None
    if matched:
        nf = sum(bool(m["hits"]) for m in matched)
        print(f"coppie gold abbinate a un record: {nf}/{len(matched)}", file=sys.stderr)

    rows = []
    for ml, me, mt in itertools.product(GRID, GRID, TYPE_GRID):
        n, ph, th = gold_triples_eval(records, ml, me, mt, triples)
        row = {"ml": ml, "me": me, "mt": mt, "n": n, "pair": ph, "typed": th}
        if matched:
            row["pe"] = pairs_eval(matched, ml, me, mt)
        rows.append(row)
    rows.sort(key=lambda r: (-(r["pe"]["f1"]) if matched else -r["typed"], -r["typed"], r["n"]))
    shown = rows if a.all else rows[: a.top]

    nt = len(triples)
    pc = lambda x: f"{x*100:.0f}%"
    head = "| min_link | min_evidence | min_type_prob | relazioni | pair recall | typed recall |"
    sep = "|---|---|---|---|---|---|"
    if matched:
        head += " P | R | F1 | TP/FP/FN |"
        sep += "---|---|---|---|"
    print(head)
    print(sep)
    for r in shown:
        line = (f"| {r['ml']} | {r['me']} | {r['mt']} | {r['n']} | {pc(r['pair']/nt)} ({r['pair']}/{nt}) | "
                f"{pc(r['typed']/nt)} ({r['typed']}/{nt}) |")
        if matched:
            e = r["pe"]
            line += f" {pc(e['p'])} | {pc(e['r'])} | {pc(e['f1'])} | {e['tp']}/{e['fp']}/{e['fn']} |"
        print(line)

    if a.by_kind and matched:
        for label, r in (("migliore", rows[0]), ("default (0.5/0.5/0.0)", next(x for x in rows if (x["ml"], x["me"], x["mt"]) == (0.5, 0.5, 0.0)))):
            print(f"\nPer kind, {label}: min_link={r['ml']} min_evidence={r['me']} min_type_prob={r['mt']}\n")
            print("| kind | coppie | trovate nei record | accettate | corrette | falsi positivi |")
            print("|---|---|---|---|---|---|")
            for k, v in sorted(r["pe"]["kinds"].items()):
                print(f"| {k} | {v['n']} | {v['found']} | {v['accepted']} | {v['correct']} | {v['fp']} |")


if __name__ == "__main__":
    main()
