# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Sonda di stabilita' di Jev Request A (classificazione relazione + link per coppia ordinata).

Uso:
    export TYPESAFE_API_KEY=...   # mai stampata ne' salvata
    uv run tests/e2e/scripts/jev_stability_probe.py run       # chiama Jev (budget massimo BUDGET_USD)
    uv run tests/e2e/scripts/jev_stability_probe.py report    # solo offline, sul file salvato
    uv run tests/e2e/scripts/jev_stability_probe.py filter    # offline: perdite g5 -> g6 dovute al filtro E

Per ogni chunk con coppie bersaglio manda 7 Request A con lo stesso testo, le stesse frasi, lo stesso schema
e lo stesso prompt di produzione (graphrag-core/src/entity/jev.rs, request_a in modalita' filter_entities:
niente domande t{i}). Cambia solo la composizione del batch, cioe' la lista dei nomi (e quindi le altre
coppie nella stessa richiesta):
  C0  composizione di g5 (lista nomi del jev_raw di g5)        x3: due ripetizioni identiche
  C1  composizione di g6 (nomi tenuti da E), con le entita' bersaglio rimesse se il filtro le aveva tolte
  C2  solo le entita' delle coppie bersaglio
  C3  composizione di g5 in ordine inverso
  C4  entita' bersaglio + meta' delle altre di g5 (sottoinsieme fisso, seed 20261111)
Registra le probabilita' di tutte le opzioni (NONE compreso) e il noul di link. Output in
tests/e2e/results/jev/stability_probe.json.
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WS = ROOT / "tests/e2e/results/workspaces"
OUT = ROOT / "tests/e2e/results/jev/stability_probe.json"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
USD_PER_M_INPUT = 0.05  # stima, nessun listino ufficiale
BUDGET_USD = 0.05
MAX_SENTENCES = 200
NONE = "NONE"
NONE_DESC = "il testo non afferma esplicitamente nessuna relazione del soggetto verso l'oggetto"
CANDIDATE_FLOOR = 0.2  # g6: B parte solo se scelta != NONE e link >= candidate_floor

# Ontologia generica 1.0.0 (graphrag-core/src/ontology.rs), stesso ordine: l'ordine dei criteri conta.
RELATIONS = {
    "FAMILY_OF": "kinship or marriage",
    "FRIEND_OF": "friendship, acquaintance or alliance between people or groups",
    "LOVES": "romantic love or strong affection",
    "OPPOSES": "conflict, rivalry, hostility or disagreement",
    "WORKS_FOR": "employed by or serving an organization or person",
    "LEADS": "heads, directs, manages or rules an organization, group or place",
    "MEMBER_OF": "belongs to an organization or group",
    "FOUNDED": "established an organization or place",
    "CREATED": "authored, wrote, made or invented a work or object",
    "LOCATED_IN": "is situated in, lives in, is based in or was born in a place",
    "PART_OF": "is a component, subdivision or subset of",
    "PARTICIPATED_IN": "took part in or attended an event",
    "OCCURRED_AT": "an event happened at a place or time",
    "CAUSES": "causes, leads to or results in",
    "INFLUENCES": "influences, inspires, teaches or mentors",
    "DISCUSSES": "talks about, describes, praises or refers to a topic, work or entity",
    "OWNS": "owns or possesses",
    NONE: NONE_DESC,
}

# Coppie bersaglio (chunk, soggetto, oggetto, etichetta attesa, categoria). Etichette: jev_pairs_gold.json
# dove presenti; "?" = coppia di un personaggio collettivo accettata in g5, senza gold manuale.
TARGETS = [
    (68, "Volpe", "Gatto", "FRIEND_OF", "forte"),
    (68, "Gatto", "Volpe", "FRIEND_OF", "borderline"),
    (68, "Volpe", "Pinocchio", NONE, "borderline"),
    (68, "Gatto", "Pinocchio", NONE, "none"),
    (68, "Pinocchio", "Volpe", NONE, "none"),
    (53, "Arlecchino", "Pulcinella", "OPPOSES", "forte"),
    (53, "Pinocchio", "Mangiafoco", NONE, "none"),
    (213, "Pinocchio", "Alidoro", "OPPOSES", "borderline"),
    (342, "Pinocchio", "Giangio", "WORKS_FOR", "forte-mancata"),
    (342, "Giangio", "Pinocchio", NONE, "none"),
    (152, "faine", "Melampo", "FRIEND_OF", "collettivo"),
    (152, "Melampo", "faine", "?", "collettivo"),
    (152, "Pinocchio", "Melampo", NONE, "none"),
    (156, "Pinocchio", "faine", "?", "collettivo"),
    (207, "carabinieri", "Pinocchio", "?", "collettivo"),
    (210, "Pinocchio", "carabinieri", "?", "collettivo"),
    (220, "Pinocchio", "pescatore", "?", "collettivo-borderline"),
    (224, "pescatore", "Pinocchio", "?", "borderline"),
]
COMPOSITIONS = ["C0", "C0r1", "C0r2", "C1", "C2", "C3", "C4"]


def load(ws):
    d = {}
    with (WS / ws / "jev_raw.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            d[int(r["chunk_id"].rsplit("_", 1)[1])] = r
    return d


def compositions(c, g5, g6):
    core = []
    for t in TARGETS:
        if t[0] == c:
            for n in (t[1], t[2]):
                if n not in core:
                    core.append(n)
    base = [e["name"] for e in g5[c]["entities"]]
    assert all(n in base for n in core), (c, core, base)
    kept6 = [e["name"] for e in g6[c]["entities"] if e.get("kept")][:8]
    c1 = kept6 + [n for n in core if n not in kept6]
    others = [n for n in base if n not in core]
    half = sorted(random.Random(20261111 + c).sample(others, len(others) // 2), key=base.index)
    c4 = [n for n in base if n in core or n in half]
    return {"C0": base, "C1": c1, "C2": [n for n in base if n in core], "C3": base[::-1], "C4": c4}


def request_a(text, sentences, names):
    qs = {}
    for i, x in enumerate(names):
        for j, y in enumerate(names):
            if i == j:
                continue
            qs[f"r{i}_{j}"] = {
                "type": "choice",
                "instructions": (
                    f"Nel testo, quale relazione ESPLICITA ha «{x}» (soggetto) verso «{y}» (oggetto)? "
                    "Conta solo ciò che il testo afferma direttamente, non ciò che si può dedurre dalla trama "
                    f"o dalla conoscenza esterna. Se il testo non afferma nessuna relazione di «{x}» verso «{y}», scegli NONE."
                ),
                "criteria": RELATIONS,
            }
            qs[f"n{i}_{j}"] = {
                "type": "noul",
                "instructions": f"Il testo afferma esplicitamente un'interazione diretta o un legame tra «{x}» e «{y}».",
            }
    s = {f"s{i + 1}": x for i, x in enumerate(sentences[:MAX_SENTENCES])}
    return {"model": MODEL, "state": {"text": text, "sentences": s}, "questions": qs}


def post(body, key):
    data = json.dumps(body, ensure_ascii=False).encode()
    for attempt in range(5):
        if attempt:
            time.sleep(2**attempt)
        req = urllib.request.Request(URL, data=data, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "User-Agent": "graphrag-probe/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise SystemExit(f"HTTP {e.code}: {e.read()[:300]!r}")
    raise SystemExit("gave up after 5 attempts")


def run():
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY non impostata")
    g5, g6 = load("it_pinocchio_jev"), load("it_pinocchio_jev_g6")
    chunks = sorted({t[0] for t in TARGETS})
    plan = []
    for c in chunks:
        comps = compositions(c, g5, g6)
        for k in COMPOSITIONS:
            plan.append((c, k, comps[k[:2]]))
    # Stima prima di spendere: ~4 caratteri per token sul corpo JSON.
    est = sum(len(json.dumps(request_a(g5[c]["text"], g5[c]["sentences"], n), ensure_ascii=False)) / 4
              for c, _, n in plan)
    print(f"{len(plan)} richieste, stima ~{est:.0f} token input, ~{est * USD_PER_M_INPUT / 1e6:.4f} $", file=sys.stderr)
    if est * USD_PER_M_INPUT / 1e6 > BUDGET_USD:
        raise SystemExit("stima oltre il budget")
    out, tok = [], 0
    for c, k, names in plan:
        rec = g5[c]
        t0 = time.time()
        resp = post(request_a(rec["text"], rec["sentences"], names), key)
        u = resp.get("usage", {})
        tok += u.get("input_tokens", 0)
        a = resp["answers"]
        pairs = {}
        for t in TARGETS:
            if t[0] != c:
                continue
            i, j = names.index(t[1]), names.index(t[2])
            r = a[f"r{i}_{j}"]
            pairs[f"{t[1]}->{t[2]}"] = {"choice": r["choice"], "probs": r.get("probabilities", {}),
                                        "confidence": r.get("confidence"), "link": a[f"n{i}_{j}"]["noul"]}
        out.append({"chunk": c, "comp": k, "names": names, "n_questions": len(names) * (len(names) - 1) * 2,
                    "usage": u, "latency_ms": int((time.time() - t0) * 1000), "pairs": pairs, "answers": a})
        print(f"chunk {c} {k}: {len(names)} nomi, usage {u}", file=sys.stderr)
        if tok * USD_PER_M_INPUT / 1e6 > BUDGET_USD * 0.95:
            print("budget quasi esaurito, stop", file=sys.stderr)
            break
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"input {tok} token, costo stimato {tok * USD_PER_M_INPUT / 1e6:.4f} $")


def report():
    runs = json.loads(OUT.read_text())
    tok = sum(r["usage"].get("input_tokens", 0) for r in runs)
    print(f"{len(runs)} richieste, {tok} token input, costo stimato {tok * USD_PER_M_INPUT / 1e6:.4f} $\n")
    print("| chunk | coppia | gold | cat | C0 | C0r1 | C0r2 | C1 | C2 | C3 | C4 | p(gold/top) min–max | p(NONE) min–max | link min–max | flip ripetizioni | flip composizioni |")
    print("|---" * 16 + "|")
    agg = {"rep_flip": 0, "comp_flip": 0, "cand_rep": 0, "cand_comp": 0, "n": 0,
           "dn_rep": [], "dn_comp": [], "dl_rep": [], "dl_comp": []}
    for t in TARGETS:
        c, key = t[0], f"{t[1]}->{t[2]}"
        byc = {r["comp"]: r["pairs"][key] for r in runs if r["chunk"] == c and key in r["pairs"]}
        if len(byc) < len(COMPOSITIONS):
            print(f"| {c} | {key} | incompleta |")
            continue
        lab = t[3] if t[3] not in ("?", NONE) else byc["C0"]["choice"]
        pl = [byc[k]["probs"].get(lab, 0) for k in COMPOSITIONS]
        pn = [byc[k]["probs"].get(NONE, 0) for k in COMPOSITIONS]
        ln = [byc[k]["link"] for k in COMPOSITIONS]
        reps = ["C0", "C0r1", "C0r2"]
        comps = ["C0", "C1", "C2", "C3", "C4"]
        cand = {k: byc[k]["choice"] != NONE and byc[k]["link"] >= CANDIDATE_FLOOR for k in COMPOSITIONS}
        rf = len({byc[k]["choice"] for k in reps}) > 1
        cf = len({byc[k]["choice"] for k in comps}) > 1
        agg["n"] += 1
        agg["rep_flip"] += rf
        agg["comp_flip"] += cf
        agg["cand_rep"] += len({cand[k] for k in reps}) > 1
        agg["cand_comp"] += len({cand[k] for k in comps}) > 1
        r_pn = [byc[k]["probs"].get(NONE, 0) for k in reps]
        c_pn = [byc[k]["probs"].get(NONE, 0) for k in comps]
        agg["dn_rep"].append(max(r_pn) - min(r_pn))
        agg["dn_comp"].append(max(c_pn) - min(c_pn))
        r_l = [byc[k]["link"] for k in reps]
        c_l = [byc[k]["link"] for k in comps]
        agg["dl_rep"].append(max(r_l) - min(r_l))
        agg["dl_comp"].append(max(c_l) - min(c_l))
        cells = " | ".join(f"{byc[k]['choice'][:9]} {byc[k]['probs'].get(byc[k]['choice'], 0):.2f}" for k in COMPOSITIONS)
        print(f"| {c} | {key} | {t[3]} | {t[4]} | {cells} | {lab[:9]} {min(pl):.2f}–{max(pl):.2f} | "
              f"{min(pn):.2f}–{max(pn):.2f} | {min(ln):.2f}–{max(ln):.2f} | {'sì' if rf else 'no'} | {'sì' if cf else 'no'} |")
    n = agg["n"]

    def mean(x):
        return sum(x) / len(x) if x else 0

    print(f"\ncoppie complete: {n}")
    print(f"scelta cambiata tra 3 ripetizioni identiche: {agg['rep_flip']}/{n}")
    print(f"scelta cambiata tra 5 composizioni:          {agg['comp_flip']}/{n}")
    print(f"candidata a B (scelta != NONE e link >= {CANDIDATE_FLOOR}) cambiata: ripetizioni {agg['cand_rep']}/{n}, composizioni {agg['cand_comp']}/{n}")
    print(f"escursione media p(NONE): ripetizioni {mean(agg['dn_rep']):.3f}, composizioni {mean(agg['dn_comp']):.3f}; "
          f"max {max(agg['dn_rep']):.2f} / {max(agg['dn_comp']):.2f}")
    print(f"escursione media link:    ripetizioni {mean(agg['dl_rep']):.3f}, composizioni {mean(agg['dl_comp']):.3f}; "
          f"max {max(agg['dl_rep']):.2f} / {max(agg['dl_comp']):.2f}")


def filter_losses():
    """Coppie accettate in g5 e non in g6: causa filtro E (entita' scartata) o A/B (coppia presente)."""
    g5, g6 = load("it_pinocchio_jev"), load("it_pinocchio_jev_g6")
    cnt = {"filtro E": 0, "tetto": 0, "A: scelta cambiata": 0, "B/link: stessa scelta, rifiutata": 0}
    for c, r in g5.items():
        if c not in g6:
            continue
        p6 = {(q["head"], q["tail"]): q for q in g6[c]["pairs"]}
        kept = {e["name"]: e.get("kept") for e in g6[c]["entities"]}
        for q in r["pairs"]:
            if not q["accepted"]:
                continue
            q6 = p6.get((q["head"], q["tail"]))
            if q6 is None:
                if kept.get(q["head"]) is False or kept.get(q["tail"]) is False:
                    cnt["filtro E"] += 1
                else:
                    cnt["tetto"] += 1
            elif not q6["accepted"]:
                cnt["A: scelta cambiata" if q6["rel_choice"] != q["rel_choice"] else "B/link: stessa scelta, rifiutata"] += 1
    gain = sum(1 for c, r in g6.items() for q in r["pairs"] if q["accepted"]
               and not any(x["head"] == q["head"] and x["tail"] == q["tail"] and x["accepted"]
                           for x in g5.get(c, {"pairs": []})["pairs"]))
    print("relazioni accettate in g5 e perse in g6, per causa:", cnt)
    print("relazioni accettate in g6 e non in g5:", gain)


if __name__ == "__main__":
    {"run": run, "report": report, "filter": filter_losses}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
