# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Sonda offline: Jev puo' filtrare i candidati entita' di GLiNER?

Uso:
    export TYPESAFE_API_KEY=...   # mai stampata ne' salvata
    uv run tests/e2e/scripts/jev_entity_probe.py run [dev|heldout]     # chiama Jev, salva gli score grezzi
    uv run tests/e2e/scripts/jev_entity_probe.py raw <workspace>  # log di produzione (g6) sul gold
    uv run tests/e2e/scripts/jev_entity_probe.py sweep [dev|heldout]   # solo offline, sugli score salvati

Per ogni chunk del gold (tests/e2e/data/it/jev_entities_gold.json) prende testo e candidati dal
jev_raw.jsonl di g5 e manda UNA richiesta con, per ogni candidato i:
  t{i}  choice PERSON / LOCATION / EVENT / OTHER / NOT_ENTITY  (variante A: NOT_ENTITY tra i tipi)
  e{i}  noul "e' un'entita' specifica della storia?"            (variante B: companion noul)
Salva tutto in tests/e2e/results/jev/entity_probe.json (score grezzi, usage, latenza).
Lo sweep calcola, per soglia, quanti GENERIC/JUNK vengono scartati e quante entita' vere si perdono;
ALIAS e' riportato a parte (con e senza contarlo come da scartare). Nessuna soglia fissata a priori.
Costo: stima 0.05 $/M token di input (fonti terze, nessun listino ufficiale).
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# GOLD=<path> per un altro set (es. jev_entities_heldout2.json, regola v3 con GROUP).
GOLD = Path(os.environ.get("GOLD", ROOT / "tests/e2e/data/it/jev_entities_gold.json"))
RAW = ROOT / "tests/e2e/results/workspaces/it_pinocchio_jev/jev_raw.jsonl"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "dev"
# PROMPT=v1 (prima versione, chiedeva nome proprio) o v2 (default: regola del gold v2, luoghi specifici
# anche senza nome proprio). v1 resta per riprodurre i file entity_probe.{split}.json.
PROMPT = os.environ.get("PROMPT", "v2")
OUT = ROOT / f"tests/e2e/results/jev/entity_probe.{SPLIT}{'' if PROMPT == 'v1' else '.' + PROMPT}.json"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
USD_PER_M_INPUT = 0.05  # stima
PREFIX = "doc_2ce8a648ce099cc2_"

TYPES = {
    "PERSON": "a real or fictional individual",
    "LOCATION": "geographic or physical place",
    "EVENT": "something that happens at a time: battle, meeting, ceremony",
    "OTHER": "nessuno dei tipi precedenti si adatta",
    "NOT_ENTITY": (
        "non e' un'entita' specifica della storia: nome comune generico (mare, scuola, stanza), "
        "parte del corpo, oggetto qualunque, verbo o frammento di parola"
    ),
}
NOUL = {
    "v1": (
        "«{n}», cosi' come compare in questo testo, e' un'entita' specifica e identificabile della storia "
        "(un personaggio, un luogo o un evento preciso, con nome proprio o nome che lo designa in modo univoco)? "
        "Rispondi no se e' un nome comune generico, una parte del corpo, un verbo o un frammento di parola."
    ),
    "v2": (
        "In questo testo, «{n}» indica un referente specifico e identificabile della storia: un personaggio, "
        "un evento preciso, oppure un luogo preciso e riconoscibile (anche senza nome proprio, per esempio la "
        "bottega o la casa di un certo personaggio)? Rispondi no se la parola e' usata in modo generico, senza "
        "un referente univoco (il mare, la scuola, il bosco in generale), oppure se e' una parte del corpo, "
        "un oggetto qualunque, un verbo o un frammento di parola."
    ),
}
KEEP = {"PERSON", "LOCATION", "EVENT", "GROUP"}  # GROUP solo nel gold v3 (heldout2)
DROP = {"GENERIC", "JUNK"}


def raw_records():
    recs = {}
    with RAW.open() as f:
        for line in f:
            r = json.loads(line)
            recs[r["chunk_id"]] = r
    return recs


def request(text, sentences, names):
    qs = {}
    for i, n in enumerate(names):
        qs[f"t{i}"] = {
            "type": "choice",
            "instructions": f"Qual e' il tipo dell'entita' «{n}» cosi' come compare in questo testo?",
            "criteria": TYPES,
        }
        qs[f"e{i}"] = {"type": "noul", "instructions": NOUL[PROMPT].format(n=n)}
    s = {f"s{i + 1}": x for i, x in enumerate(sentences[:60])}
    return {"model": MODEL, "state": {"text": text, "sentences": s}, "questions": qs}


def post(body, key):
    data = json.dumps(body).encode()
    for attempt in range(5):
        if attempt:
            time.sleep(2**attempt)
        req = urllib.request.Request(
            URL,
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "graphrag-probe/0.1",
            },
        )
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
    gold = [g for g in json.loads(GOLD.read_text())["chunks"] if g["split"] == SPLIT]
    recs = raw_records()
    out = []
    tot_in = tot_out = 0
    for g in gold:
        rec = recs[f"{PREFIX}{g['chunk']}"]
        names = [e["name"] for e in rec["entities"]] + [e["name"] for e in rec["dropped_entities"]]
        missing = set(g["labels"]) - set(names)
        if missing:
            raise SystemExit(f"chunk {g['chunk']}: etichette senza candidato {missing}")
        t0 = time.time()
        resp = post(request(rec["text"], rec["sentences"], names), key)
        ms = int((time.time() - t0) * 1000)
        res = resp.get("results") or resp.get("answers") or resp
        usage = resp.get("usage", {})
        tot_in += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        tot_out += usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        out.append({"chunk": g["chunk"], "names": names, "latency_ms": ms, "usage": usage, "response": resp})
        print(f"chunk {g['chunk']}: {len(names)} candidati, {ms} ms, usage {usage}", file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"input {tot_in} output {tot_out} token, costo stimato {tot_in * USD_PER_M_INPUT / 1e6:.4f} $")


def answers(resp):
    """Mappa id domanda -> risposta; accetta le forme note della risposta Jev."""
    for k in ("answers", "results", "questions"):
        if isinstance(resp.get(k), dict):
            return resp[k]
    return resp


def rows():
    gold = {g["chunk"]: g["labels"] for g in json.loads(GOLD.read_text())["chunks"]}
    for c in json.loads(OUT.read_text()):
        a = answers(c["response"])
        for i, n in enumerate(c["names"]):
            lab = gold[c["chunk"]].get(n)
            if lab is None or lab == "AMBIG":
                continue
            t = a[f"t{i}"]
            e = a[f"e{i}"]
            probs = t.get("probabilities", {})
            noul = e["noul"]
            yield {"chunk": c["chunk"], "name": n, "gold": lab, "choice": t.get("choice"),
                   "p_not": probs.get("NOT_ENTITY", 0.0), "noul": float(noul), "probs": probs}


def sweep():
    rs = list(rows())
    by = {}
    for r in rs:
        by.setdefault(r["gold"], 0)
        by[r["gold"]] += 1
    print("candidati per etichetta:", by)
    print("\nper candidato (gold, choice, p_not, noul):")
    for r in rs:
        print(f"  {r['chunk']:>3} {r['name']:<22} {r['gold']:<8} {r['choice']:<10} {r['p_not']:.2f} {r['noul']:.2f}")
    # tipo corretto sulle entita' tenute
    keep = [r for r in rs if r["gold"] in KEEP]
    ok = sum(1 for r in keep if r["choice"] == r["gold"])
    print(f"\ntipo esatto sulle entita' vere: {ok}/{len(keep)}")

    def report(name, drop_fn):
        print(f"\n{name}")
        print("| soglia | vere scartate | GENERIC+JUNK scartati | ALIAS scartati | precisione (alias=rumore) | precisione (alias=ok) |")
        print("|---|---|---|---|---|---|")
        for th in [x / 10 for x in range(1, 10)]:
            d = [r for r in rs if drop_fn(r, th)]
            kept = [r for r in rs if not drop_fn(r, th)]
            lost = sum(1 for r in d if r["gold"] in KEEP)
            junk = sum(1 for r in d if r["gold"] in DROP)
            ali = sum(1 for r in d if r["gold"] == "ALIAS")
            n_keep = sum(1 for r in rs if r["gold"] in KEEP)
            n_drop = sum(1 for r in rs if r["gold"] in DROP)
            n_ali = by.get("ALIAS", 0)
            good = sum(1 for r in kept if r["gold"] in KEEP)
            p1 = good / len(kept) if kept else 0
            p2 = (good + sum(1 for r in kept if r["gold"] == "ALIAS")) / len(kept) if kept else 0
            print(f"| {th:.1f} | {lost}/{n_keep} | {junk}/{n_drop} | {ali}/{n_ali} | {p1:.0%} | {p2:.0%} |")

    report("A) scarta se p(NOT_ENTITY) >= soglia", lambda r, th: r["p_not"] >= th)
    report("B) scarta se noul < soglia", lambda r, th: r["noul"] < th)
    report("A|B) scarta se p(NOT_ENTITY) >= soglia oppure noul < 1-soglia",
           lambda r, th: r["p_not"] >= th or r["noul"] < 1 - th)
    base = [r for r in rs]
    good = sum(1 for r in base if r["gold"] in KEEP)
    print(f"\nsenza filtro: precisione {good / len(base):.0%} (alias=rumore)")


def cats():
    """Filtro di produzione (scarta se noul < soglia) a 0.5/0.6/0.7: FN per categoria, P/R, sopravvissuti."""
    rs = list(rows())
    out = json.loads(OUT.read_text())
    tok = sum(c["usage"].get("input_tokens", 0) or c["usage"].get("prompt_tokens", 0) for c in out)
    n_all = sum(len(c["names"]) for c in out)
    print(f"{len(out)} chunk, {n_all} candidati ({len(rs)} non AMBIG), {tok} token input, "
          f"costo {tok * USD_PER_M_INPUT / 1e6:.4f} $")
    cat = sorted(KEEP)
    tot = {k: sum(1 for r in rs if r["gold"] == k) for k in cat + ["ALIAS", "GENERIC", "JUNK"]}
    print("candidati per etichetta:", tot)
    print("| soglia | " + " | ".join(f"FN {k}" for k in cat)
          + " | recall | precisione (alias=rumore) | precisione (alias=ok) | ALIAS tenuti | sopravvissuti | costo $ |")
    print("|---" * (len(cat) + 7) + "|")
    for th in (0.5, 0.6, 0.7):
        kept = [r for r in rs if r["noul"] >= th]
        fn = {k: sum(1 for r in rs if r["gold"] == k and r["noul"] < th) for k in cat}
        good = sum(1 for r in kept if r["gold"] in KEEP)
        ali = sum(1 for r in kept if r["gold"] == "ALIAS")
        n_true = sum(tot[k] for k in cat)
        surv = sum(1 for c in out for i, _ in enumerate(c["names"])
                   if float(answers(c["response"])[f"e{i}"]["noul"]) >= th)
        print(f"| {th:.1f} | " + " | ".join(f"{fn[k]}/{tot[k]}" for k in cat)
              + f" | {good / n_true:.0%} | {good / len(kept):.0%} | {(good + ali) / len(kept):.0%}"
              + f" | {ali}/{tot['ALIAS']} | {surv}/{n_all} | {tok * USD_PER_M_INPUT / 1e6:.4f} |")
    print("\nentita' vere perse a 0.7 (noul):")
    for r in sorted(rs, key=lambda r: r["noul"]):
        if r["gold"] in KEEP and r["noul"] < 0.7:
            print(f"  {r['chunk']:>3} {r['name']!r:<28} {r['gold']:<8} {r['noul']:.2f}")


def raw():
    """Valuta il log di produzione (jev_raw.jsonl con campi exists/kept, filter_entities=true) sul gold v2.

    uv run tests/e2e/scripts/jev_entity_probe.py raw <workspace|path/jev_raw.jsonl>
    """
    arg = sys.argv[2]
    path = Path(arg) if arg.endswith(".jsonl") else ROOT / f"tests/e2e/results/workspaces/{arg}/jev_raw.jsonl"
    gold = {f"{PREFIX}{g['chunk']}": g for g in json.loads(GOLD.read_text())["chunks"]}
    cnt = {}
    for line in path.open():
        r = json.loads(line)
        g = gold.get(r["chunk_id"])
        if not g:
            continue
        seen = set()
        for e in r["entities"]:
            lab = g["labels"].get(e["name"])
            seen.add(e["name"])
            if lab in (None, "AMBIG"):
                continue
            grp = "vere" if lab in KEEP else ("rumore" if lab in DROP else "epiteti")
            k = cnt.setdefault(grp, [0, 0])
            k[0] += 1
            k[1] += 0 if e.get("kept", True) else 1
            if grp == "vere" and not e.get("kept", True):
                print(f"  persa: chunk {g['chunk']} {e['name']} exists={e.get('exists')}")
            if grp == "rumore" and e.get("kept", True):
                print(f"  rumore tenuto: chunk {g['chunk']} {e['name']} exists={e.get('exists')}")
        miss = set(g["labels"]) - seen
        if miss:
            print(f"  chunk {g['chunk']}: candidati del gold assenti nel log {sorted(miss)}")
    for grp, (n, d) in cnt.items():
        print(f"{grp}: {n} candidati, {d} scartati")


if __name__ == "__main__":
    {"run": run, "sweep": sweep, "raw": raw, "cats": cats}[sys.argv[1] if len(sys.argv) > 1 else "sweep"]()
