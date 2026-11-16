# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Giudice post-hoc delle risposte del bench con il modello Jev di TypeSafe (POST /v1/systemone).

Uso:
    export TYPESAFE_API_KEY=...   # mai stampata ne' salvata
    uv run tests/e2e/scripts/jev_judge.py tests/e2e/data/it/gold.json tests/e2e/results/A.json [B.json ...] [--limit N]

Per ogni risposta una richiesta con state = {domanda, risposta_di_riferimento (= notes del gold),
risposta_da_valutare} e due domande: noul `corretta` (probabilita' che la risposta sia coerente col riferimento)
e score `punteggio` (0 sbagliata, 1 parziale, 2 corretta e completa). Scrive
tests/e2e/results/jev/<basename>.json e stampa una riga di tabella markdown per run.
Costo: stima a 0.05 $/M token di input (output gratuito), da fonti terze, nessun listino ufficiale.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "tests/e2e/results/jev"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
USD_PER_M_INPUT = 0.05  # stima

QUESTIONS = {
    "corretta": {
        "type": "noul",
        "instructions": (
            "La risposta_da_valutare e' fattualmente coerente con la risposta_di_riferimento e risponde alla domanda? "
            "Rispondi no se contraddice il riferimento, scambia attributi o personaggi (per esempio assegna a un "
            "personaggio una caratteristica che il riferimento attribuisce a un altro), omette un elemento essenziale "
            "del riferimento o dice di non sapere. Dettagli aggiuntivi corretti o non contraddittori sono accettabili."
        ),
    },
    "punteggio": {
        "type": "score",
        "instructions": (
            "Valuta la risposta_da_valutare confrontandola con la risposta_di_riferimento per la domanda data. "
            "Gli attributi assegnati al personaggio sbagliato sono un errore."
        ),
        "criteria": [
            "Sbagliata: contraddice il riferimento, scambia attributi o personaggi, o non risponde.",
            "Parziale: manca un elemento chiave del riferimento o contiene un errore minore.",
            "Corretta e completa: contiene tutti gli elementi chiave del riferimento senza errori.",
        ],
    },
}


def call(key: str, state: dict) -> dict:
    body = json.dumps({"model": MODEL, "state": state, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(
        URL, data=body, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "graphrag-probe/0.1"})
    last = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:300]}") from None
            time.sleep(2)
    raise RuntimeError(f"richiesta fallita: {type(last).__name__}")


def judge_run(key: str, gold: dict, result_path: Path, limit: int | None) -> dict:
    res = json.loads(result_path.read_text())
    qa = {q["id"]: q for q in res["questions_and_answers"]}
    items, tin, tout = [], 0, 0
    for g in gold["questions"][:limit]:
        r = qa.get(g["id"])
        if r is None:
            print(f"  manca {g['id']} in {result_path.name}", file=sys.stderr)
            continue
        state = {"domanda": g["question"], "risposta_di_riferimento": g["notes"],
                 "risposta_da_valutare": r["answer"]}
        out = call(key, state)
        a = out["answers"]
        u = out.get("usage", {})
        tin += u.get("input_tokens", 0)
        tout += u.get("output_tokens", 0)
        items.append({"id": g["id"], "keyword_score": r["score"]["score"], "keyword_passed": r["score"]["passed"],
                      "jev_prob_correct": a["corretta"]["noul"], "jev_score": a["punteggio"]["score"],
                      "usage": u})
    n = len(items)
    mean = lambda k: sum(i[k] for i in items) / n if n else 0.0
    disagreements = [i["id"] for i in items if i["keyword_passed"] != (i["jev_prob_correct"] >= 0.5)]
    summary = {
        "n": n, "mean_keyword_score": mean("keyword_score"), "mean_jev_prob": mean("jev_prob_correct"),
        "jev_correct_at_0.5": sum(i["jev_prob_correct"] >= 0.5 for i in items),
        "disagreements": disagreements, "input_tokens": tin, "output_tokens": tout,
        "cost_estimate_usd": tin * USD_PER_M_INPUT / 1e6, "cost_note": "stima, 0.05 $/M input, output gratuito",
        "model": MODEL,
    }
    return {"run": result_path.name, "questions": items, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--limit", type=int, default=None, help="solo le prime N domande (prove)")
    args = ap.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("TYPESAFE_API_KEY non impostata")
    gold = json.loads(Path(args.gold).read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("| Run | punteggio keyword | prob. Jev media | corrette per Jev | disaccordi | token in | costo Jev stima |")
    print("|---|---|---|---|---|---|---|")
    for rp in map(Path, args.results):
        out = judge_run(key, gold, rp, args.limit)
        s = out["summary"]
        (OUT_DIR / rp.name).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"| {rp.stem} | {s['mean_keyword_score']:.3f} | {s['mean_jev_prob']:.3f} | "
              f"{s['jev_correct_at_0.5']}/{s['n']} | {len(s['disagreements'])} ({', '.join(s['disagreements'])}) | "
              f"{s['input_tokens']} | {s['cost_estimate_usd']:.5f} $ (stima) |", flush=True)


if __name__ == "__main__":
    main()
