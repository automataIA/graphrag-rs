# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Valutazione della qualita' del grafo (Pinocchio, italiano) di un workspace graphrag-rs.

Uso:
    uv run tests/e2e/scripts/graph_eval.py it_pinocchio_v3 it_pinocchio_cerebras [--n 40] [--seed 42] [--no-judge] [--lang it|en]

Metriche (tutte APPROSSIMATE, vedi tests/e2e/data/it/BASELINE.md):
  1. dimensione: entita' per tipo, relazioni per tipo;
  2. key-entity recall: quota dei personaggi principali presenti come entita' (alias, senza maiuscole/accenti,
     corrispondenza a inizio parola cosi' da tollerare le varianti morfologiche);
  3. pair recall: quota delle triple gold (tests/e2e/data/it/gold_triples.json) la cui coppia (head, tail)
     e' collegata da QUALSIASI relazione, in entrambe le direzioni;
  4. typed recall: come 3, ma la relazione deve appartenere alla famiglia della tripla. La famiglia di una
     relazione e' dedotta da relation_type + description con la tabella FAMILY_KEYWORDS (lenient, a sottostringa);
  5. precisione stimata: campione casuale (seed fisso) di N relazioni giudicato dal 9B locale
     (llama-server :8080) sul testo di fino a 2 chunk di contesto. Risposta attesa: {"supported": true|false}.
"""
import argparse
import json
import random
import re
import sys
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WS_DIR = ROOT / "tests/e2e/results/workspaces"
OUT_DIR = ROOT / "tests/e2e/results/graph_eval"
GOLD = ROOT / "tests/e2e/data/it/gold_triples.json"
LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
LLM_MODEL = "qwen3.5-9b"

# Personaggi principali: nome canonico -> alias (confronto normalizzato, match a inizio parola).
KEY_ENTITIES = {
    "Pinocchio": ["pinocchio"],
    "Geppetto": ["geppetto", "polendina"],
    "Maestro Ciliegia": ["ciliegia", "mastr antonio", "mastro antonio"],
    "Grillo-parlante": ["grillo"],
    "Mangiafoco": ["mangiafoco", "mangiafuoco"],
    "Arlecchino": ["arlecchino"],
    "Pulcinella": ["pulcinella"],
    "Gatto": ["gatto"],
    "Volpe": ["volpe"],
    "Fata": ["fata", "fatina", "capelli turchini", "turchina"],
    "Lucignolo": ["lucignolo"],
    "Omino di burro": ["omino"],
    "Pesce-cane": ["pesce cane", "pescecane"],
    "Tonno": ["tonno"],
    "Melampo": ["melampo"],
    "Colombo": ["colombo", "colomba"],
    "Giudice (gorilla)": ["giudice", "gorilla"],
    "Carabinieri": ["carabinier"],
    "Alidoro": ["alidoro"],
    "Lumaca": ["lumaca"],
    "Pescatore verde": ["pescatore"],
    "Medoro": ["medoro"],
    "Eugenio": ["eugenio"],
    "Delfino": ["delfino"],
}

# Versione inglese (traduzione di Carol Della Chiesa, Gutenberg #500), scelta con --lang en.
KEY_ENTITIES_EN = {
    "Pinocchio": ["pinocchio"],
    "Geppetto": ["geppetto", "polendina"],
    "Mastro Cherry": ["cherry", "mastro antonio"],
    "Talking Cricket": ["cricket"],
    "Fire Eater": ["fire eater"],
    "Harlequin": ["harlequin"],
    "Pulcinella": ["pulcinella"],
    "Cat": ["cat"],
    "Fox": ["fox"],
    "Fairy": ["fairy", "azure hair", "maiden"],
    "Lamp-Wick": ["lamp wick", "lampwick"],
    "Little Man": ["little man"],
    "Terrible Shark": ["shark"],
    "Tunny": ["tunny"],
    "Melampo": ["melampo"],
    "Pigeon": ["pigeon"],
    "Judge (Gorilla)": ["judge", "gorilla"],
    "Carabineers": ["carabineer"],
    "Alidoro": ["alidoro"],
    "Snail": ["snail"],
    "Green Fisherman": ["fisherman"],
    "Medoro": ["medoro", "poodle"],
    "Eugene": ["eugene"],
    "Dolphin": ["dolphin"],
}

# Famiglia -> sottostringhe (su relation_type + description normalizzati, "_" -> spazio). Lenient e grossolano.
FAMILY_KEYWORDS = {
    "kinship": ["family of", "created", "padre", "babbo", "papa", "figli", "mamma", "madre", "father", "mother", " son",
                "daughter", "parent", "creat", "carv", "scolp", "fratell", "sorell", "brother", "sister", "famigl",
                "genitor", "creator", "creatore"],
    "friendship": ["friend", "amic", "compagn", "companion", "ally", "alleat", "loves", " ama", "affett", "amore",
                   "amicizia", "colleague"],
    "hostility": ["opposes", "nemic", "enemy", "oppos", "attack", "attacc", "arrest", "imprig", "imprison", "jail",
                  "cattur", "captur", "catch", "uccid", "kill", "impicc", "hang", "fry", "friggere", "frigge",
                  "burn", "brucia", "minacc", "threat", "punish", "punisc", "chase", "insegu", "swallow", "ingoia",
                  "inghiott", "devour", "divor", "mangia", " eat", "hostile", "ostil", "conflict", "fight", "lotta",
                  "zuffa", "quarrel", "litig", "scaglia", "throw", "aggred", "assault", "rimprover", "scold", "vend",
                  "sell", "sold", "buys", "compra", "costring", "force", "rinchius", "rapisc", "kidnap", "persecu"],
    "deception": ["decei", "inganna", "ingann", "truff", "cheat", "trick", "raggir", "abbindol", "fool", "bugi", "lie to",
                  "lure", "attira", "tempt", "mislead", "scam", "defraud", "manipul", "convince", "convinc",
                  "persuad", "swindl", "betray", "tradi"],
    "help": ["helps", "aiut", "help", "assist", "salv", "rescue", "save", "soccor", "guid", "cur", "heal", "support",
             "protect", "advis", "consigli", "carry", "porta", "trasport", "serve", "servant", "employ", "works for",
             "regala", "give", "dona", "offre", "offer", "free", "liber", "rescu", "inform", "comand", "command",
             "nurse", "care"],
    "meeting": ["meet", "incontr", "encounter", "ritrov", "reunit", "visit", "talk", "parla", "speak", "convers",
                "conoc", "finds", "found", "trova", "interact", "replac"],
    "transformation": ["transform", "trasform", "diventa", "divent", "become", "turn", "convert", "metamorf",
                       "change", "cambia", "trasmut"],
    "location": ["located", "located in", "occurred at", "situated", "lives", "resid", "abita", "vive", "si trova",
                 "trovav", "rinchius", "imprig", "prigion", "imprison", "arriv", "va a ", "goes to", "go to", "travel",
                 "viaggi", "reached", "raggiung", "stay", "inside", "dentro", "part of", "lives in", "locat"],
}
# relation_type canonico -> famiglie (in aggiunta alle keyword sul testo)
TYPE_FAMILIES = {
    "FAMILY_OF": ["kinship"], "CREATED": ["kinship"], "FRIEND_OF": ["friendship"], "LOVES": ["friendship"],
    "OPPOSES": ["hostility"], "HELPS": ["help"], "WORKS_FOR": ["help"],
    "LOCATED_IN": ["location"], "OCCURRED_AT": ["location"], "PART_OF": ["location"],
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def name_matches(alias_n: str, name_n: str) -> bool:
    """alias a inizio parola dentro il nome (tollera articoli e varianti morfologiche: 'carabinier' in 'i carabinieri')."""
    return alias_n != "" and re.search(r"(?<![a-z0-9])" + re.escape(alias_n), name_n) is not None


def families_of(rel: dict) -> set[str]:
    text = " " + norm(rel.get("relation_type", "") + " " + (rel.get("description") or "")) + " "
    fams = set(TYPE_FAMILIES.get(rel.get("relation_type", ""), []))
    for fam, kws in FAMILY_KEYWORDS.items():
        if any(k in text for k in kws):
            fams.add(fam)
    return fams


def load_ws(name: str) -> dict:
    return json.loads((WS_DIR / name / "graph.json").read_text())


def ask_judge(prompt: str) -> tuple[bool | None, str]:
    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "max_tokens": 30,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": "Sei un verificatore rigoroso. Rispondi SOLO con JSON: {\"supported\": true} oppure {\"supported\": false}."},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(LLM_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    m = re.search(r'"supported"\s*:\s*(true|false)', out)
    return (m.group(1) == "true" if m else None), out


def judge_sample(data: dict, n: int, seed: int, ent_name: dict, chunk_text: dict) -> tuple[list, dict]:
    rels = data["relationships"]
    rng = random.Random(seed)
    sample = rng.sample(rels, min(n, len(rels)))
    out = []
    for r in sample:
        ctx = [chunk_text[c] for c in (r.get("context_chunks") or []) if c in chunk_text][:2]
        head, tail = ent_name.get(r["source_id"], r["source_id"]), ent_name.get(r["target_id"], r["target_id"])
        rel_line = f"Relazione: {head} --[{r['relation_type']}]--> {tail}. Descrizione: {r.get('description') or '(nessuna)'}"
        if ctx:
            ctx_txt = "\n\n".join(f"[Testo {i+1}]\n{c[:1800]}" for i, c in enumerate(ctx))
            prompt = (f"{ctx_txt}\n\n{rel_line}\n\nLa relazione e' supportata dal testo sopra? Rispondi true SOLO se il testo afferma o implica chiaramente "
                      f"che il legame esiste tra QUESTE due entita' (veri personaggi, luoghi, oggetti o eventi della storia, non frammenti di frase), "
                      f"che il TIPO di relazione e' appropriato e che la DIREZIONE testa->coda e' corretta "
                      f"(es. PART_OF: la testa e' parte della coda; CREATED: la testa ha creato la coda). Altrimenti false. "
                      f'Rispondi solo {{"supported": true}} o {{"supported": false}}.')
        else:
            prompt = (f"{rel_line}\n\nNon e' disponibile il testo. Basandoti sulla trama di 'Le avventure di Pinocchio' di Collodi, "
                      f"questa relazione e' plausibile e corretta? Rispondi solo {{\"supported\": true}} o {{\"supported\": false}}.")
        verdict, raw = ask_judge(prompt)
        out.append({"head": head, "relation_type": r["relation_type"], "tail": tail, "description": r.get("description"),
                    "confidence": r.get("confidence"), "context_resolved": bool(ctx),
                    "context_snippets": [c[:600] for c in ctx], "supported": verdict, "raw": raw})
        print(f"  judge {len(out)}/{len(sample)}: {verdict}", file=sys.stderr)
    return out, {}


def evaluate(name: str, n: int, seed: int, judge: bool) -> dict:
    data = load_ws(name)
    ents, rels = data["entities"], data["relationships"]
    ent_name = {e["id"]: e["name"] for e in ents}
    ent_norm = {e["id"]: norm(e["name"]) for e in ents}
    chunk_text = {c["id"]: c["content"] for c in data["chunks"]}

    res: dict = {"workspace": name}
    res["size"] = {"entities": len(ents), "relationships": len(rels), "chunks": len(data["chunks"]),
                   "entities_by_type": dict(Counter(e["type"] for e in ents).most_common()),
                   "relationships_by_type": dict(Counter(r["relation_type"] for r in rels).most_common())}

    # 2. key-entity recall
    found, missing = {}, []
    for canon, aliases in KEY_ENTITIES.items():
        al = [norm(a) for a in aliases]
        hit = [ent_name[i] for i, nn in ent_norm.items() if any(name_matches(a, nn) for a in al)]
        (found.__setitem__(canon, hit[:5]) if hit else missing.append(canon))
    res["key_entity_recall"] = {"recall": round(len(found) / len(KEY_ENTITIES), 3), "found": len(found),
                                "total": len(KEY_ENTITIES), "missing": missing}

    # 3-4. pair / typed recall
    triples = json.loads(GOLD.read_text())["triples"]
    adj: dict[tuple[str, str], list[dict]] = {}
    for r in rels:
        adj.setdefault((r["source_id"], r["target_id"]), []).append(r)

    def ids_for(label, aliases):
        al = [norm(label)] + [norm(a) for a in aliases]
        return {i for i, nn in ent_norm.items() if any(name_matches(a, nn) for a in al)}

    per, pair_hits, typed_hits = [], 0, 0
    fam_tot, fam_pair, fam_typed = Counter(), Counter(), Counter()
    for t in triples:
        hs, ts = ids_for(t["head"], t.get("head_aliases", [])), ids_for(t["tail"], t.get("tail_aliases", []))
        connecting = [r for h in hs for tt in ts for r in adj.get((h, tt), []) + adj.get((tt, h), []) if h != tt]
        typed = [r for r in connecting if t["family"] in families_of(r)]
        pair_hits += bool(connecting)
        typed_hits += bool(typed)
        fam_tot[t["family"]] += 1
        fam_pair[t["family"]] += bool(connecting)
        fam_typed[t["family"]] += bool(typed)
        per.append({"head": t["head"], "rel": t["rel"], "tail": t["tail"], "family": t["family"],
                    "head_entities": len(hs), "tail_entities": len(ts), "n_connecting": len(connecting),
                    "pair": bool(connecting), "typed": bool(typed),
                    "example": (f"{connecting[0]['relation_type']}: {connecting[0].get('description') or ''}"[:160] if connecting else None)})
    nt = len(triples)
    # controllo: coppie (head_i, tail_(i+7)) mescolate. Misura quanto il pair recall sia "gratis" per hub/alias larghi.
    ctrl = 0
    for i, t in enumerate(triples):
        u = triples[(i + 7) % nt]
        hs, ts = ids_for(t["head"], t.get("head_aliases", [])), ids_for(u["tail"], u.get("tail_aliases", []))
        ctrl += any(adj.get((h, x)) or adj.get((x, h)) for h in hs for x in ts if h != x)
    res["pair_recall_control_shuffled"] = {"rate": round(ctrl / nt, 3), "hits": ctrl, "total": nt}
    # sottoinsieme senza Pinocchio (il grafo e' hub-centrico: la coppia con Pinocchio e' quasi sempre connessa)
    idx = [i for i, t in enumerate(triples) if "pinocchio" not in (norm(t["head"]), norm(t["tail"]))]
    np_ = len(idx)
    c2 = 0
    for k, i in enumerate(idx):
        u = triples[idx[(k + 3) % np_]]
        t = triples[i]
        hs, ts = ids_for(t["head"], t.get("head_aliases", [])), ids_for(u["tail"], u.get("tail_aliases", []))
        c2 += any(adj.get((h, x)) or adj.get((x, h)) for h in hs for x in ts if h != x)
    res["no_pinocchio_subset"] = {"total": np_, "pair": sum(per[i]["pair"] for i in idx),
                                  "typed": sum(per[i]["typed"] for i in idx), "pair_control_shuffled": c2}
    res["pair_recall"] = {"recall": round(pair_hits / nt, 3), "hits": pair_hits, "total": nt}
    res["typed_recall"] = {"recall": round(typed_hits / nt, 3), "hits": typed_hits, "total": nt,
                           "note": "approssimato: famiglia dedotta da keyword su relation_type + description"}
    res["by_family"] = {f: {"total": fam_tot[f], "pair": fam_pair[f], "typed": fam_typed[f]} for f in fam_tot}
    res["triples"] = per

    # 5. precisione stimata
    resolved = sum(1 for r in rels if any(c in chunk_text for c in (r.get("context_chunks") or [])))
    res["context_resolvable_fraction"] = round(resolved / max(1, len(rels)), 3)
    if judge:
        sample, _ = judge_sample(data, n, seed, ent_name, chunk_text)
        valid = [s for s in sample if s["supported"] is not None]
        sup = sum(1 for s in valid if s["supported"])
        res["precision"] = {"n": len(sample), "judged": len(valid), "supported": sup,
                            "precision": round(sup / len(valid), 3) if valid else None,
                            "with_context": sum(1 for s in sample if s["context_resolved"]),
                            "without_context_flag": any(not s["context_resolved"] for s in sample), "seed": seed}
        (OUT_DIR / f"{name}.judged_sample.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspaces", nargs="+")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--lang", choices=["it", "en"], default="it", help="gold triples e personaggi: it (default) o en")
    a = ap.parse_args()
    if a.lang == "en":
        global GOLD, KEY_ENTITIES
        GOLD = ROOT / "tests/e2e/data/en/gold_triples.json"
        KEY_ENTITIES = KEY_ENTITIES_EN
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for w in a.workspaces:
        print(f"== {w}", file=sys.stderr)
        results.append(evaluate(w, a.n, a.seed, not a.no_judge))
    pc = lambda x: "-" if x is None else f"{x*100:.0f}%"
    print("| Workspace | Entita' | Relazioni | Key-entity | Pair recall (controllo) | Typed recall | Pair / typed senza Pinocchio | Precisione (N) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        p = r.get("precision")
        ps = f"{pc(p['precision'])} ({p['supported']}/{p['judged']})" if p else "n/d"
        print(f"| {r['workspace']} | {r['size']['entities']} | {r['size']['relationships']} | "
              f"{pc(r['key_entity_recall']['recall'])} ({r['key_entity_recall']['found']}/{r['key_entity_recall']['total']}) | "
              f"{pc(r['pair_recall']['recall'])} ({r['pair_recall']['hits']}/{r['pair_recall']['total']}; ctrl {pc(r['pair_recall_control_shuffled']['rate'])}) | "
              f"{pc(r['typed_recall']['recall'])} ({r['typed_recall']['hits']}/{r['typed_recall']['total']}) | "
              f"{r['no_pinocchio_subset']['pair']}/{r['no_pinocchio_subset']['total']} (ctrl {r['no_pinocchio_subset']['pair_control_shuffled']}) / {r['no_pinocchio_subset']['typed']}/{r['no_pinocchio_subset']['total']} | {ps} |")


if __name__ == "__main__":
    main()
