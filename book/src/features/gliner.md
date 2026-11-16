# GLiNER-Relex extraction

GLiNER-Relex is an optional entity and relation extraction path. It uses a
model-driven encoder rather than the default pattern extractor and therefore
has additional model, runtime, and feature requirements.

Enable the documented core feature only when the model is available for the
target environment. Keep the local extractor as a fallback and validate labels,
relations, and retrieval results on representative input; model output is not
guaranteed by the base quickstart.

## Jev classification (g5 mode)

With `backend = "gliner2"` GLiNER2.5 can find the entities only, and TypeSafe
Jev can classify their types and the relations between them against the
ontology (`gliner.jev.enabled = true`). Jev replaces GLiNER's own relations and
keeps only those the chunk states explicitly. Per chunk it takes two requests:
the first classifies each entity type and, for every ordered entity pair, the
relation type (ontology types plus `NONE`) and a `link` score; the second runs
only for plausible pairs (`link >= candidate_floor`) and scores the `evidence`
and picks the sentence that proves the relation. A relation is accepted when the
type is not `NONE` and `min_type_prob`, `min_link` and `min_evidence` are met.

The API key is read from the environment variable named by `api_key_env`
(default `TYPESAFE_API_KEY`); never put it in the config. The stage needs the
`ureq` feature, which `gliner2` enables. It is mutually exclusive with
`llm_relations`. A chunk that fails after the retries keeps its GLiNER entities
without relations and is counted in the end-of-build line
`Jev usage: ... tokens, ... errors, ... relations accepted of ... pairs`.

Every answer is appended to `jev_raw.jsonl` (in the workspace directory when
`auto_save` is on, or at `raw_log_path`). `tests/e2e/scripts/jev_sweep.py`
replays it to sweep `min_link`, `min_evidence` and `min_type_prob` offline,
without calling the API again.

```json5
gliner: {
  enabled: true, backend: "gliner2", model_path: "/path/to/gliner2.5-multi",
  entity_labels: ["character", "place", "event"],
  jev: { enabled: true, max_entities_per_chunk: 8, min_link: 0.5, min_evidence: 0.5 },
},
```
