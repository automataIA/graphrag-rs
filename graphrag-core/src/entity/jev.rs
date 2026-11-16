//! TypeSafe Jev classification of entity types and relations ("g5" extraction mode).
//!
//! GLiNER (any backend) finds the entities of a chunk; Jev (`POST /v1/systemone`) then
//! classifies, against the ontology, the type of each entity and the relation of every
//! ORDERED entity pair, replacing GLiNER's own relations. Only relations explicitly stated
//! in the chunk are accepted. Per chunk, two requests:
//!
//! * **A** - one `choice` per entity (ontology entity types + `OTHER`), and per ordered pair
//!   a `choice` over the ontology relation types + `NONE` and a `noul` "link" (the text
//!   states a direct interaction/tie between the two).
//! * **B** - only for pairs with a non-`NONE` option and `link >= candidate_floor`: a
//!   `noul` "evidence" ("the text explicitly states that X <relation description> Y") and a
//!   `choice` "span" over the chunk's sentences (which one proves it).
//!
//! With `filter_entities` (default) a request **E** comes first: per candidate, a `noul`
//! "this is a specific, identifiable referent" (plus the type `choice`). Candidates below
//! `min_entity` are removed from the graph; A then carries only the pair questions, over the
//! surviving (capped) entities.
//!
//! A relation is accepted when the chosen type is not `NONE`, `p(type) >= min_type_prob`,
//! `link >= min_link` and `evidence >= min_evidence`. Everything Jev answered is appended
//! to a JSONL raw log (one line per chunk) so thresholds can be swept offline without
//! calling the API again (`tests/e2e/scripts/jev_sweep.py`).
//!
//! The instructions sent to Jev are in Italian (they worked best on the Italian test
//! corpus; Jev is multilingual). Feature: needs `ureq` (blocking HTTP, run inside
//! `spawn_blocking` by the caller).

use crate::config::JevConfig;
use crate::core::{error::GraphRAGError, Entity, EntityId, Relationship, TextChunk};
use crate::ontology::Ontology;
use serde_json::{json, Map, Value};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// "No relation" option of the relation choice.
pub const NONE: &str = "NONE";
/// "No fitting type" option of the entity choice.
pub const OTHER: &str = "OTHER";
/// "No sentence proves it" option of the span choice.
pub const NO_SPAN: &str = "none";

const MAX_ATTEMPTS: u32 = 5;
/// Option-count limit of a Jev choice question.
const MAX_SENTENCES: usize = 200;

/// Split a chunk into sentences on `. ! ? ;` and newlines (deterministic, trimmed, no
/// empty items). Runs of terminators and closing quotes stay with the sentence.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        let s = cur.trim();
        if !s.is_empty() {
            out.push(s.to_string());
        }
        cur.clear();
    };
    while let Some(c) = chars.next() {
        if c == '\n' {
            flush(&mut cur, &mut out);
            continue;
        }
        cur.push(c);
        if matches!(c, '.' | '!' | '?' | ';') {
            while let Some(&n) = chars.peek() {
                if matches!(
                    n,
                    '.' | '!' | '?' | ';' | '»' | '"' | '”' | '\'' | '’' | ')'
                ) {
                    cur.push(n);
                    chars.next();
                } else {
                    break;
                }
            }
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

/// One answer of a `choice` question.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// Chosen option
    pub choice: String,
    /// Probability of every option, in the order returned by the API
    pub probs: Vec<(String, f64)>,
    /// Jev's own confidence in the choice
    pub confidence: f64,
}

impl Choice {
    /// Probability of `option` (0.0 when absent).
    pub fn prob(&self, option: &str) -> f64 {
        self.probs
            .iter()
            .find(|(o, _)| o == option)
            .map_or(0.0, |(_, p)| *p)
    }

    /// Most probable option other than `NONE`.
    pub fn top_non_none(&self) -> Option<&str> {
        self.probs
            .iter()
            .filter(|(o, _)| o != NONE)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(o, _)| o.as_str())
    }

    fn probs_json(&self) -> Value {
        Value::Object(
            self.probs
                .iter()
                .map(|(o, p)| (o.clone(), json!(p)))
                .collect(),
        )
    }
}

/// Token usage reported by one request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Input tokens
    pub input_tokens: u64,
    /// Output tokens
    pub output_tokens: u64,
}

/// Parsed `/v1/systemone` response.
#[derive(Debug, Clone)]
pub struct Response {
    answers: Value,
    /// Token usage
    pub usage: Usage,
}

impl Response {
    /// Parse a response body. A body without `answers` (API error object) is an error.
    pub fn parse(body: &Value) -> Result<Self, String> {
        let Some(answers) = body.get("answers").filter(|a| a.is_object()) else {
            let detail = body
                .get("error")
                .map(|e| e.to_string())
                .unwrap_or_else(|| body.to_string());
            return Err(format!("response without answers: {detail}"));
        };
        let usage = body.get("usage").map_or_else(Usage::default, |u| Usage {
            input_tokens: u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
            output_tokens: u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
        });
        Ok(Self {
            answers: answers.clone(),
            usage,
        })
    }

    /// Answer of a `choice` question.
    pub fn choice(&self, id: &str) -> Result<Choice, String> {
        let a = self
            .answers
            .get(id)
            .ok_or_else(|| format!("missing answer {id}"))?;
        let choice = a
            .get("choice")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("answer {id} has no choice"))?
            .to_string();
        let probs = a
            .get("probabilities")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_f64().unwrap_or(0.0)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Choice {
            choice,
            probs,
            confidence: a.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
        })
    }

    /// Answer of a `noul` question, in `[0, 1]`.
    pub fn noul(&self, id: &str) -> Result<f64, String> {
        self.answers
            .get(id)
            .and_then(|a| a.get("noul"))
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("missing noul answer {id}"))
    }
}

fn type_criteria(types: &[crate::ontology::TypeDef], extra: (&str, &str)) -> Value {
    let mut m = Map::new();
    for t in types {
        let d = if t.description.is_empty() {
            t.name.as_str()
        } else {
            t.description.as_str()
        };
        m.insert(t.name.clone(), json!(d));
    }
    m.insert(extra.0.to_string(), json!(extra.1));
    Value::Object(m)
}

fn state(text: &str, sentences: &[String]) -> Value {
    let s: Map<String, Value> = sentences
        .iter()
        .take(MAX_SENTENCES)
        .enumerate()
        .map(|(i, s)| (format!("s{}", i + 1), json!(s)))
        .collect();
    json!({ "text": text, "sentences": s })
}

fn insert_type_questions(qs: &mut Map<String, Value>, ontology: &Ontology, names: &[String]) {
    let crit = type_criteria(
        &ontology.entity_types,
        (OTHER, "nessuno dei tipi precedenti si adatta"),
    );
    for (i, n) in names.iter().enumerate() {
        qs.insert(
            format!("t{i}"),
            json!({"type": "choice",
                "instructions": format!("Qual è il tipo dell'entità «{n}» così come compare in questo testo?"),
                "criteria": crit}),
        );
    }
}

/// Instructions of the entity-existence `noul` question of request E (`{n}` = candidate
/// name). Chosen with `tests/e2e/scripts/jev_entity_probe.py` (prompt v2) on
/// `tests/e2e/data/it/jev_entities_gold.json`.
pub const ENTITY_EXISTS_PROMPT: &str = "In questo testo, «{n}» indica un referente specifico e identificabile della storia: un personaggio, un evento preciso, oppure un luogo preciso e riconoscibile (anche senza nome proprio, per esempio la bottega o la casa di un certo personaggio)? Rispondi no se la parola e' usata in modo generico, senza un referente univoco (il mare, la scuola, il bosco in generale), oppure se e' una parte del corpo, un oggetto qualunque, un verbo o un frammento di parola.";

/// Request E (entity filter): per candidate the type choice `t{i}` (when
/// `classify_entity_types`) and the existence `noul` `e{i}`.
pub fn request_e(
    cfg: &JevConfig,
    ontology: &Ontology,
    text: &str,
    sentences: &[String],
    names: &[String],
) -> Value {
    let mut qs = Map::new();
    if cfg.classify_entity_types {
        insert_type_questions(&mut qs, ontology, names);
    }
    for (i, n) in names.iter().enumerate() {
        qs.insert(
            format!("e{i}"),
            json!({"type": "noul", "instructions": ENTITY_EXISTS_PROMPT.replace("{n}", n)}),
        );
    }
    json!({"model": cfg.model, "state": state(text, sentences), "questions": qs})
}

/// Entity filter rule: a candidate is kept when its existence score is `>= min_entity`.
pub fn entity_kept(cfg: &JevConfig, exists: f64) -> bool {
    // Compared in f32 so a score printed as the threshold (e.g. 0.8) is kept.
    (exists as f32) >= cfg.min_entity
}

/// Request A: entity types, relation choice and link for every ordered pair.
/// Question ids: `t{i}`, `r{i}_{j}`, `n{i}_{j}`. With `filter_entities` the types are decided
/// by request E, so there are no `t{i}` questions.
pub fn request_a(
    cfg: &JevConfig,
    ontology: &Ontology,
    text: &str,
    sentences: &[String],
    names: &[String],
) -> Value {
    let mut qs = Map::new();
    if cfg.classify_entity_types && !cfg.filter_entities {
        insert_type_questions(&mut qs, ontology, names);
    }
    let rel_crit = type_criteria(
        &ontology.relation_types,
        (
            NONE,
            "il testo non afferma esplicitamente nessuna relazione del soggetto verso l'oggetto",
        ),
    );
    for (i, x) in names.iter().enumerate() {
        for (j, y) in names.iter().enumerate() {
            if i == j {
                continue;
            }
            qs.insert(
                format!("r{i}_{j}"),
                json!({"type": "choice",
                    "instructions": format!(
                        "Nel testo, quale relazione ESPLICITA ha «{x}» (soggetto) verso «{y}» (oggetto)? \
                         Conta solo ciò che il testo afferma direttamente, non ciò che si può dedurre dalla trama \
                         o dalla conoscenza esterna. Se il testo non afferma nessuna relazione di «{x}» verso «{y}», scegli NONE."),
                    "criteria": rel_crit}),
            );
            qs.insert(
                format!("n{i}_{j}"),
                json!({"type": "noul",
                    "instructions": format!(
                        "Il testo afferma esplicitamente un'interazione diretta o un legame tra «{x}» e «{y}».")}),
            );
        }
    }
    json!({"model": cfg.model, "state": state(text, sentences), "questions": qs})
}

/// Request B: evidence and proving sentence for the candidate pairs
/// `(i, j, relation type)`. Question ids: `v{i}_{j}`, `s{i}_{j}`.
pub fn request_b(
    cfg: &JevConfig,
    ontology: &Ontology,
    text: &str,
    sentences: &[String],
    names: &[String],
    candidates: &[(usize, usize, String)],
) -> Value {
    let mut span_crit = Map::new();
    for (i, s) in sentences.iter().take(MAX_SENTENCES).enumerate() {
        span_crit.insert(format!("s{}", i + 1), json!(s));
    }
    span_crit.insert(NO_SPAN.to_string(), json!("nessuna frase lo dimostra"));
    let mut qs = Map::new();
    for (i, j, ty) in candidates {
        let (x, y) = (&names[*i], &names[*j]);
        let desc = ontology
            .relation_types
            .iter()
            .find(|t| &t.name == ty)
            .map(|t| {
                if t.description.is_empty() {
                    t.name.as_str()
                } else {
                    t.description.as_str()
                }
            })
            .unwrap_or(ty.as_str());
        qs.insert(
            format!("v{i}_{j}"),
            json!({"type": "noul",
                "instructions": format!("Il testo afferma esplicitamente che «{x}» {desc} «{y}».")}),
        );
        qs.insert(
            format!("s{i}_{j}"),
            json!({"type": "choice",
                "instructions": format!("Quale frase dimostra che «{x}» {desc} «{y}»?"),
                "criteria": span_crit}),
        );
    }
    json!({"model": cfg.model, "state": state(text, sentences), "questions": qs})
}

/// Acceptance rule: chosen type is not `NONE`, `p(type) >= min_type_prob`,
/// `link >= min_link`, `evidence >= min_evidence`.
pub fn accept(cfg: &JevConfig, rel: &Choice, link: f64, evidence: Option<f64>) -> bool {
    rel.choice != NONE
        && rel.prob(&rel.choice) >= f64::from(cfg.min_type_prob)
        && link >= f64::from(cfg.min_link)
        && evidence.is_some_and(|e| e >= f64::from(cfg.min_evidence))
}

/// Counters of one chunk (summed by the caller for the end-of-build usage line).
#[derive(Debug, Clone, Copy, Default)]
pub struct JevStats {
    /// HTTP requests that returned an answer
    pub requests: u64,
    /// Input tokens
    pub input_tokens: u64,
    /// Output tokens
    pub output_tokens: u64,
    /// Failed chunks
    pub errors: u64,
    /// Time spent in requests, ms
    pub latency_ms: u64,
    /// Accepted relations
    pub accepted: u64,
    /// Ordered pairs classified
    pub pairs: u64,
    /// Entities not sent to Jev because of `max_entities_per_chunk`
    pub dropped_entities: u64,
    /// Entities removed by the existence filter (`min_entity`)
    pub filtered_entities: u64,
}

impl JevStats {
    /// Add `o` to `self`.
    pub fn add(&mut self, o: &JevStats) {
        self.requests += o.requests;
        self.input_tokens += o.input_tokens;
        self.output_tokens += o.output_tokens;
        self.errors += o.errors;
        self.latency_ms += o.latency_ms;
        self.accepted += o.accepted;
        self.pairs += o.pairs;
        self.dropped_entities += o.dropped_entities;
        self.filtered_entities += o.filtered_entities;
    }

    /// The end-of-build summary line.
    pub fn summary(&self) -> String {
        let mut line = format!(
            "Jev usage: {} requests, {} input / {} output tokens, {} errors, {} ms total, {} relations accepted of {} pairs",
            self.requests,
            self.input_tokens,
            self.output_tokens,
            self.errors,
            self.latency_ms,
            self.accepted,
            self.pairs
        );
        if self.filtered_entities > 0 {
            line.push_str(&format!(
                ", {} entities filtered out",
                self.filtered_entities
            ));
        }
        line
    }
}

/// Result for one chunk: final entities (all of the input, re-typed when classified) and
/// the accepted relations.
#[derive(Debug, Default)]
pub struct JevChunkResult {
    /// Entities to add to the graph
    pub entities: Vec<Entity>,
    /// Accepted relations
    pub relationships: Vec<Relationship>,
    /// Counters
    pub stats: JevStats,
}

struct Http {
    agent: ureq::Agent,
    endpoint: String,
    key: String,
    base_backoff_ms: u64,
}

impl Http {
    /// POST with exponential backoff on 429/529/5xx and transport errors.
    fn post(&self, body: &Value) -> Result<Response, String> {
        let mut last = String::new();
        for attempt in 0..MAX_ATTEMPTS {
            if attempt > 0 {
                std::thread::sleep(Duration::from_millis(self.base_backoff_ms << (attempt - 1)));
            }
            let r = self
                .agent
                .post(&self.endpoint)
                .set("Authorization", &format!("Bearer {}", self.key))
                .set("Content-Type", "application/json")
                .send_json(body);
            match r {
                Ok(resp) => {
                    let v: Value = resp
                        .into_json()
                        .map_err(|e| format!("response was not valid JSON: {e}"))?;
                    return Response::parse(&v);
                },
                Err(ureq::Error::Status(code, resp)) => {
                    let detail = resp.into_string().unwrap_or_default();
                    let detail: String = detail.chars().take(300).collect();
                    last = format!("HTTP {code}: {detail}");
                    if code != 429 && code < 500 {
                        return Err(last);
                    }
                },
                Err(ureq::Error::Transport(t)) => last = format!("transport error: {t}"),
            }
        }
        Err(format!("gave up after {MAX_ATTEMPTS} attempts ({last})"))
    }
}

/// Jev classifier shared by all chunk tasks.
pub struct JevClassifier {
    cfg: JevConfig,
    ontology: Ontology,
    http: Http,
    log: Option<Mutex<std::fs::File>>,
}

impl std::fmt::Debug for JevClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevClassifier")
            .field("cfg", &self.cfg)
            .finish()
    }
}

fn norm(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

impl JevClassifier {
    /// Build the classifier. The API key is read from the env var named by
    /// `cfg.api_key_env`. `default_log` is used when `cfg.raw_log_path` is unset.
    pub fn new(
        cfg: JevConfig,
        ontology: Ontology,
        default_log: Option<PathBuf>,
    ) -> Result<Self, GraphRAGError> {
        let key = std::env::var(&cfg.api_key_env)
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| GraphRAGError::Config {
                message: format!(
                    "gliner.jev.enabled needs the API key in the env var {}",
                    cfg.api_key_env
                ),
            })?;
        let log_path = cfg.raw_log_path.as_ref().map(PathBuf::from).or(default_log);
        let log = match log_path {
            Some(p) => {
                if let Some(dir) = p.parent() {
                    std::fs::create_dir_all(dir).ok();
                }
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                    .map_err(|e| GraphRAGError::Config {
                        message: format!("cannot open Jev raw log {}: {e}", p.display()),
                    })?;
                Some(Mutex::new(f))
            },
            None => None,
        };
        let http = Http {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(cfg.timeout_secs.max(1)))
                .build(),
            endpoint: cfg.api_endpoint.clone(),
            key,
            base_backoff_ms: 1000,
        };
        Ok(Self {
            cfg,
            ontology,
            http,
            log,
        })
    }

    /// Configured concurrency (>= 1).
    pub fn max_concurrent(&self) -> usize {
        self.cfg.max_concurrent.max(1)
    }

    fn write_log(&self, record: &Value) {
        if let Some(log) = &self.log {
            if let Ok(mut f) = log.lock() {
                if let Err(e) = writeln!(f, "{record}") {
                    tracing_warn(&format!("Jev raw log write failed: {e}"));
                }
            }
        }
    }

    /// Classify one chunk (blocking). Never fails: on error the input entities are
    /// returned without relations and `stats.errors` is 1.
    pub fn classify_chunk(&self, chunk: &TextChunk, entities: &[Entity]) -> JevChunkResult {
        let t0 = Instant::now();
        let mut stats = JevStats::default();
        let sentences = split_sentences(&chunk.content);

        // One candidate per distinct name (highest GLiNER confidence), best first.
        let mut by_conf: Vec<&Entity> = Vec::new();
        for e in entities {
            match by_conf.iter().position(|x| norm(&x.name) == norm(&e.name)) {
                Some(i) if by_conf[i].confidence >= e.confidence => {},
                Some(i) => by_conf[i] = e,
                None => by_conf.push(e),
            }
        }
        by_conf.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

        let mut record = json!({
            "chunk_id": chunk.id.to_string(),
            "text": chunk.content,
            "sentences": sentences,
            "model": self.cfg.model,
            "entities": [],
            "dropped_entities": [],
            "pairs": [],
            "errors": [],
        });

        let outcome = if by_conf.is_empty() {
            Ok((entities.to_vec(), Vec::new()))
        } else if self.cfg.filter_entities {
            self.classify_filtered(chunk, &sentences, by_conf, &mut stats, &mut record)
        } else {
            self.classify_legacy(chunk, &sentences, by_conf, &mut stats, &mut record)
        };
        stats.latency_ms = t0.elapsed().as_millis() as u64;
        let result = match outcome {
            Ok((all, rels)) => {
                stats.accepted = rels.len() as u64;
                JevChunkResult {
                    entities: all,
                    relationships: rels,
                    stats,
                }
            },
            Err(e) => {
                stats.errors = 1;
                stats.accepted = 0;
                stats.filtered_entities = 0;
                tracing_warn(&format!("Jev failed for chunk {}: {e}", chunk.id));
                record["errors"] = json!([e]);
                JevChunkResult {
                    entities: entities.to_vec(),
                    relationships: Vec::new(),
                    stats,
                }
            },
        };
        record["usage"] = json!({
            "requests": result.stats.requests,
            "input_tokens": result.stats.input_tokens,
            "output_tokens": result.stats.output_tokens,
        });
        record["latency_ms"] = json!(result.stats.latency_ms);
        self.write_log(&record);
        result
    }

    /// `filter_entities = false`: cap first, types inside request A.
    fn classify_legacy(
        &self,
        chunk: &TextChunk,
        sentences: &[String],
        mut by_conf: Vec<&Entity>,
        stats: &mut JevStats,
        record: &mut Value,
    ) -> Result<(Vec<Entity>, Vec<Relationship>), String> {
        let cap = self.cfg.max_entities_per_chunk.max(1);
        let dropped: Vec<&Entity> = by_conf.split_off(by_conf.len().min(cap));
        stats.dropped_entities = dropped.len() as u64;
        if !dropped.is_empty() {
            tracing_debug(&format!(
                "Jev: chunk {} has {} entities, dropped {} below max_entities_per_chunk={}",
                chunk.id,
                by_conf.len() + dropped.len(),
                dropped.len(),
                cap
            ));
        }
        record["dropped_entities"] = dropped_log(&dropped);
        let names: Vec<String> = by_conf.iter().map(|e| e.name.clone()).collect();
        let (mut all, rels) = self.run(chunk, sentences, &by_conf, &names, None, stats, record)?;
        // Entities not sent to Jev keep their GLiNER type.
        for d in dropped {
            all.push(d.clone());
        }
        Ok((all, rels))
    }

    /// `filter_entities = true`: request E on all candidates, removal below `min_entity`,
    /// cap on the survivors, request A (pairs only) and B on the capped survivors.
    fn classify_filtered(
        &self,
        chunk: &TextChunk,
        sentences: &[String],
        cands: Vec<&Entity>,
        stats: &mut JevStats,
        record: &mut Value,
    ) -> Result<(Vec<Entity>, Vec<Relationship>), String> {
        let cfg = &self.cfg;
        let names: Vec<String> = cands.iter().map(|e| e.name.clone()).collect();
        let body = request_e(cfg, &self.ontology, &chunk.content, sentences, &names);
        let resp = self
            .http
            .post(&body)
            .map_err(|e| format!("request E: {e}"))?;
        stats.requests += 1;
        stats.input_tokens += resp.usage.input_tokens;
        stats.output_tokens += resp.usage.output_tokens;

        let mut kept: Vec<(&Entity, Option<Choice>)> = Vec::new();
        let mut ent_log = Vec::new();
        let mut filtered_log = Vec::new();
        for (i, e) in cands.iter().enumerate() {
            let exists = resp.noul(&format!("e{i}"))?;
            let c = if cfg.classify_entity_types {
                Some(resp.choice(&format!("t{i}"))?)
            } else {
                None
            };
            let keep = entity_kept(cfg, exists);
            ent_log.push(json!({
                "name": e.name, "gliner_type": e.entity_type, "gliner_conf": e.confidence,
                "jev_type": c.as_ref().map(|c| c.choice.clone()),
                "jev_probs": c.as_ref().map(Choice::probs_json),
                "jev_confidence": c.as_ref().map(|c| c.confidence),
                "exists": exists, "kept": keep,
            }));
            if keep {
                kept.push((e, c));
            } else {
                filtered_log.push(json!({
                    "name": e.name, "gliner_type": e.entity_type, "exists": exists,
                }));
            }
        }
        stats.filtered_entities = filtered_log.len() as u64;
        record["entities"] = Value::Array(ent_log);
        record["filtered_entities"] = Value::Array(filtered_log);

        // Cap on the survivors (already best GLiNER confidence first).
        let cap = cfg.max_entities_per_chunk.max(1);
        let beyond = kept.split_off(kept.len().min(cap));
        stats.dropped_entities = beyond.len() as u64;
        record["dropped_entities"] =
            dropped_log(&beyond.iter().map(|(e, _)| *e).collect::<Vec<_>>());

        let ents: Vec<&Entity> = kept.iter().map(|(e, _)| *e).collect();
        let types: Vec<Option<Choice>> = kept.into_iter().map(|(_, c)| c).collect();
        let (mut all, rels) = if ents.len() >= 2 {
            let names: Vec<String> = ents.iter().map(|e| e.name.clone()).collect();
            self.run(chunk, sentences, &ents, &names, Some(types), stats, record)?
        } else {
            // Fewer than two survivors: no pair to classify (and no request A).
            let mut out: Vec<Entity> = Vec::new();
            for (e, c) in ents.iter().zip(&types) {
                let e = retyped(e, c.as_ref());
                if !out.iter().any(|x| x.id == e.id) {
                    out.push(e);
                }
            }
            (out, Vec::new())
        };
        // Survivors beyond the cap keep their Jev type and get no relations.
        for (e, c) in beyond {
            let e = retyped(e, c.as_ref());
            if !all.iter().any(|x| x.id == e.id) {
                all.push(e);
            }
        }
        Ok((all, rels))
    }

    /// Requests A (and B). `pre_types` = types already decided by request E (then A has no
    /// type questions and the raw-log `entities` list is left to the caller).
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        chunk: &TextChunk,
        sentences: &[String],
        ents: &[&Entity],
        names: &[String],
        pre_types: Option<Vec<Option<Choice>>>,
        stats: &mut JevStats,
        record: &mut Value,
    ) -> Result<(Vec<Entity>, Vec<Relationship>), String> {
        let cfg = &self.cfg;
        // Request A
        let body = request_a(cfg, &self.ontology, &chunk.content, sentences, names);
        let resp = self
            .http
            .post(&body)
            .map_err(|e| format!("request A: {e}"))?;
        stats.requests += 1;
        stats.input_tokens += resp.usage.input_tokens;
        stats.output_tokens += resp.usage.output_tokens;

        let type_of: Vec<Option<Choice>> = match pre_types {
            Some(t) => t,
            None => {
                let mut type_of = Vec::new();
                let mut ent_log = Vec::new();
                for (i, e) in ents.iter().enumerate() {
                    let c = if cfg.classify_entity_types {
                        Some(resp.choice(&format!("t{i}"))?)
                    } else {
                        None
                    };
                    ent_log.push(json!({
                        "name": e.name, "gliner_type": e.entity_type, "gliner_conf": e.confidence,
                        "jev_type": c.as_ref().map(|c| c.choice.clone()),
                        "jev_probs": c.as_ref().map(Choice::probs_json),
                        "jev_confidence": c.as_ref().map(|c| c.confidence),
                    }));
                    type_of.push(c);
                }
                record["entities"] = Value::Array(ent_log);
                type_of
            },
        };

        struct Pair {
            i: usize,
            j: usize,
            rel: Choice,
            link: f64,
            evidence: Option<f64>,
            span: Option<Choice>,
        }
        let mut pairs: Vec<Pair> = Vec::new();
        for i in 0..names.len() {
            for j in 0..names.len() {
                if i == j {
                    continue;
                }
                pairs.push(Pair {
                    i,
                    j,
                    rel: resp.choice(&format!("r{i}_{j}"))?,
                    link: resp.noul(&format!("n{i}_{j}"))?,
                    evidence: None,
                    span: None,
                });
            }
        }
        stats.pairs += pairs.len() as u64;

        // Request B
        let cands: Vec<(usize, usize, String)> = pairs
            .iter()
            .filter(|p| p.link >= f64::from(cfg.candidate_floor))
            .filter_map(|p| p.rel.top_non_none().map(|t| (p.i, p.j, t.to_string())))
            .collect();
        if !cands.is_empty() {
            let body = request_b(
                cfg,
                &self.ontology,
                &chunk.content,
                sentences,
                names,
                &cands,
            );
            let resp_b = self
                .http
                .post(&body)
                .map_err(|e| format!("request B: {e}"))?;
            stats.requests += 1;
            stats.input_tokens += resp_b.usage.input_tokens;
            stats.output_tokens += resp_b.usage.output_tokens;
            for (i, j, _) in &cands {
                if let Some(p) = pairs.iter_mut().find(|p| p.i == *i && p.j == *j) {
                    p.evidence = Some(resp_b.noul(&format!("v{i}_{j}"))?);
                    p.span = Some(resp_b.choice(&format!("s{i}_{j}"))?);
                }
            }
        }

        // Final entities: Jev type when classified, one entity per distinct id.
        let mut final_ents: Vec<Entity> = Vec::new();
        let mut ids: Vec<EntityId> = Vec::new();
        for (e, c) in ents.iter().zip(&type_of) {
            let e = retyped(e, c.as_ref());
            ids.push(e.id.clone());
            if !final_ents.iter().any(|x| x.id == e.id) {
                final_ents.push(e);
            }
        }

        let mut rels = Vec::new();
        let mut pair_log = Vec::new();
        for p in &pairs {
            let ok = accept(cfg, &p.rel, p.link, p.evidence);
            if ok && ids[p.i] != ids[p.j] {
                let sentence = p
                    .span
                    .as_ref()
                    .and_then(|s| s.choice.strip_prefix('s'))
                    .and_then(|n| n.parse::<usize>().ok())
                    .and_then(|n| sentences.get(n.wrapping_sub(1)));
                let mut r = Relationship::new(
                    ids[p.i].clone(),
                    ids[p.j].clone(),
                    p.rel.choice.clone(),
                    p.evidence.unwrap_or(0.0) as f32,
                );
                r.description = sentence.cloned();
                r.context.push(chunk.id.clone());
                rels.push(r);
            }
            pair_log.push(json!({
                "head": names[p.i], "tail": names[p.j],
                "rel_choice": p.rel.choice, "rel_probs": p.rel.probs_json(),
                "rel_confidence": p.rel.confidence,
                "link": p.link, "evidence": p.evidence,
                "span_choice": p.span.as_ref().map(|s| s.choice.clone()),
                "span_probs": p.span.as_ref().map(Choice::probs_json),
                "accepted": ok,
            }));
        }
        record["pairs"] = Value::Array(pair_log);
        Ok((final_ents, rels))
    }
}

fn dropped_log(dropped: &[&Entity]) -> Value {
    Value::Array(
        dropped
            .iter()
            .map(|e| json!({"name": e.name, "gliner_type": e.entity_type, "gliner_conf": e.confidence}))
            .collect(),
    )
}

/// The entity with the Jev type applied (GLiNER type kept when not classified).
fn retyped(e: &Entity, c: Option<&Choice>) -> Entity {
    let mut e = e.clone();
    if let Some(c) = c {
        e.entity_type = c.choice.clone();
        e.id = EntityId::from_type_and_name(&e.entity_type, &e.name);
    }
    e
}

fn tracing_warn(msg: &str) {
    #[cfg(feature = "tracing")]
    tracing::warn!("{msg}");
    #[cfg(not(feature = "tracing"))]
    eprintln!("warning: {msg}");
}

fn tracing_debug(msg: &str) {
    #[cfg(feature = "tracing")]
    tracing::info!("{msg}");
    #[cfg(not(feature = "tracing"))]
    eprintln!("{msg}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(c: &str, probs: &[(&str, f64)]) -> Choice {
        Choice {
            choice: c.into(),
            probs: probs.iter().map(|(o, p)| (o.to_string(), *p)).collect(),
            confidence: 0.9,
        }
    }

    #[test]
    fn sentences_split_on_terminators_and_newlines() {
        let s = split_sentences("Ciao a tutti! Come va? Bene; grazie...\nSecondo rigo. «Sì.» Fine");
        assert_eq!(
            s,
            [
                "Ciao a tutti!",
                "Come va?",
                "Bene;",
                "grazie...",
                "Secondo rigo.",
                "«Sì.»",
                "Fine"
            ]
        );
        assert!(split_sentences("  \n \n").is_empty());
        assert_eq!(split_sentences("a. b"), split_sentences("a. b"));
    }

    #[test]
    fn request_a_has_ordered_pairs_and_none_option() {
        // filter_entities=false: the pre-filter (g5) shape, types inside request A.
        let cfg = JevConfig {
            filter_entities: false,
            ..JevConfig::default()
        };
        let o = Ontology::generic();
        let names = vec![
            "Geppetto".to_string(),
            "Pinocchio".to_string(),
            "Gatto".to_string(),
        ];
        let sents = split_sentences("Geppetto scolpì Pinocchio. Il Gatto rise.");
        let body = request_a(&cfg, &o, "testo", &sents, &names);
        assert_eq!(body["model"], "jev-latest");
        assert_eq!(body["state"]["text"], "testo");
        assert_eq!(body["state"]["sentences"]["s2"], "Il Gatto rise.");
        let qs = body["questions"].as_object().unwrap();
        // 3 entity types + 6 ordered pairs * (choice + noul)
        assert_eq!(qs.len(), 3 + 6 * 2);
        let r = &qs["r0_1"];
        assert_eq!(r["type"], "choice");
        assert!(r["criteria"].get(NONE).is_some());
        assert!(r["criteria"].get("FRIEND_OF").is_some());
        let instr = r["instructions"].as_str().unwrap();
        assert!(instr.contains("«Geppetto» (soggetto) verso «Pinocchio» (oggetto)"));
        assert!(qs.contains_key("r1_0") && qs.contains_key("n1_0"));
        assert!(!qs.contains_key("r0_0"));
        assert!(qs["t0"]["criteria"].get(OTHER).is_some());
        assert_eq!(qs["n0_1"]["type"], "noul");
    }

    #[test]
    fn request_a_without_entity_classification_has_no_type_questions() {
        let cfg = JevConfig {
            classify_entity_types: false,
            ..JevConfig::default()
        };
        let names = vec!["A".to_string(), "B".to_string()];
        let body = request_a(&cfg, &Ontology::generic(), "t", &[], &names);
        assert_eq!(body["questions"].as_object().unwrap().len(), 4);
    }

    #[test]
    fn request_a_in_filter_mode_has_only_pair_questions() {
        let cfg = JevConfig::default();
        assert!(cfg.filter_entities && cfg.classify_entity_types);
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let body = request_a(&cfg, &Ontology::generic(), "t", &[], &names);
        let qs = body["questions"].as_object().unwrap();
        assert_eq!(qs.len(), 6 * 2);
        assert!(qs
            .keys()
            .all(|k| !k.starts_with('t') && !k.starts_with('e')));
    }

    #[test]
    fn request_e_has_type_and_existence_questions_only() {
        let cfg = JevConfig::default();
        let names = vec!["Geppetto".to_string(), "mare".to_string()];
        let body = request_e(&cfg, &Ontology::generic(), "testo", &[], &names);
        assert_eq!(body["model"], "jev-latest");
        assert_eq!(body["state"]["text"], "testo");
        let qs = body["questions"].as_object().unwrap();
        assert_eq!(qs.len(), 4);
        assert!(qs["t1"]["criteria"].get(OTHER).is_some());
        assert_eq!(qs["t0"]["type"], "choice");
        assert_eq!(qs["e1"]["type"], "noul");
        assert_eq!(
            qs["e1"]["instructions"].as_str().unwrap(),
            ENTITY_EXISTS_PROMPT.replace("{n}", "mare")
        );
        assert!(qs["e0"]["instructions"]
            .as_str()
            .unwrap()
            .starts_with("In questo testo, «Geppetto» indica un referente specifico"));
        assert!(!qs.keys().any(|k| k.starts_with('r') || k.starts_with('n')));
        let no_types = JevConfig {
            classify_entity_types: false,
            ..JevConfig::default()
        };
        let body = request_e(&no_types, &Ontology::generic(), "t", &[], &names);
        assert_eq!(body["questions"].as_object().unwrap().len(), 2);
    }

    #[test]
    fn entity_filter_threshold_is_inclusive() {
        let cfg = JevConfig::default();
        assert!(
            entity_kept(&cfg, 0.5),
            "score exactly at min_entity is kept"
        );
        assert!(entity_kept(&cfg, 0.9));
        assert!(!entity_kept(&cfg, 0.4999));
        let strict = JevConfig {
            min_entity: 0.8,
            ..JevConfig::default()
        };
        assert!(!entity_kept(&strict, 0.5) && entity_kept(&strict, 0.8));
    }

    #[test]
    fn request_b_has_evidence_and_span_with_none() {
        let o = Ontology::generic();
        let names = vec!["Geppetto".to_string(), "Pinocchio".to_string()];
        let sents = split_sentences("Uno. Due.");
        let body = request_b(
            &JevConfig::default(),
            &o,
            "t",
            &sents,
            &names,
            &[(0, 1, "CREATED".to_string())],
        );
        let qs = body["questions"].as_object().unwrap();
        assert_eq!(qs.len(), 2);
        let v = qs["v0_1"]["instructions"].as_str().unwrap();
        assert!(v.starts_with("Il testo afferma esplicitamente che «Geppetto» "));
        assert!(v.ends_with(" «Pinocchio»."));
        let crit = qs["s0_1"]["criteria"].as_object().unwrap();
        assert_eq!(crit.len(), 3);
        assert!(crit.contains_key("s1") && crit.contains_key("s2") && crit.contains_key(NO_SPAN));
    }

    #[test]
    fn acceptance_logic() {
        let cfg = JevConfig::default(); // min_link 0.5, min_evidence 0.5, min_type_prob 0
        let f = choice("FRIEND_OF", &[("FRIEND_OF", 0.8), (NONE, 0.2)]);
        assert!(accept(&cfg, &f, 0.7, Some(0.6)));
        assert!(!accept(&cfg, &f, 0.4, Some(0.6)), "link too low");
        assert!(!accept(&cfg, &f, 0.7, Some(0.4)), "evidence too low");
        assert!(!accept(&cfg, &f, 0.7, None), "no evidence requested");
        let n = choice(NONE, &[("FRIEND_OF", 0.3), (NONE, 0.7)]);
        assert!(!accept(&cfg, &n, 0.9, Some(0.9)), "NONE never accepted");
        let strict = JevConfig {
            min_type_prob: 0.9,
            ..JevConfig::default()
        };
        assert!(!accept(&strict, &f, 0.7, Some(0.6)), "type prob too low");
        assert_eq!(n.top_non_none(), Some("FRIEND_OF"));
    }

    #[test]
    fn response_parsing_and_errors() {
        let ok = json!({
            "answers": {
                "r0_1": {"choice": "FRIEND_OF", "probabilities": {"FRIEND_OF": 0.9, "NONE": 0.1}, "confidence": 0.8},
                "n0_1": {"noul": 0.75}
            },
            "usage": {"input_tokens": 123, "output_tokens": 4}
        });
        let r = Response::parse(&ok).unwrap();
        assert_eq!(
            r.usage,
            Usage {
                input_tokens: 123,
                output_tokens: 4
            }
        );
        let c = r.choice("r0_1").unwrap();
        assert_eq!(c.choice, "FRIEND_OF");
        assert!((c.prob("FRIEND_OF") - 0.9).abs() < 1e-9);
        assert_eq!(r.noul("n0_1").unwrap(), 0.75);
        assert!(r.choice("zzz").unwrap_err().contains("missing answer"));
        assert!(r.noul("zzz").is_err());

        let err = json!({"error": {"type": "max_tokens_exceeded"}});
        assert!(Response::parse(&err)
            .unwrap_err()
            .contains("max_tokens_exceeded"));
    }

    #[test]
    fn stats_summary_line() {
        let mut s = JevStats::default();
        s.add(&JevStats {
            requests: 2,
            input_tokens: 100,
            output_tokens: 5,
            errors: 1,
            latency_ms: 300,
            accepted: 3,
            pairs: 12,
            dropped_entities: 0,
            filtered_entities: 0,
        });
        assert_eq!(
            s.summary(),
            "Jev usage: 2 requests, 100 input / 5 output tokens, 1 errors, 300 ms total, 3 relations accepted of 12 pairs"
        );
        s.filtered_entities = 7;
        assert!(s.summary().ends_with(", 7 entities filtered out"));
    }

    #[test]
    fn raw_log_line_and_failed_chunk_keeps_entities() {
        // Pre-filter flow (g5): request A fails.
        failed_chunk_keeps_entities(false, "request A");
    }

    #[test]
    fn filter_mode_failed_request_e_keeps_entities() {
        // Default flow: request E fails first, same fallback.
        failed_chunk_keeps_entities(true, "request E");
    }

    fn failed_chunk_keeps_entities(filter_entities: bool, failed_request: &str) {
        // The endpoint is unreachable: the chunk fails, entities survive, one JSONL line
        // with the error is written.
        let dir =
            std::env::temp_dir().join(format!("jev_test_{}_{filter_entities}", std::process::id()));
        let log = dir.join("raw.jsonl");
        std::env::set_var("JEV_TEST_KEY", "k");
        let cfg = JevConfig {
            enabled: true,
            api_endpoint: "http://127.0.0.1:1/v1/systemone".into(),
            api_key_env: "JEV_TEST_KEY".into(),
            raw_log_path: Some(log.to_string_lossy().into_owned()),
            timeout_secs: 1,
            filter_entities,
            ..JevConfig::default()
        };
        let mut jev = JevClassifier::new(cfg, Ontology::generic(), None).unwrap();
        jev.http.base_backoff_ms = 1;
        let chunk = TextChunk::new(
            crate::core::ChunkId::new("c1".into()),
            crate::core::DocumentId::new("d".into()),
            "Geppetto scolpì Pinocchio.".into(),
            0,
            26,
        );
        let ents = vec![
            Entity::new(
                EntityId::new("a".into()),
                "Geppetto".into(),
                "PERSON".into(),
                0.9,
            ),
            Entity::new(
                EntityId::new("b".into()),
                "Pinocchio".into(),
                "PERSON".into(),
                0.8,
            ),
        ];
        let r = jev.classify_chunk(&chunk, &ents);
        assert_eq!(r.entities.len(), 2);
        assert!(r.relationships.is_empty());
        assert_eq!(r.stats.errors, 1);
        let line = std::fs::read_to_string(&log).unwrap();
        let v: Value = serde_json::from_str(line.lines().next().unwrap()).unwrap();
        assert_eq!(v["chunk_id"], "c1");
        assert_eq!(v["text"], "Geppetto scolpì Pinocchio.");
        assert!(v["errors"][0].as_str().unwrap().contains(failed_request));
        assert!(v["sentences"].is_array() && v["entities"].is_array() && v["pairs"].is_array());
        std::fs::remove_dir_all(&dir).ok();
    }
}
