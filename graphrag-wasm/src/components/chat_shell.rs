//! Chat shell components — 3-column layout matching the Nordic-Minimal mockup.
//!
//! Class names mirror `Chat discussion.html` verbatim so the playwright DOM
//! parity test can use the same selectors against both pages.

use std::collections::{HashMap, HashSet};

use graphrag_core::core::{Entity, Relationship};
use graphrag_core::GraphRAG;

use super::force_layout::{ForceLayout, LayoutConfig};

// ─── Data types ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RefCard {
    pub num: u32,
    pub chunk_id: String,
    pub doc: String,
    pub loc: String,
    pub heading: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub enum AnswerSegment {
    Text(String),
    Cite(u32),
}

#[derive(Clone, Debug)]
pub struct ChatTurn {
    pub user_q: String,
    pub eyebrow: String,
    pub answer: Vec<AnswerSegment>,
    pub refs: Vec<RefCard>,
    pub timestamp: String,
}

#[derive(Clone, Debug, Default)]
pub struct SubgraphData {
    pub nodes: Vec<SubgraphNode>,
    pub edges: Vec<(f64, f64, f64, f64, bool)>, // x1, y1, x2, y2, primary
}

#[derive(Clone, Debug)]
pub struct SubgraphNode {
    pub label: String,
    pub kind: NodeKind,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub primary: bool,
}

#[derive(Clone, Debug)]
pub enum NodeKind {
    Concept,
    Person,
    Location,
    Deity,
    Event,
}

impl NodeKind {
    fn from_entity_type(t: &str) -> Self {
        let l = t.to_lowercase();
        if l.contains("person") || l.contains("character") || l.contains("people") {
            NodeKind::Person
        } else if l.contains("location") || l.contains("place") || l.contains("geo") {
            NodeKind::Location
        } else if l.contains("deity") || l.contains("god") {
            NodeKind::Deity
        } else if l.contains("event") || l.contains("date") {
            NodeKind::Event
        } else {
            NodeKind::Concept
        }
    }

    pub fn ring_color(&self) -> &'static str {
        "var(--accentSoft)"
    }

    pub fn fill_color(&self) -> &'static str {
        match self {
            NodeKind::Concept => "var(--accent)",
            NodeKind::Person => "var(--ink)",
            NodeKind::Location => "var(--hi)",
            NodeKind::Deity => "var(--danger)",
            NodeKind::Event => "var(--accentInk)",
        }
    }
}

// ─── Citation parsing ───────────────────────────────────────────────────────

/// Split LLM answer text on `[N]` citation markers into `Text`/`Cite` segments.
/// Fallback: if no markers, append one cite chip per available reference at end.
pub fn parse_answer_with_cites(raw: &str, ref_count: usize) -> Vec<AnswerSegment> {
    let mut out: Vec<AnswerSegment> = Vec::new();
    let bytes = raw.as_bytes();
    let mut i = 0;
    let mut buf = String::new();
    let mut found = false;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 && j < bytes.len() && bytes[j] == b']' {
                let num_str = &raw[i + 1..j];
                if let Ok(n) = num_str.parse::<u32>() {
                    if n >= 1 && (n as usize) <= ref_count {
                        if !buf.is_empty() {
                            out.push(AnswerSegment::Text(std::mem::take(&mut buf)));
                        }
                        out.push(AnswerSegment::Cite(n));
                        found = true;
                        i = j + 1;
                        continue;
                    }
                }
            }
        }
        buf.push(bytes[i] as char);
        i += 1;
    }
    if !buf.is_empty() {
        out.push(AnswerSegment::Text(buf));
    }
    if !found && ref_count > 0 {
        for n in 1..=(ref_count.min(3) as u32) {
            out.push(AnswerSegment::Cite(n));
        }
    }
    out
}

// ─── Ref card builder ───────────────────────────────────────────────────────

pub fn build_ref_cards(
    graphrag: &GraphRAG,
    top: &[(String, f64)],
    doc_lookup: &HashMap<String, String>,
) -> Vec<RefCard> {
    let mut out = Vec::with_capacity(top.len());
    for (idx, (chunk_id, _sim)) in top.iter().enumerate() {
        let chunk = match graphrag.get_chunk(chunk_id) {
            Some(c) => c,
            None => continue,
        };
        let content = &chunk.content;
        let heading = first_sentence(content, 70);
        let text = content.chars().take(360).collect::<String>();
        let doc_name = doc_lookup
            .get(chunk.document_id.0.as_str())
            .cloned()
            .unwrap_or_else(|| "Document".to_string());
        let loc = format!("§{}", chunk.start_offset / 512 + 1);
        out.push(RefCard {
            num: (idx + 1) as u32,
            chunk_id: chunk_id.clone(),
            doc: doc_name,
            loc,
            heading,
            text,
        });
    }
    out
}

fn first_sentence(s: &str, max_chars: usize) -> String {
    let cleaned: String = s.chars().take(800).collect();
    let stop = cleaned.find(['.', '?', '!']).unwrap_or(cleaned.len());
    let candidate: String = cleaned[..stop].trim().chars().take(max_chars).collect();
    if candidate.is_empty() {
        cleaned.chars().take(max_chars).collect()
    } else {
        candidate
    }
}

// ─── Subgraph builder ───────────────────────────────────────────────────────

pub fn build_subgraph(graphrag: &GraphRAG, top_chunk_ids: &[String]) -> SubgraphData {
    let mut entity_ids: HashSet<String> = HashSet::new();
    let mut primary_ids: HashSet<String> = HashSet::new();

    for cid in top_chunk_ids.iter().take(3) {
        if let Some(chunk) = graphrag.get_chunk(cid) {
            for eid in chunk.entities.iter().take(8) {
                entity_ids.insert(eid.0.clone());
                primary_ids.insert(eid.0.clone());
            }
        }
    }
    for cid in top_chunk_ids.iter().skip(3) {
        if let Some(chunk) = graphrag.get_chunk(cid) {
            for eid in chunk.entities.iter().take(6) {
                entity_ids.insert(eid.0.clone());
            }
        }
    }

    let mut entities: Vec<&Entity> = entity_ids
        .iter()
        .filter_map(|id| graphrag.get_entity(id))
        .collect();

    // Cap to 16 entities, prefer those with mentions
    if entities.len() > 16 {
        entities.sort_by_key(|b| std::cmp::Reverse(b.mentions.len()));
        entities.truncate(16);
    }
    let kept: HashSet<String> = entities.iter().map(|e| e.id.0.clone()).collect();

    let mut edges: Vec<(String, String, bool)> = Vec::new();
    let mut seen = HashSet::new();
    for e in &entities {
        for rel in graphrag.get_entity_relationships(&e.id.0) {
            let a = &rel.source.0;
            let b = &rel.target.0;
            if kept.contains(a) && kept.contains(b) {
                let key = if a < b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                if seen.insert(key) {
                    let primary = primary_ids.contains(a) || primary_ids.contains(b);
                    edges.push((a.clone(), b.clone(), primary));
                    if edges.len() >= 21 {
                        break;
                    }
                }
            }
        }
        if edges.len() >= 21 {
            break;
        }
    }

    layout_subgraph(&entities, &edges, &primary_ids, graphrag)
}

fn layout_subgraph(
    entities: &[&Entity],
    edges: &[(String, String, bool)],
    primary_ids: &HashSet<String>,
    _graphrag: &GraphRAG,
) -> SubgraphData {
    if entities.is_empty() {
        return SubgraphData::default();
    }

    let cfg = LayoutConfig {
        width: 320.0,
        height: 240.0,
        repulsion: 1600.0,
        attraction: 0.18,
        damping: 0.78,
        dt: 0.18,
        min_movement: 0.05,
    };
    let mut layout = ForceLayout::new(cfg);
    for e in entities {
        layout.add_node(e.id.0.clone());
    }
    for (a, b, _) in edges {
        layout.add_edge(a.clone(), b.clone());
    }
    layout.run(220);

    let positions = layout.get_positions();
    let to_canvas = |x: f64, y: f64| -> (f64, f64) {
        // Layout uses centered coords; viewBox is 0..320 / 0..240
        let cx = x + 160.0;
        let cy = y + 120.0;
        (cx.clamp(20.0, 300.0), cy.clamp(20.0, 220.0))
    };

    let mut nodes: Vec<SubgraphNode> = Vec::with_capacity(entities.len());
    for e in entities {
        let (x, y) = positions
            .get(&e.id.0)
            .map(|p| to_canvas(p.0, p.1))
            .unwrap_or((160.0, 120.0));
        let primary = primary_ids.contains(&e.id.0);
        let radius = if primary {
            7.0 + (e.mentions.len() as f64).sqrt().min(4.0)
        } else {
            5.0 + (e.mentions.len() as f64).sqrt().min(2.0)
        };
        nodes.push(SubgraphNode {
            label: e.name.clone(),
            kind: NodeKind::from_entity_type(&e.entity_type),
            x,
            y,
            radius,
            primary,
        });
    }

    let pos_lookup: HashMap<String, (f64, f64)> = entities
        .iter()
        .map(|e| {
            let p = positions
                .get(&e.id.0)
                .map(|p| to_canvas(p.0, p.1))
                .unwrap_or((160.0, 120.0));
            (e.id.0.clone(), p)
        })
        .collect();

    let edge_lines: Vec<(f64, f64, f64, f64, bool)> = edges
        .iter()
        .filter_map(|(a, b, primary)| {
            let pa = pos_lookup.get(a)?;
            let pb = pos_lookup.get(b)?;
            Some((pa.0, pa.1, pb.0, pb.1, *primary))
        })
        .collect();

    SubgraphData {
        nodes,
        edges: edge_lines,
    }
}

// Workaround unused import warning when graphrag is empty
#[allow(dead_code)]
fn _phantom_rel(_r: &Relationship) {}
