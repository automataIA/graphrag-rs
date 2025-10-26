# 🌳 Hierarchical GraphRAG Integration Plan for WASM

## ✅ Progress Status

### Completed
- [x] Verified graphrag-core with leiden feature compiles for WASM target
- [x] Added "leiden" feature to graphrag-wasm/Cargo.toml dependencies

### In Progress
- [ ] Integrate HierarchicalCommunities in graphrag-wasm struct GraphRAG
- [ ] Add WASM-bindgen methods for Leiden clustering
- [ ] Implement Adaptive Query Routing in WASM
- [ ] Create Leptos UI for hierarchical navigation
- [ ] Add community levels visualization in UI
- [ ] Update persistence system for hierarchical communities
- [ ] Test complete integration and create demo

---

## 📚 Key Structures from graphrag-core

### 1. HierarchicalCommunities
```rust
pub struct HierarchicalCommunities {
    // Communities at each level (0 = finest, higher = coarser)
    pub levels: HashMap<usize, HashMap<NodeIndex, usize>>,

    // Parent-child relationships between levels
    pub hierarchy: HashMap<usize, Option<usize>>,

    // LLM-generated summaries for each community
    pub summaries: HashMap<usize, String>,

    // Entity name → metadata mapping
    pub entity_mapping: Option<HashMap<String, EntityMetadata>>,
}
```

**Methods:**
- `get_community_entities(level, community_id, graph)` - Get entities in a community
- `get_entities_metadata(entity_names)` - Get metadata for entities
- `get_community_stats(level, community_id, graph)` - Get stats (count, avg confidence, types)
- `generate_community_summary(...)` - Generate extractive summary
- `generate_hierarchical_summaries(...)` - Bottom-up summary generation
- `adaptive_retrieve(query, graph, config)` - Auto-select level based on query
- `retrieve_at_level(query, graph, level)` - Manual level selection
- `adaptive_retrieve_detailed(...)` - Returns (QueryAnalysis, results)

### 2. LeidenConfig
```rust
pub struct LeidenConfig {
    pub max_cluster_size: usize,      // Default: 10
    pub use_lcc: bool,                 // Default: true
    pub seed: Option<u64>,             // For reproducibility
    pub resolution: f32,               // Default: 1.0
    pub max_levels: usize,             // Default: 5
    pub min_improvement: f32,          // Default: 0.001
}
```

### 3. LeidenCommunityDetector
```rust
pub struct LeidenCommunityDetector {
    config: LeidenConfig,
}

impl LeidenCommunityDetector {
    pub fn new(config: LeidenConfig) -> Self
    pub fn detect_communities(graph) -> Result<HierarchicalCommunities>
}
```

### 4. EntityMetadata
```rust
pub struct EntityMetadata {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub confidence: f32,
    pub mention_count: usize,
}
```

### 5. AdaptiveRoutingConfig (from query module)
```rust
pub struct AdaptiveRoutingConfig {
    pub enabled: bool,
    pub default_level: usize,
    pub keyword_weight: f32,    // 0.0-1.0, default: 0.5
    pub length_weight: f32,     // 0.0-1.0, default: 0.3
    pub entity_weight: f32,     // 0.0-1.0, default: 0.2
}
```

---

## 🎯 Integration Tasks

### Task 1: Extend GraphRAG struct in lib.rs

Add field to store hierarchical communities:

```rust
#[wasm_bindgen]
pub struct GraphRAG {
    // ... existing fields ...

    // NEW: Hierarchical community structure
    #[wasm_bindgen(skip)]
    pub hierarchical_communities: Option<HierarchicalCommunities>,
}
```

### Task 2: Add WASM-bindgen Methods

```rust
#[wasm_bindgen]
impl GraphRAG {
    /// Detect hierarchical communities using Leiden algorithm
    pub async fn detect_communities(&mut self, config_json: &str) -> Result<(), JsValue>

    /// Get communities at a specific level as JSON
    pub fn get_communities_at_level(&self, level: usize) -> Result<String, JsValue>

    /// Get number of levels in hierarchy
    pub fn get_max_level(&self) -> usize

    /// Adaptive query (auto-selects best level)
    pub async fn query_adaptive(&self, query: &str, config_json: &str) -> Result<String, JsValue>

    /// Query at specific level
    pub async fn query_at_level(&self, query: &str, level: usize) -> Result<String, JsValue>

    /// Get community summary
    pub fn get_community_summary(&self, community_id: usize) -> Result<String, JsValue>

    /// Get all summaries as JSON
    pub fn get_all_summaries(&self) -> Result<String, JsValue>
}
```

### Task 3: Update Persistence

Add to save_to_storage():
```rust
// Save hierarchical communities if they exist
if let Some(ref communities) = self.hierarchical_communities {
    db.put("communities", "hierarchical", &communities).await?;
}
```

Add to load_from_storage():
```rust
// Load hierarchical communities if they exist
self.hierarchical_communities = db.get("communities", "hierarchical").await
    .ok(); // Optional, may not exist in old saves
```

### Task 4: Create Leptos UI Components

**New components needed:**

1. **HierarchyExplorer.rs**
   - Shows hierarchical levels (tabs for L0, L1, L2, ...)
   - Each level shows communities with entity counts
   - Click community → show entities and summary

2. **CommunityCard.rs**
   - Display single community
   - Shows: community_id, entity_count, avg_confidence, types
   - Expandable to show full entity list
   - Summary preview

3. **LevelSelector.rs**
   - Dropdown/buttons to select hierarchical level
   - Shows stats for each level (# communities, # entities)
   - Visual indicator of query complexity → suggested level

4. **AdaptiveQueryPanel.rs**
   - Enhanced query interface
   - Shows "Suggested Level" based on query analysis
   - Displays QueryAnalysis scores (keyword, length, entity)
   - Option to override and manually select level

### Task 5: Update main.rs UI

Add new tab for "Hierarchy" exploration:

```rust
enum Tab {
    Build,
    Explore,
    Query,
    Hierarchy,  // NEW!
    Settings,
}
```

---

## 🔧 Implementation Order

1. **Phase 1: Core Integration** (Current)
   - ✅ Enable leiden feature
   - [ ] Add HierarchicalCommunities field to GraphRAG
   - [ ] Add basic detection method (detect_communities)
   - [ ] Test compilation

2. **Phase 2: Query Integration**
   - [ ] Add query_adaptive method
   - [ ] Add query_at_level method
   - [ ] Integrate with existing query pipeline
   - [ ] Test adaptive routing

3. **Phase 3: Persistence**
   - [ ] Update save_to_storage for communities
   - [ ] Update load_from_storage for communities
   - [ ] Test save/load cycle

4. **Phase 4: UI Components**
   - [ ] Create HierarchyExplorer component
   - [ ] Create CommunityCard component
   - [ ] Create LevelSelector component
   - [ ] Create AdaptiveQueryPanel component
   - [ ] Integrate into main.rs

5. **Phase 5: Polish & Demo**
   - [ ] Add Symposium hierarchical demo
   - [ ] Performance testing
   - [ ] Documentation
   - [ ] Screenshots/video

---

## 📊 Expected UI Flow

1. **Build Tab** (existing)
   - User uploads documents
   - Extracts entities and relationships
   - **NEW**: Click "Detect Communities" button
   - Shows progress: "Detecting hierarchical communities..."
   - Complete: "Found 3 levels with 15/7/3 communities"

2. **Hierarchy Tab** (NEW!)
   - Level selector: [L0] [L1] [L2] [L3]
   - Grid of community cards for selected level
   - Each card shows:
     - Community ID
     - Entity count
     - Most common entity type
     - Summary preview
   - Click card → expand to show full entity list

3. **Query Tab** (enhanced)
   - Query input field
   - **NEW**: "Use Adaptive Routing" checkbox (default: ON)
   - **NEW**: Query Analysis panel showing:
     - Detected complexity: "Medium"
     - Suggested level: 1
     - Component scores (keyword: 0.2, length: 0.3, entity: 0.5)
   - **NEW**: Manual level override dropdown
   - Results now show which level was used

---

## 🎨 UI Mockup (ASCII)

```
┌─────────────────────────────────────────────────────┐
│  GraphRAG WASM - Hierarchical Knowledge Graph       │
├─────────────────────────────────────────────────────┤
│  [Build] [Explore] [Hierarchy] [Query] [Settings]   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Hierarchy Explorer                                 │
│  ┌───────────────────────────────────────────────┐ │
│  │  Level: [L0▼] [L1] [L2]    3 communities      │ │
│  └───────────────────────────────────────────────┘ │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐│
│  │ Community 0  │  │ Community 1  │  │ Community2││
│  │ 12 entities  │  │ 8 entities   │  │ 5 entities││
│  │ Types: 3     │  │ Types: 2     │  │ Types: 2  ││
│  │ Avg conf:0.9 │  │ Avg conf:0.8 │  │Avg conf:..││
│  │              │  │              │  │           ││
│  │ Summary:     │  │ Summary:     │  │Summary:   ││
│  │ AI, ML, DL,  │  │ NLP, Trans..│  │CV, Image. ││
│  │ Neural Net..│  │              │  │           ││
│  │ [Expand]     │  │ [Expand]     │  │ [Expand]  ││
│  └──────────────┘  └──────────────┘  └───────────┘│
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🔍 Testing Strategy

### Unit Tests
- Test detect_communities() with sample graph
- Test query_adaptive() with different query types
- Test level selection logic
- Test persistence (save/load hierarchical data)

### Integration Tests
- Full pipeline: Build → Detect → Query → Save → Load
- Test backward compatibility (old saves without communities)
- Test with Symposium.txt (philosophical text)

### Performance Tests
- Leiden algorithm on 100+ entity graphs
- UI responsiveness with large hierarchies
- Save/load time for hierarchical structures

---

## 📝 Documentation Needed

1. **User Guide**: "Using Hierarchical Communities in Browser"
2. **API Reference**: WASM methods for hierarchical queries
3. **Example**: Symposium hierarchical exploration demo
4. **Tutorial**: When to use Level 0 vs Level 2 queries

---

## 🚀 Next Steps

**IMMEDIATE**:
1. Add `hierarchical_communities` field to GraphRAG struct
2. Implement `detect_communities()` method
3. Test compilation and basic functionality

**SOON**:
4. Add query methods (adaptive + manual)
5. Update persistence layer
6. Create basic UI components

**LATER**:
7. Polish UI with animations and better visualizations
8. Add export functionality (download hierarchy as JSON)
9. Performance optimizations
