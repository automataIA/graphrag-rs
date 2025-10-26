# Graph Traversal Algorithms - Usage Guide

## Overview

The `GraphTraversal` module provides efficient graph traversal algorithms optimized for knowledge graph querying in GraphRAG systems. These algorithms support deterministic, ML-free query execution strategies.

## Features

- **6 Traversal Algorithms**: BFS, DFS, Ego-network, Multi-source BFS, All-paths finding, Query-focused subgraph
- **Configurable Parameters**: Max depth, path limits, edge weighting, relationship filtering
- **Optimized for Knowledge Graphs**: Entity-centric traversal with relationship strength consideration
- **Zero ML Dependencies**: Pure algorithmic approach suitable for resource-constrained environments

## Configuration

### In JSON Configuration

```json
{
  "graph": {
    "max_connections": 10,
    "similarity_threshold": 0.8,
    "extract_relationships": true,
    "relationship_confidence_threshold": 0.5,
    "traversal": {
      "max_depth": 3,
      "max_paths": 10,
      "use_edge_weights": true,
      "min_relationship_strength": 0.3
    }
  }
}
```

### In Code

```rust
use graphrag_core::graph::{GraphTraversal, TraversalConfig};

let config = TraversalConfig {
    max_depth: 3,
    max_paths: 10,
    use_edge_weights: true,
    min_relationship_strength: 0.3,
};

let traversal = GraphTraversal::new(config);
```

## Algorithms

### 1. Breadth-First Search (BFS)

**Purpose**: Explore entities level-by-level, finding shortest paths and nearest neighbors.

**Use Cases**:
- Finding shortest connections between entities
- Discovering immediate context around a query entity
- Community boundary detection

**Example**:
```rust
use graphrag_core::core::EntityId;

let source = EntityId::new("entity_123".to_string());
let result = traversal.bfs(&knowledge_graph, &source)?;

println!("Visited {} entities", result.entities.len());
println!("Found {} edges", result.edges.len());
println!("Max depth reached: {}", result.depth);
```

**Output Structure**:
```rust
TraversalResult {
    entities: Vec<EntityId>,    // Entities in BFS order
    edges: Vec<(EntityId, EntityId, f32)>,  // (source, target, strength)
    depth: usize,                // Maximum depth explored
    paths: Vec<Vec<EntityId>>,   // Empty for BFS
}
```

### 2. Depth-First Search (DFS)

**Purpose**: Deep exploration along paths, useful for finding connected components and cycles.

**Use Cases**:
- Exploring full connection chains
- Detecting strongly connected subgraphs
- Tracing causal or temporal chains

**Example**:
```rust
let source = EntityId::new("entity_123".to_string());
let result = traversal.dfs(&knowledge_graph, &source)?;

// DFS explores deeply before backtracking
for (idx, entity) in result.entities.iter().enumerate() {
    println!("Visit order {}: {:?}", idx, entity);
}
```

### 3. Ego-Network Extraction

**Purpose**: Extract the k-hop neighborhood around a central entity (ego).

**Use Cases**:
- Query context window extraction
- Local subgraph analysis
- Entity-centric document retrieval

**Example**:
```rust
let ego_id = EntityId::new("central_entity".to_string());
let k_hops = Some(2);  // 2-hop neighborhood
let result = traversal.ego_network(&knowledge_graph, &ego_id, k_hops)?;

println!("Ego-network contains {} entities", result.entities.len());
println!("Ego-network has {} edges", result.edges.len());

// The ego is always the first entity
assert_eq!(result.entities[0], ego_id);
```

**Visualization**:
```
      E4        E5
       \      /
    E2--EGO--E3
       /      \
      E1        E6--E7

k=1: {EGO, E1, E2, E3}
k=2: {EGO, E1, E2, E3, E4, E5, E6, E7}
```

### 4. Multi-Source BFS

**Purpose**: Simultaneously explore from multiple seed entities, finding their common neighbors.

**Use Cases**:
- Multi-entity query processing
- Finding mediating entities between query concepts
- Intersection analysis

**Example**:
```rust
let sources = vec![
    EntityId::new("entity_A".to_string()),
    EntityId::new("entity_B".to_string()),
    EntityId::new("entity_C".to_string()),
];

let result = traversal.multi_source_bfs(&knowledge_graph, &sources)?;

// Entities are visited in breadth-first order from all sources
println!("Explored {} entities from {} sources",
         result.entities.len(), sources.len());
```

### 5. Find All Paths

**Purpose**: Enumerate all paths between two entities up to max_depth.

**Use Cases**:
- Relationship chain discovery
- Alternative explanation paths
- Multi-hop reasoning support

**Example**:
```rust
let source = EntityId::new("entity_start".to_string());
let target = EntityId::new("entity_end".to_string());

let result = traversal.find_all_paths(&knowledge_graph, &source, &target)?;

println!("Found {} paths", result.paths.len());

for (idx, path) in result.paths.iter().enumerate() {
    println!("Path {}: {} entities", idx + 1, path.len());
    for entity_id in path {
        println!("  -> {:?}", entity_id);
    }
}
```

**Path Format**:
```
Path 1: [entity_start, entity_mid1, entity_end]
Path 2: [entity_start, entity_mid2, entity_mid3, entity_end]
```

### 6. Query-Focused Subgraph

**Purpose**: Extract a relevant subgraph around multiple seed entities with controlled expansion.

**Use Cases**:
- Multi-entity query context extraction
- Building query-specific knowledge bases
- Focused reasoning over graph regions

**Example**:
```rust
let seeds = vec![
    EntityId::new("concept_1".to_string()),
    EntityId::new("concept_2".to_string()),
    EntityId::new("concept_3".to_string()),
];

let expansion_hops = 2;  // Expand 2 hops from seeds

let result = traversal.query_focused_subgraph(
    &knowledge_graph,
    &seeds,
    expansion_hops
)?;

println!("Query subgraph: {} entities, {} edges",
         result.entities.len(),
         result.edges.len());
```

## Configuration Parameters

### `max_depth`
- **Type**: `usize`
- **Default**: `3`
- **Description**: Maximum depth for BFS/DFS traversals. Higher values explore further but increase computation.
- **Range**: 1-10 (recommended 2-4 for most use cases)

### `max_paths`
- **Type**: `usize`
- **Default**: `10`
- **Description**: Maximum number of paths to find in pathfinding algorithms. Limits result size.
- **Range**: 1-100 (use 5-20 for balanced performance)

### `use_edge_weights`
- **Type**: `bool`
- **Default**: `true`
- **Description**: Whether to consider relationship confidence/strength during traversal.
- **Effect**: When true, prioritizes high-confidence relationships in exploration order.

### `min_relationship_strength`
- **Type**: `f32`
- **Default**: `0.3`
- **Description**: Minimum relationship confidence threshold. Relationships below this are ignored.
- **Range**: 0.0-1.0 (use 0.2-0.5 depending on data quality)

## Performance Considerations

### Time Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| BFS | O(V + E) | O(V) |
| DFS | O(V + E) | O(V) |
| Ego-Network | O(V + E) | O(V) |
| Multi-source BFS | O(V + E) | O(V) |
| Find All Paths | O(V^d) | O(V * max_paths) |
| Query Subgraph | O(V + E) | O(V) |

Where:
- V = number of entities (vertices)
- E = number of relationships (edges)
- d = max_depth parameter

### Optimization Tips

1. **Limit Depth**: Keep `max_depth` ≤ 3 for large graphs (>10k entities)
2. **Filter Relationships**: Set `min_relationship_strength` to prune weak connections
3. **Use Ego-Networks**: For localized queries, ego-networks are faster than full traversals
4. **Batch Queries**: Process multiple queries in parallel when possible

## Integration with Query Pipeline

### Example: Query Processing Workflow

```rust
use graphrag_core::graph::{GraphTraversal, TraversalConfig};
use graphrag_core::core::EntityId;

// 1. Configure traversal
let config = TraversalConfig {
    max_depth: 2,
    max_paths: 5,
    use_edge_weights: true,
    min_relationship_strength: 0.4,
};
let traversal = GraphTraversal::new(config);

// 2. Extract seed entities from query
let query = "What is the relationship between AI and ethics?";
let seed_entities = vec![
    EntityId::new("AI".to_string()),
    EntityId::new("ethics".to_string()),
];

// 3. Extract query-focused subgraph
let subgraph = traversal.query_focused_subgraph(
    &knowledge_graph,
    &seed_entities,
    2  // expansion hops
)?;

// 4. Find paths between query entities
let paths = traversal.find_all_paths(
    &knowledge_graph,
    &seed_entities[0],
    &seed_entities[1]
)?;

// 5. Process results
println!("Subgraph: {} relevant entities", subgraph.entities.len());
println!("Found {} connection paths", paths.paths.len());

for path in paths.paths {
    // Use path for answer generation
    process_reasoning_path(path);
}
```

## Error Handling

All traversal methods return `Result<TraversalResult>`. Common errors:

```rust
match traversal.bfs(&graph, &entity_id) {
    Ok(result) => {
        // Process successful traversal
    }
    Err(e) => {
        match e {
            GraphRAGError::EntityNotFound { .. } => {
                // Entity doesn't exist in graph
            }
            GraphRAGError::InvalidInput { .. } => {
                // Invalid parameters (e.g., empty source list)
            }
            _ => {
                // Other errors
            }
        }
    }
}
```

## Best Practices

1. **Start Small**: Begin with `max_depth=2` and `max_paths=5`, adjust based on results
2. **Filter Early**: Use `min_relationship_strength` to reduce noise
3. **Choose Algorithm Wisely**:
   - Single entity query → Ego-Network
   - Multi-entity query → Query-Focused Subgraph
   - Shortest path → BFS
   - All connections → Find All Paths
4. **Monitor Performance**: Log traversal times and result sizes to tune parameters
5. **Combine Strategies**: Use BFS for initial exploration, then DFS for deep dives

## Examples by Use Case

### Use Case 1: Document Retrieval
```rust
// Extract 2-hop neighborhood around query entities
let ego_net = traversal.ego_network(&graph, &query_entity, Some(2))?;
// Use ego_net.entities to retrieve relevant documents
let docs = get_documents_for_entities(&ego_net.entities);
```

### Use Case 2: Relationship Explanation
```rust
// Find all paths between two entities
let paths = traversal.find_all_paths(&graph, &entity_a, &entity_b)?;
// Generate explanations from paths
for path in paths.paths {
    let explanation = generate_explanation_from_path(&graph, &path);
    println!("{}", explanation);
}
```

### Use Case 3: Multi-Concept Query
```rust
// Build query-specific subgraph
let seeds = extract_entities_from_query(query);
let subgraph = traversal.query_focused_subgraph(&graph, &seeds, 2)?;
// Rank entities by centrality in subgraph
let ranked = rank_entities_by_connections(&subgraph);
```

## Testing

Run the test suite:
```bash
cargo test --lib graph::traversal::tests
```

Available tests:
- `test_bfs_traversal`
- `test_dfs_traversal`
- `test_ego_network`
- `test_multi_source_bfs`
- `test_find_all_paths`
- `test_query_focused_subgraph`

## References

- Original implementation: `graphrag-core/src/graph/traversal.rs`
- Configuration: `graphrag-core/src/config/mod.rs` (lines 162-191)
- Core traits: `graphrag-core/src/core/mod.rs`

## Future Enhancements

Planned features:
- Weighted shortest paths (Dijkstra)
- Bidirectional search for faster pathfinding
- Approximate traversal for very large graphs
- Parallel traversal for multi-query scenarios
