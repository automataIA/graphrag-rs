# GraphRAG WASM UI/UX Design Documentation

## Overview

This document details the complete UI/UX design for the GraphRAG WASM application, including the document ingestion pipeline, graph visualization, and query interface.

## Design Philosophy

### Core Principles

1. **Progressive Disclosure**: Show complexity only when needed
2. **Clear Mental Model**: Users understand the pipeline: Document → Graph → Query
3. **Immediate Feedback**: Every action has visual confirmation
4. **Graceful Degradation**: Works across all device sizes and capabilities
5. **Accessibility First**: WCAG 2.1 AA compliant from the start

## Information Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAG WASM Application                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Header: Brand + Technology Badges                         │
│                                                             │
│  ┌─────────────┬─────────────┬─────────────┐             │
│  │ 1. Build    │ 2. Explore  │ 3. Query    │ Tab Nav     │
│  │ Graph       │ Graph       │ Graph       │             │
│  └─────────────┴─────────────┴─────────────┘             │
│                                                             │
│  ┌───────────────────────────────────────┐                │
│  │         Active Tab Content            │                │
│  │                                       │                │
│  │  TAB 1: Document Ingestion           │                │
│  │  ├─ Add Documents                    │                │
│  │  │  ├─ File Upload                   │                │
│  │  │  └─ Text Paste                    │                │
│  │  ├─ Document Library                 │                │
│  │  │  └─ List with Remove Actions      │                │
│  │  └─ Build Pipeline                   │                │
│  │     ├─ Progress Visualization        │                │
│  │     └─ Build Button                  │                │
│  │                                       │                │
│  │  TAB 2: Graph Exploration            │                │
│  │  ├─ Statistics Grid                  │                │
│  │  │  ├─ Documents, Chunks, Entities   │                │
│  │  │  ├─ Relationships, Embeddings     │                │
│  │  │  └─ Graph Density                 │                │
│  │  ├─ Health Indicators                │                │
│  │  │  ├─ Coverage                      │                │
│  │  │  ├─ Entity Linking                │                │
│  │  │  └─ Embedding Quality             │                │
│  │  └─ System Configuration             │                │
│  │                                       │                │
│  │  TAB 3: Query Interface              │                │
│  │  ├─ Query Input                      │                │
│  │  ├─ Search Button                    │                │
│  │  └─ Results Display                  │                │
│  │     └─ Entity Cards with Context     │                │
│  └───────────────────────────────────────┘                │
│                                                             │
│  Footer: Credits + Version                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## User Flow

### Primary User Journey

```
1. ARRIVE AT APP
   ↓
   User sees: Clear 3-step process (Build → Explore → Query)
   Default: Build tab active, empty state

2. ADD DOCUMENTS
   ↓
   Options:
   a) Upload files (.txt, .md, .pdf)
   b) Paste text directly
   ↓
   Result: Documents appear in library with preview

3. BUILD GRAPH
   ↓
   Click "Build Knowledge Graph" button
   ↓
   Watch progress:
   - Chunking Documents (40% weight)
   - Extracting Entities (30% weight)
   - Computing Embeddings (20% weight)
   - Building Search Index (10% weight)
   ↓
   Success state: "Graph Ready" message

4. EXPLORE GRAPH (Optional)
   ↓
   Switch to Explore tab
   ↓
   View:
   - 6 stat cards with key metrics
   - 3 health indicators
   - System configuration details

5. QUERY GRAPH
   ↓
   Switch to Query tab
   ↓
   Enter question
   ↓
   Receive results with:
   - Entity matches with relevance scores
   - Source attribution
   - Context snippets
```

## Component Specifications

### 1. Tab Navigation

**Purpose**: Guide users through the linear workflow

**Accessibility**:
- `role="tablist"` on container
- `role="tab"` on each button
- `aria-selected` indicates active tab
- `aria-controls` links to panel
- Keyboard navigation (Arrow keys + Tab)

**Visual States**:
- **Active**: Purple background, white text, shadow
- **Inactive**: Semi-transparent slate, hover effect
- **With Badge**: Document count or status indicator

**Responsive Behavior**:
- Mobile: Stack vertically
- Tablet+: Horizontal row

### 2. Build Tab Components

#### A. Document Input Section

**File Upload**:
- Input: `type="file" multiple accept=".txt,.md,.pdf"`
- Visual: Styled file button with hover state
- Feedback: Files immediately added to library
- Error: Invalid file types show warning

**Text Paste**:
- Name input (optional): 1-line text field
- Content textarea: 8 rows, monospace font
- Add button: Disabled when empty
- Clear: Fields reset after successful add

**Touch Targets**: All buttons minimum 44x44px

#### B. Document Library

**Empty State**:
- Icon: Large empty mailbox emoji
- Message: "No documents yet"
- Guidance: "Upload files or paste text to get started"

**Populated State**:
- Max height: 384px (24rem) with scroll
- Each document card shows:
  - Name (truncated)
  - Preview (150 chars max)
  - Size in bytes
  - Relative timestamp
  - Remove button (trash icon)

**Interaction**:
- Hover: Border color changes
- Remove: Confirm via immediate removal (no modal)
- Focus: Keyboard accessible remove buttons

#### C. Build Progress

**Idle State**:
- Instructional text
- Build button enabled when docs > 0

**Building State**:
- Progress bar: Gradient purple to pink
- Stage label: Icon + text
- Percentage: Right-aligned
- Detail text: Current/total items
- Button disabled

**Ready State**:
- Success card: Green background
- Checkmark icon
- Message: "Knowledge Graph Ready!"
- Guidance: "You can now query..."
- Button: "Rebuild?" option

**Error State**:
- Error card: Red background
- Warning icon
- Error message display
- Button: "Retry Build"

**Progress Stages**:
1. Chunking: 200ms per document
2. Extracting: 100ms per chunk (5 chunks/doc)
3. Embedding: 50ms per entity (3 entities/chunk)
4. Indexing: 100ms × 10 steps

### 3. Explore Tab Components

#### A. Statistics Grid

**Layout**: CSS Grid
- Mobile: 1 column
- Tablet: 2 columns
- Desktop: 3 columns

**Stat Cards**:
- Icon (3xl size)
- Value (2xl, bold)
- Label (sm, muted)
- Color-coded borders
- Hover: Scale up 5%

**Colors by Type**:
- Documents: Blue
- Chunks: Green
- Entities: Yellow
- Relationships: Purple
- Embeddings: Pink
- Density: Indigo

#### B. Health Indicators

**Three Metrics**:
1. Coverage: % of documents processed
2. Entity Linking: Connection strength
3. Embedding Quality: Vector representation quality

**Visual**:
- Horizontal progress bars
- Color coding:
  - Green: 80-100% (good)
  - Yellow: 50-79% (warning)
  - Red: 0-49% (error)
- Description text below

#### C. System Configuration

**Two-Column Grid**:
- Tokenizer info
- Vocabulary size
- Embedding method
- LLM details
- Vector search engine
- Storage backend

**Visual**: Dark cards with colored text per technology

### 4. Query Tab Components

#### A. Query Input

**Warning Banner** (when graph not ready):
- Yellow background
- Warning icon
- Clear message

**Input Field**:
- Label: "Enter your query"
- Placeholder: "What would you like to know?"
- Disabled when graph not ready
- Focus ring: Purple

**Submit Button**:
- Full width
- Purple gradient
- Loading state: Spinner + "Searching..."
- Disabled when no graph or already loading

#### B. Results Display

**Loading State**:
- Centered spinner
- Purple gradient border animation
- Search icon in center

**Results Format**:
- Monospace font
- Query echo
- Statistics summary
- Entity results with:
  - Entity name
  - Relevance score
  - Context snippet
  - Source attribution

## Design System

### Colors

**Primary Palette**:
```css
--purple-600: #9333ea  /* Primary actions */
--purple-700: #7e22ce  /* Hover states */
--slate-900: #0f172a   /* Dark backgrounds */
--slate-800: #1e293b   /* Cards */
--slate-700: #334155   /* Borders */
--slate-400: #94a3b8   /* Muted text */
--slate-300: #cbd5e1   /* Body text */
```

**Semantic Colors**:
```css
--green-500: #22c55e   /* Success */
--yellow-500: #eab308  /* Warning */
--red-500: #ef4444     /* Error */
--blue-500: #3b82f6    /* Info */
```

### Typography

**Font Families**:
- Sans: System font stack
- Mono: `font-mono` for code/data

**Scale**:
- xs: 0.75rem (12px)
- sm: 0.875rem (14px)
- base: 1rem (16px)
- xl: 1.25rem (20px)
- 2xl: 1.5rem (24px)
- 5xl: 3rem (48px)

**Line Heights**:
- Tight: 1.25
- Normal: 1.5
- Relaxed: 1.75

### Spacing

**8px Grid System**:
- 2: 0.5rem (8px)
- 3: 0.75rem (12px)
- 4: 1rem (16px)
- 6: 1.5rem (24px)
- 8: 2rem (32px)
- 12: 3rem (48px)

### Borders & Shadows

**Border Radius**:
- Default: 0.5rem (8px)
- Full: 9999px (pills)

**Shadows**:
- Default: `shadow-xl`
- Focus: `ring-2 ring-purple-500 ring-offset-2`

### Animation

**Timing**:
- Fast: 200ms
- Normal: 300ms
- Slow: 500ms

**Easing**: ease-out

**Animated Elements**:
- Progress bars: width transition
- Spinners: 360° rotation
- Hover states: scale, color
- Tab switches: fade

## Responsive Breakpoints

### Mobile (320px - 767px)

- Single column layout
- Stacked tabs
- Full-width buttons
- Reduced padding (px-4)
- Smaller text where appropriate

### Tablet (768px - 1023px)

- Two-column grids
- Horizontal tabs with wrap
- Moderate padding (px-6)
- Standard text sizes

### Desktop (1024px+)

- Three-column grids
- Full horizontal tabs
- Max width: 1280px (7xl)
- Generous padding (px-8)
- Optimal line lengths

## Accessibility Features

### Keyboard Navigation

**Tab Order**:
1. Skip to main content (invisible)
2. Tab navigation buttons
3. Active tab content (sequential)
4. Footer links

**Shortcuts**:
- Tab: Next focusable element
- Shift+Tab: Previous element
- Enter/Space: Activate button
- Arrow keys: Switch tabs (when focused)

### Screen Reader Support

**ARIA Labels**:
- `aria-label` on icon-only buttons
- `aria-live="polite"` on progress updates
- `aria-describedby` for error messages
- `aria-invalid` on form errors
- `aria-selected` on tabs

**Semantic HTML**:
- `<nav role="tablist">` for tabs
- `<main>` for content area
- `<header>` and `<footer>` landmarks
- `<button>` for interactive elements
- `<form>` for query input

### Visual Accessibility

**Contrast Ratios** (WCAG AA):
- Normal text: 4.5:1 minimum
- Large text: 3:1 minimum
- Interactive elements: 3:1 minimum

**Focus Indicators**:
- Visible on all interactive elements
- Purple ring (2px)
- Offset from element (2px)

**Color Independence**:
- Status shown with icons AND color
- Disabled state: opacity + cursor
- Progress: percentage text + bar

## Performance Considerations

### Initial Load

- Minimal CSS (Tailwind CDN)
- Small WASM bundle
- Lazy load icons
- No large assets

### Runtime

- Reactive updates (Leptos signals)
- Virtual scrolling for large document lists
- Debounced file reads
- Chunked graph building

### Browser Storage

- IndexedDB for document persistence
- Cache API for models
- LocalStorage for preferences

## Future Enhancements

### Phase 2 Features

1. **Drag-and-drop upload zone**
2. **Document preview modal**
3. **Graph visualization (D3.js/Canvas)**
4. **Export results (JSON/CSV)**
5. **Dark/light theme toggle**
6. **Query history**
7. **Advanced filters**
8. **Batch operations**

### Advanced Features

1. **Real-time collaboration**
2. **Cloud sync**
3. **Custom entity types**
4. **Graph editing**
5. **A/B testing metrics**

## Testing Checklist

### Functional Tests

- [ ] File upload works (txt, md, pdf)
- [ ] Text paste adds documents
- [ ] Remove document updates list
- [ ] Build progresses through all stages
- [ ] Tab switching preserves state
- [ ] Query requires built graph
- [ ] Results display correctly

### Accessibility Tests

- [ ] Keyboard navigation works
- [ ] Screen reader announces states
- [ ] Focus visible on all elements
- [ ] Color contrast meets WCAG AA
- [ ] ARIA attributes correct
- [ ] Alt text present

### Responsive Tests

- [ ] Mobile (iPhone SE, 375px)
- [ ] Tablet (iPad, 768px)
- [ ] Desktop (1920px)
- [ ] Touch targets adequate
- [ ] Text readable at all sizes

### Browser Tests

- [ ] Chrome/Edge (Chromium)
- [ ] Firefox
- [ ] Safari
- [ ] Mobile Safari
- [ ] WebAssembly support check

## Implementation Notes

### Leptos 0.8 Patterns

**Signals**:
```rust
let (state, set_state) = signal(initial_value);
```

**Computed Values**:
```rust
let derived = move || some_signal.get() * 2;
```

**Effects**:
```rust
Effect::new(move |_| {
    // Runs when dependencies change
});
```

### WASM Constraints

- No direct file system access
- FileReader API for uploads
- IndexedDB for persistence
- Web Workers for heavy computation
- SharedArrayBuffer for parallelism

### Build Pipeline

```bash
# Development
trunk serve

# Production
trunk build --release

# With optimizations
trunk build --release --features hydrate
```

## Design Decisions Rationale

### Why Three Tabs?

- **Linear workflow**: Users naturally progress through stages
- **Reduced cognitive load**: One task at a time
- **Clear progress**: Visual indication of completion
- **Mobile friendly**: Easy to understand on small screens

### Why Prominent Progress Bar?

- **Transparency**: Users see what's happening
- **Trust building**: Professional appearance
- **Time estimation**: Users know how long to wait
- **Educational**: Shows the GraphRAG pipeline

### Why Empty States?

- **Guidance**: New users know what to do
- **Confidence**: Clear that nothing is broken
- **Motivation**: Encourages taking action
- **Professional**: Attention to detail

### Why Stat Cards?

- **Scannable**: Quick overview of graph health
- **Engaging**: Visual interest with icons/colors
- **Informative**: Key metrics at a glance
- **Expandable**: Room for more metrics later

## Conclusion

This UI/UX design creates an intuitive, accessible, and beautiful interface for GraphRAG WASM. The three-tab workflow guides users naturally from document ingestion through graph exploration to querying, with clear visual feedback at every step.

The design prioritizes:
- **Clarity**: Users always know what to do next
- **Feedback**: Every action has a response
- **Accessibility**: Works for everyone
- **Performance**: Fast and responsive
- **Delight**: Smooth animations and polished details

Built with modern web standards and Leptos 0.8's reactive system, this interface showcases the power of GraphRAG technology while remaining approachable for all users.
