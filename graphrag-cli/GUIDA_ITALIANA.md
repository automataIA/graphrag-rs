# 🚀 Guida Completa a GraphRAG-CLI (Italiano)

**GraphRAG-CLI** è un'interfaccia utente terminale (TUI) moderna e interattiva per operazioni GraphRAG, costruita con Ratatui.

## 📋 Indice

- [Installazione e Build](#installazione-e-build)
- [Avvio Rapido](#avvio-rapido)
- [Modalità di Utilizzo](#modalità-di-utilizzo)
- [Comandi Slash](#comandi-slash)
- [Gestione Workspace](#gestione-workspace)
- [Scorciatoie da Tastiera](#scorciatoie-da-tastiera)
- [Esempi Pratici](#esempi-pratici)
- [Risoluzione Problemi](#risoluzione-problemi)

---

## 🔧 Installazione e Build

### Prerequisiti
- Rust 1.70 o superiore
- Ollama installato e in esecuzione (per i modelli LLM)

### Compilazione

```bash
# Dalla directory principale del progetto
cargo build --release -p graphrag-cli

# Il binario sarà disponibile in:
# ./target/release/graphrag-cli
```

### Aggiungere al PATH (opzionale)

```bash
# Linux/macOS
export PATH="$PATH:/home/dio/graphrag-rs/target/release"

# Oppure creare un link simbolico
sudo ln -s /home/dio/graphrag-rs/target/release/graphrag-cli /usr/local/bin/graphrag-cli
```

---

## ⚡ Avvio Rapido

### 1. Avviare la TUI Interattiva

```bash
# Avvio base
./target/release/graphrag-cli

# Con file di configurazione
./target/release/graphrag-cli --config docs-example/symposium_config.toml

# Con workspace specifico
./target/release/graphrag-cli --workspace my_project

# Con logging debug
./target/release/graphrag-cli --debug
```

### 2. Primo Utilizzo - Setup Iniziale

1. **Avvia la TUI:**
   ```bash
   ./target/release/graphrag-cli
   ```

2. **Carica una configurazione:**
   Nella TUI, premi `Shift+Tab` per entrare in modalità comando, poi digita:
   ```
   /config docs-example/symposium_config.toml
   ```

3. **Carica un documento:**
   ```
   /load docs-example/platos_symposium.txt
   ```

4. **Esegui query:**
   Torna in modalità Query (premi `Shift+Tab`) e digita:
   ```
   What does Socrates say about love?
   ```

---

## 🎯 Modalità di Utilizzo

GraphRAG-CLI ha **due modalità principali**:

### 📝 Query Mode (Modalità Query)
- **Scopo:** Eseguire query sul knowledge graph
- **Come usarla:** Digita direttamente la tua domanda
- **Esempio:** `What are the main themes in the Symposium?`

### ⚙️ Command Mode (Modalità Comando)
- **Scopo:** Eseguire comandi di sistema (slash commands)
- **Come usarla:** Premi `Shift+Tab` per passare da Query Mode
- **Esempio:** `/config myfile.toml`

**Passare tra modalità:** `Shift+Tab`

---

## 🔀 Comandi Slash

I comandi slash sono disponibili solo in **Command Mode**.

### `/config <file>`
Carica un file di configurazione GraphRAG (TOML o JSON5)

```bash
# Esempio con TOML
/config docs-example/symposium_config.toml

# Esempio con percorso assoluto
/config /home/dio/graphrag-rs/my_config.toml

# Esempio con percorso relativo
/config ../config/templates/academic_research.toml
```

**Cosa fa:**
- Carica la configurazione LLM (Ollama, OpenAI, etc.)
- Configura embeddings e chunking strategy
- Inizializza il knowledge graph

---

### `/load <file>`
Carica e processa un documento nel knowledge graph

```bash
# Carica un file di testo
/load docs-example/platos_symposium.txt

# Carica un file markdown
/load ~/documents/research_paper.md

# Carica più documenti (esegui più volte)
/load document1.txt
/load document2.txt
/load document3.txt
```

**Cosa fa:**
1. Legge il documento
2. Divide in chunks (basato su config)
3. Estrae entità e relazioni
4. Costruisce il knowledge graph
5. Genera embeddings

**Tempo di elaborazione:** Dipende dalla dimensione del documento (10-60 secondi per 10KB)

---

### `/stats`
Mostra statistiche del knowledge graph

```bash
/stats
```

**Output:**
```
Knowledge Graph Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 Documents:      3
📄 Chunks:         127
👤 Entities:       45
🔗 Relationships:  89
📊 Graph Density:  0.087
```

---

### `/entities [filter]`
Lista le entità nel knowledge graph con filtro opzionale

```bash
# Lista tutte le entità
/entities

# Filtra per nome
/entities socrates

# Filtra per tipo
/entities PERSON

# Filtra per concetto
/entities love
```

**Output:**
```
Entities (filtered: "socrates"):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👤 Socrates [PERSON]
   Mentions: 34
   Confidence: 0.95
   Description: Greek philosopher, student of Plato...

💡 Socratic Method [CONCEPT]
   Mentions: 12
   Confidence: 0.89
   Description: Method of inquiry through questioning...
```

---

### `/workspace <name>`
Cambia workspace corrente

```bash
# Cambia workspace
/workspace my_project

# Cambia a workspace diverso
/workspace philosophy_research
```

**Cosa sono i workspace?**
- Directory separate per progetti diversi
- Ogni workspace ha il proprio knowledge graph
- Ogni workspace ha la propria storia delle query
- Permette di lavorare su più progetti simultaneamente

---

### `/help`
Mostra l'elenco di tutti i comandi disponibili

```bash
/help
```

---

## 🗂️ Gestione Workspace

I workspace permettono di organizzare progetti multipli separatamente.

### Comandi da Terminale

```bash
# Lista tutti i workspace
./target/release/graphrag-cli workspace list

# Crea nuovo workspace
./target/release/graphrag-cli workspace create philosophy_research

# Mostra informazioni su workspace
./target/release/graphrag-cli workspace info <workspace-id>

# Elimina workspace
./target/release/graphrag-cli workspace delete <workspace-id>
```

### Esempio: Creare e Usare un Workspace

```bash
# 1. Crea workspace
./target/release/graphrag-cli workspace create philosophy_research

# Output:
# ✅ Workspace created successfully!
#    Name: philosophy_research
#    ID:   a1b2c3d4-e5f6-7890-abcd-ef1234567890
#
# Use it with: graphrag-cli tui --workspace a1b2c3d4-e5f6-7890-abcd-ef1234567890

# 2. Avvia TUI con il workspace
./target/release/graphrag-cli --workspace a1b2c3d4-e5f6-7890-abcd-ef1234567890

# 3. Nella TUI, carica configurazione e documenti
# (modalità comando)
/config docs-example/symposium_config.toml
/load docs-example/platos_symposium.txt
```

### Percorsi dei Workspace

I workspace sono salvati in:
```
~/.local/share/graphrag-cli/workspaces/
└── <workspace-id>/
    ├── metadata.json          # Informazioni workspace
    ├── query_history.json     # Storia delle query
    ├── graph.db              # Knowledge graph (se persistente)
    └── embeddings/           # Vector embeddings
```

---

## ⌨️ Scorciatoie da Tastiera

### Navigazione Generale

| Tasto | Azione |
|-------|--------|
| `Shift+Tab` | Cambia modalità (Query ↔ Command) |
| `Ctrl+C` | Esci dall'applicazione |
| `?` | Mostra help overlay (in Query Mode) |
| `Esc` | Chiudi help overlay |

### Modalità Input

| Tasto | Azione |
|-------|--------|
| `Enter` | Esegui query/comando |
| `↑` / `↓` | Naviga storia query/comandi |
| `Ctrl+U` | Cancella linea input |
| `Ctrl+W` | Cancella parola precedente |

### Navigazione Risultati

| Tasto | Azione |
|-------|--------|
| `↑` / `↓` | Scroll risultati |
| `PgUp` / `PgDn` | Scroll veloce |
| `Home` / `End` | Inizio/fine risultati |

### Modalità Help (premendo `?`)

| Tasto | Azione |
|-------|--------|
| `Esc` | Chiudi help |
| `q` | Chiudi help |
| `↑` / `↓` | Scroll help |

---

## 💡 Esempi Pratici

### Esempio 1: Setup Filosofico (Simposio di Platone)

```bash
# 1. Avvia TUI
./target/release/graphrag-cli

# 2. In Command Mode (Shift+Tab)
/config docs-example/symposium_config.toml
/load docs-example/platos_symposium.txt

# 3. Aspetta elaborazione (30-60 secondi)

# 4. In Query Mode (Shift+Tab per tornare)
What does Socrates say about love?
Who are the main speakers in the Symposium?
Explain the myth of the androgyne
```

### Esempio 2: Analisi Multi-Documento

```bash
# 1. Avvia con configurazione
./target/release/graphrag-cli --config config/templates/academic_research.toml

# 2. Carica documenti multipli (Command Mode)
/load papers/paper1.txt
/load papers/paper2.txt
/load papers/paper3.txt

# 3. Verifica statistiche
/stats

# 4. Query analitiche (Query Mode)
Compare the methodologies used in the three papers
What are the common themes across all documents?
List all authors mentioned
```

### Esempio 3: Ricerca Tecnica

```bash
# 1. Crea workspace dedicato
./target/release/graphrag-cli workspace create rust_docs

# 2. Avvia con workspace
./target/release/graphrag-cli --workspace <workspace-id> \
  --config config/templates/technical_documentation.toml

# 3. Carica documentazione
/load rust-docs/ownership.md
/load rust-docs/concurrency.md
/load rust-docs/async.md

# 4. Query tecniche
Explain Rust's ownership system
How does async/await work in Rust?
What are the main concurrency primitives?
```

### Esempio 4: Esplorazione Entità

```bash
# 1. Dopo aver caricato documenti
/entities

# 2. Filtra per tipo
/entities PERSON
/entities CONCEPT
/entities LOCATION

# 3. Cerca entità specifiche
/entities socrates
/entities love
/entities athens
```

---

## 🔍 Risoluzione Problemi

### Problema: "Configuration not loaded"

**Sintomi:** Non riesci a caricare documenti o eseguire query

**Soluzione:**
```bash
# 1. Assicurati di caricare prima la configurazione
/config path/to/config.toml

# 2. Verifica che il file esista
ls -la path/to/config.toml

# 3. Verifica sintassi TOML
cat path/to/config.toml
```

---

### Problema: "Failed to connect to Ollama"

**Sintomi:** Errori durante caricamento documenti o query

**Soluzione:**
```bash
# 1. Verifica che Ollama sia in esecuzione
ollama list

# 2. Avvia Ollama se necessario
ollama serve

# 3. Testa connessione
curl http://localhost:11434/api/tags

# 4. Verifica modello configurato sia disponibile
ollama list | grep qwen3
```

---

### Problema: "Document processing is slow"

**Sintomi:** Caricamento documenti impiega molto tempo

**Cause possibili:**
1. Documento molto grande
2. Chunking strategy aggressivo
3. Estrazione entità complessa
4. Modello LLM lento

**Soluzione:**
```toml
# Ottimizza config.toml

[text.chunking]
method = "semantic"  # Più veloce di "hierarchical"
max_tokens = 500     # Ridurre per chunks più veloci

[entity.extraction]
use_gleaning = false # Disabilita per velocità
max_iterations = 1   # Riduce iterazioni
```

---

### Problema: "TUI is corrupted/garbled"

**Sintomi:** Caratteri strani, layout corrotto

**Soluzione:**
```bash
# 1. Resetta terminale
reset

# 2. Verifica TERM variable
echo $TERM
# Dovrebbe essere: xterm-256color o simile

# 3. Imposta se necessario
export TERM=xterm-256color

# 4. Riavvia TUI
./target/release/graphrag-cli
```

---

### Problema: "Workspace not found"

**Sintomi:** Errore al caricamento workspace

**Soluzione:**
```bash
# 1. Lista workspace disponibili
./target/release/graphrag-cli workspace list

# 2. Crea nuovo workspace se necessario
./target/release/graphrag-cli workspace create my_project

# 3. Usa ID corretto
./target/release/graphrag-cli --workspace <id-corretto>
```

---

### Problema: "Out of memory"

**Sintomi:** Crash durante elaborazione documenti grandi

**Soluzione:**
```toml
# Riduci memoria usata in config.toml

[text.chunking]
max_tokens = 300  # Riduce dimensione chunks
overlap = 20      # Riduce overlap

[embeddings]
batch_size = 16   # Riduce batch per embeddings

[caching]
enabled = true    # Abilita caching per risparmiare RAM
max_size_mb = 100 # Limita dimensione cache
```

---

## 📊 Configurazioni Consigliate

### Per Documenti Piccoli (<100KB)

```toml
[text.chunking]
method = "semantic"
max_tokens = 500
overlap = 50

[entity.extraction]
use_gleaning = true
max_iterations = 3
```

### Per Documenti Grandi (>1MB)

```toml
[text.chunking]
method = "simple"
max_tokens = 300
overlap = 20

[entity.extraction]
use_gleaning = false
max_iterations = 1

[caching]
enabled = true
max_size_mb = 500
```

### Per Performance Massime

```toml
[text.chunking]
method = "simple"
max_tokens = 400

[entity.extraction]
use_gleaning = false

[async_processing]
enabled = true
max_concurrent_operations = 8

[caching]
enabled = true
cache_embeddings = true
```

---

## 📝 Esempi di Configurazione

### Template Base (TOML)

```toml
# my_config.toml

[llm]
provider = "ollama"
model = "qwen3:8b"
temperature = 0.7
max_tokens = 2000

[embeddings]
provider = "ollama"
model = "nomic-embed-text"
dimensions = 768

[text.chunking]
method = "semantic"
max_tokens = 500
overlap = 50

[entity.extraction]
use_gleaning = true
max_iterations = 2

[graph]
enable_pagerank = true
pagerank_damping = 0.85

[storage]
type = "memory"
```

### Template Avanzato (JSON5)

```json5
// my_config.json5
{
  llm: {
    provider: "ollama",
    model: "qwen3:8b",
    temperature: 0.7,
    max_tokens: 2000,
  },

  embeddings: {
    provider: "ollama",
    model: "nomic-embed-text",
    dimensions: 768,
  },

  text: {
    chunking: {
      method: "hierarchical",
      max_tokens: 500,
      overlap: 50,
    },
  },

  entity: {
    extraction: {
      use_gleaning: true,
      max_iterations: 3,
      confidence_threshold: 0.7,
    },
  },

  graph: {
    enable_pagerank: true,
    enable_leiden: true,
  },

  caching: {
    enabled: true,
    max_size_mb: 200,
    cache_embeddings: true,
  },
}
```

---

## 🎓 Tutorial Completo: Primo Progetto

Segui questo tutorial passo-passo per il tuo primo progetto GraphRAG:

### Step 1: Preparazione

```bash
# 1. Crea directory progetto
mkdir -p ~/graphrag-projects/philosophy
cd ~/graphrag-projects/philosophy

# 2. Crea file configurazione
cat > config.toml << 'EOF'
[llm]
provider = "ollama"
model = "qwen3:8b"
temperature = 0.7

[embeddings]
provider = "ollama"
model = "nomic-embed-text"

[text.chunking]
method = "semantic"
max_tokens = 500

[entity.extraction]
use_gleaning = true
max_iterations = 2
EOF

# 3. Prepara documenti
mkdir documents
# Copia i tuoi documenti in ./documents/
```

### Step 2: Avvio e Configurazione

```bash
# 1. Avvia GraphRAG-CLI
/home/dio/graphrag-rs/target/release/graphrag-cli

# 2. Nella TUI, passa a Command Mode
# Premi: Shift+Tab

# 3. Carica configurazione
/config ~/graphrag-projects/philosophy/config.toml

# 4. Verifica successo (dovrebbe apparire messaggio di conferma)
```

### Step 3: Caricamento Documenti

```bash
# In Command Mode, carica documenti uno alla volta
/load ~/graphrag-projects/philosophy/documents/doc1.txt

# Aspetta completamento (30-60 secondi)

/load ~/graphrag-projects/philosophy/documents/doc2.txt

# Ripeti per tutti i documenti
```

### Step 4: Esplorazione

```bash
# 1. Verifica statistiche
/stats

# 2. Esplora entità
/entities

# 3. Filtra entità per tipo
/entities PERSON
/entities CONCEPT
```

### Step 5: Query

```bash
# Passa a Query Mode
# Premi: Shift+Tab

# Esegui query analitiche
What are the main philosophical concepts discussed?

# Query su entità specifiche
Tell me about Socrates and his ideas

# Query comparative
Compare Plato's and Aristotle's views on reality

# Query tematiche
What are the main themes in these documents?
```

---

## 🚀 Tips & Tricks

### 1. Usa Storia Query

- Premi `↑` / `↓` per navigare query precedenti
- Modifica e re-esegui query velocemente
- La storia è persistente per workspace

### 2. Organizza per Progetti

```bash
# Crea workspace per ogni progetto
graphrag-cli workspace create project_A
graphrag-cli workspace create project_B
graphrag-cli workspace create project_C

# Passa rapidamente tra progetti
graphrag-cli --workspace <project_A_id>
```

### 3. Ottimizza Performance

```toml
# Abilita tutte le features di performance
[async_processing]
enabled = true
max_concurrent_operations = 8

[caching]
enabled = true
cache_embeddings = true

[parallel]
max_workers = 4
```

### 4. Debugging

```bash
# Avvia con debug logging
graphrag-cli --debug

# I log sono salvati in:
# Linux: ~/.local/share/graphrag-cli/logs/graphrag-cli.log
# Visualizza log in tempo reale:
tail -f ~/.local/share/graphrag-cli/logs/graphrag-cli.log
```

### 5. Batch Processing

```bash
# Usa script per caricare molti documenti
for file in documents/*.txt; do
  echo "/load $file"
done > commands.txt

# Poi esegui manualmente nella TUI
# (o integra con automazione futura)
```

---

## 📚 Risorse Aggiuntive

### Documentazione

- [GraphRAG-Core README](../graphrag-core/README.md)
- [Configuration Guide](../CONFIGURATION_GUIDE.md)
- [Architecture Documentation](../ARCHITECTURE.md)

### Template di Configurazione

- `config/templates/academic_research.toml`
- `config/templates/technical_documentation.toml`
- `config/templates/narrative_fiction.toml`

### Esempi

- `examples/multi_document_pipeline.rs`
- `examples/symposium_real_search.rs`
- `docs-example/symposium_config.toml`

---

## 🤝 Contribuire

Hai trovato bug o vuoi contribuire? Apri una issue o pull request su:
https://github.com/yourusername/graphrag-rs

---

## 📄 Licenza

MIT License - Vedi [LICENSE](../LICENSE) per dettagli

---

## 🎯 Quick Reference Card

```
╔════════════════════════════════════════════════════════════╗
║              GraphRAG-CLI Quick Reference                  ║
╠════════════════════════════════════════════════════════════╣
║ AVVIO                                                      ║
║   graphrag-cli                      Avvia TUI              ║
║   graphrag-cli --config FILE        Con configurazione     ║
║   graphrag-cli --workspace ID       Con workspace          ║
║                                                            ║
║ MODALITÀ                                                   ║
║   Shift+Tab    Cambia Query ↔ Command Mode                ║
║   ?            Mostra help (Query Mode)                    ║
║   Ctrl+C       Esci                                        ║
║                                                            ║
║ COMANDI (Command Mode)                                     ║
║   /config FILE      Carica configurazione                  ║
║   /load FILE        Carica documento                       ║
║   /stats            Mostra statistiche                     ║
║   /entities [FILT]  Lista entità (con filtro opzionale)    ║
║   /workspace NAME   Cambia workspace                       ║
║   /help             Mostra tutti i comandi                 ║
║                                                            ║
║ WORKSPACE (da terminale)                                   ║
║   workspace list         Lista workspace                   ║
║   workspace create NAME  Crea workspace                    ║
║   workspace info ID      Info workspace                    ║
║   workspace delete ID    Elimina workspace                 ║
║                                                            ║
║ NAVIGAZIONE                                                ║
║   ↑/↓         Naviga storia / scroll risultati             ║
║   PgUp/PgDn   Scroll veloce                                ║
║   Home/End    Inizio/fine risultati                        ║
║   Ctrl+U      Cancella linea                               ║
║   Ctrl+W      Cancella parola                              ║
╚════════════════════════════════════════════════════════════╝
```

---

**Buon GraphRAG-ing! 🚀✨**
