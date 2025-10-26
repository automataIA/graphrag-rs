# GraphRAG CLI - Slash Commands

## ✅ Implementati

La TUI ora supporta comandi slash per eseguire operazioni direttamente dall'input query.

### Comandi Disponibili

#### `/config <file>`
Carica un file di configurazione GraphRAG (TOML).

**Esempio:**
```
/config docs-example/symposium_config.toml
/config my_config.toml
```

**Funzionalità:**
- Carica e valida la configurazione TOML
- Inizializza GraphRAG con le nuove impostazioni
- Mostra riepilogo delle configurazioni chiave
- Abilita enhancements (LightRAG, Leiden, ecc.)
- Aggiorna il pannello info con lo stato

**Output:**
- Chunk size e overlap
- Confidence threshold
- Similarity threshold
- Elenco degli enhancements abilitati

---

#### `/load <file>`
Carica un documento nel knowledge graph.

**Esempio:**
```
/load ~/documents/article.txt
/load /home/dio/graphrag-rs/docs-example/Symposium.txt
```

**Funzionalità:**
- Carica e processa il documento
- Estrae entità e chunking
- Salva nel workspace corrente (se specificato)
- Mostra statistiche (entità, relationships, chunks)

---

#### `/stats`
Mostra statistiche del knowledge graph nel workspace corrente.

**Output:**
- Numero di entità
- Numero di relationships
- Numero di chunks
- Numero di documenti

---

#### `/entities [filter]`
Lista le entità nel knowledge graph.

**Esempi:**
```
/entities              # Lista tutte le entità (max 50)
/entities socrates     # Filtra per nome contenente "socrates"
/entities PERSON       # Filtra per tipo "PERSON"
```

**Output:**
- Nome entità
- Tipo
- Confidence score (%)

---

#### `/workspace <name>`
Cambia workspace attivo.

**Esempio:**
```
/workspace symposium
/workspace my_project
```

**Funzionalità:**
- Switcha al nuovo workspace
- Ricarica GraphRAG se config presente
- Permette di organizzare progetti separati

---

#### `/help`
Mostra aiuto completo sui comandi slash.

---

## Come Usare

1. **Avvia la TUI:**
   ```bash
   graphrag-cli tui --workspace my_project --config config.toml
   ```

2. **Modalità Input (NUOVO!):**
   - **📝 Query Mode** (bordo verde): Per domande in linguaggio naturale
   - **⚙ Command Mode** (bordo giallo): Per comandi slash
   - **Shift+Tab**: Cambia tra le due modalità

3. **Carica una configurazione:**
   - Passa a Command Mode con `Shift+Tab`
   - Digita `/config my_config.toml`
   - Premi Enter

4. **Carica un documento:**
   - In Command Mode, digita `/load /path/to/document.txt`
   - Premi Enter

5. **Visualizza statistiche:**
   - In Command Mode, digita `/stats`
   - Guarda anche l'Info Panel a destra per statistiche in tempo reale

6. **Esplora entità:**
   - Digita `/entities` o `/entities <filter>`
   - Premi Enter

7. **Cambia workspace:**
   - Digita `/workspace <name>`
   - Premi Enter

8. **Query normali:**
   - Passa a Query Mode con `Shift+Tab`
   - Qualsiasi input viene eseguito come query GraphRAG
   - Esempio: `What is the main topic of the document?`

---

## Shortcut Utili

### Globali
- `?` - Mostra help completo
- `Tab` - Cambia tra pannelli (Input/Results)
- `Shift+Tab` - **Cambia modalità input (Query/Command)** quando l'input è attivo
- `Ctrl+D` - Pulisci input
- `q` / `Ctrl+C` - Esci

### Navigazione
- `j` / `↓` - Scroll giù nei risultati
- `k` / `↑` - Scroll su nei risultati
- `Ctrl+D` - Pagina giù
- `Ctrl+U` - Pagina su
- `Home` - Vai all'inizio
- `End` - Vai alla fine

---

## Architettura

### File Principali

- **`src/commands.rs`**: Definizione e parsing comandi slash
- **`src/app.rs`**: Gestione comandi nella TUI
  - `handle_slash_command()`: Dispatcher comandi
  - Integrazione con status bar e results viewer

### Flusso Esecuzione

1. User digita comando (es. `/load file.txt`)
2. `QueryInput` cattura Enter
3. `App::execute_query()` chiama `SlashCommand::parse()`
4. Se comando slash → `App::handle_slash_command()`
5. Esegue handler specifico (`execute_load`, `execute_stats`, ecc.)
6. Mostra risultato in `ResultsViewer`
7. Aggiorna `StatusBar` con feedback

### Estensibilità

Per aggiungere nuovi comandi:

1. Aggiungi variante a `SlashCommand` enum in `commands.rs`
2. Implementa parsing in `SlashCommand::parse()`
3. Crea handler `execute_<name>()` in `commands.rs`
4. Aggiungi match arm in `App::handle_slash_command()` in `app.rs`
5. Aggiorna `help_text()` in `commands.rs`
6. Aggiorna TUI help overlay in `app.rs`

---

## Testing

### Test Unitari
```bash
cargo test --package graphrag_cli -- commands::tests
```

### Test Funzionali

1. **Test /load:**
   ```bash
   # Avvia TUI
   ./target/release/graphrag_cli tui --workspace test --config config.toml

   # Nella TUI:
   /load docs-example/Symposium.txt
   ```

2. **Test /stats:**
   ```bash
   # Dopo aver caricato un documento:
   /stats
   ```

3. **Test /entities:**
   ```bash
   /entities
   /entities socrates
   /entities PERSON
   ```

4. **Test /workspace:**
   ```bash
   /workspace new_project
   /stats  # Dovrebbe dire "no graph found"
   ```

---

## Known Limitations

1. **0 Relationships**: L'entity extraction al momento non genera relationships tra entità
   - Issue nel backend di graphrag-core
   - Da investigare nel modulo entity extraction

2. **TUI richiede terminale reale**: Non può essere testata in ambienti senza TTY
   - Usare Windows Terminal, gnome-terminal, etc.

3. **No graph loading dal JSON**: Il graph salvato non può essere ricaricato
   - `KnowledgeGraph` non ha `load_from_json()`
   - Necessario implementare deserialization

---

## Prossimi Passi

### High Priority
- [ ] Implementare `KnowledgeGraph::load_from_json()`
- [ ] Fix relationship extraction (0 relationships issue)
- [ ] Aggiungere `/query <question>` command

### Medium Priority
- [ ] `/history` - Mostra query history
- [ ] `/clear` - Pulisce knowledge graph
- [ ] `/export <format>` - Esporta graph (GraphML, Cypher, etc.)

### Low Priority
- [ ] Tab completion per comandi slash
- [ ] Command history (freccia su/giù)
- [ ] Syntax highlighting per comandi
