# Corpus italiano per il bench

- `pinocchio.txt`: Carlo Collodi, *Le avventure di Pinocchio: Storia di un burattino* (ed. Bemporad, 1902).
  Project Gutenberg eBook #52484, <https://www.gutenberg.org/ebooks/52484>
  (testo: <https://www.gutenberg.org/cache/epub/52484/pg52484.txt>). Lingua: italiano. Autore morto nel 1890: pubblico dominio.
  Rimossi header/footer di licenza Gutenberg, frontespizio, indice e nota del trascrittore: restano i 36 capitoli (da "I." a "FINE.").
- `gold.json`: 19 domande in italiano con keyword attese (vedi `BASELINE.md` per l'uso e i risultati).
  `expected_keywords` e' una lista di gruppi: tutti i gruppi devono comparire nella risposta, basta un alias per gruppo
  (confronto senza maiuscole/accenti/punteggiatura). Ogni risposta e' stata verificata con grep sul testo.
