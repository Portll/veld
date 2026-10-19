//! The lexical leg must reach an inflected form of a query's own words.
//!
//! Port of upstream shodh-memory #560 (cef67213), adapted to veld's index shape.
//! conv-42_q62 of the LoCoMo gate asks "How many letters has Joanna recieved?":
//! "letters" occurs in no document of the corpus while several share its stem.
//! Upstream moved `content` and `tags` to tantivy's `en_stem` analyzer; veld
//! already pairs `content` with a hand-stemmed `stemmed_content` field and adds
//! each query word's stem to the query, so these tests pin that the pairing
//! reaches the gold documents, that `tags` (the holdout field) now stems too,
//! that an index built with the previous schema is rebuilt rather than reused,
//! and that `entities` stays literal so "Paris" and "Parison" do not collide.
//!
//! The documents are verbatim lines of `tests/recall/corpora/locomo-gate.jsonl`.

use std::collections::HashMap;

use veld::memory::hybrid_search::BM25Index;
use veld::memory::types::MemoryId;

const REJECTION_LETTER: &str = "Joanna: Nate, after finishing my screenplay I got a rejection \
                                letter from a major company. It really bummed me out.";
const GENERIC_LETTER: &str = "Joanna: They just sent me a generic rejection letter without much \
                              feedback. It's disheartening not knowing why it didn't work out.";
const READER_LETTER: &str = "Joanna: Yep. Last week, someone wrote me a letter after reading an \
                             online blog post I made about a hard moment in my life. Their words \
                             touched me; they said my story had brought them comfort. It was \
                             awesome to realize my words had that kind of power. It reminded me \
                             why I love writing.";
const NATE_LETTER: &str =
    "Nate: That letter is really awesome! Does it remind you of your childhood?";
const TOURNAMENT: &str =
    "Nate: Hey Joanna! That's cool! I won my first video game tournament last week - so exciting!";
const SHOOTER: &str = "Nate: Thanks! it's a team shooter game.";

struct Corpus {
    index: BM25Index,
    ids: HashMap<&'static str, MemoryId>,
}

impl Corpus {
    fn open(path: &std::path::Path, docs: &[(&'static str, &str, &[&str], &[&str])]) -> Self {
        let index = BM25Index::new(path).expect("open index");
        let mut ids = HashMap::new();
        for (handle, content, tags, entities) in docs {
            let id = MemoryId(uuid::Uuid::new_v4());
            let tags: Vec<String> = tags.iter().map(|t| t.to_string()).collect();
            let entities: Vec<String> = entities.iter().map(|e| e.to_string()).collect();
            index
                .upsert(&id, content, &tags, &entities)
                .expect("upsert");
            ids.insert(*handle, id);
        }
        index.commit().expect("commit");
        index.reload().expect("reload");
        Self { index, ids }
    }

    fn hits(&self, query: &str) -> Vec<&'static str> {
        let results = self.index.search(query, 20).expect("search");
        results
            .iter()
            .filter_map(|(id, _)| {
                self.ids
                    .iter()
                    .find(|(_, mid)| *mid == id)
                    .map(|(handle, _)| *handle)
            })
            .collect()
    }
}

fn gate_corpus(path: &std::path::Path) -> Corpus {
    Corpus::open(
        path,
        &[
            ("rejection", REJECTION_LETTER, &["conv-42", "Joanna"], &[]),
            ("generic", GENERIC_LETTER, &["conv-42", "Joanna"], &[]),
            ("reader", READER_LETTER, &["conv-42", "Joanna"], &[]),
            ("nate", NATE_LETTER, &["conv-42", "Nate"], &[]),
            ("tournament", TOURNAMENT, &["conv-42", "Nate"], &[]),
            ("shooter", SHOOTER, &["conv-42", "Nate"], &[]),
        ],
    )
}

/// conv-42_q62 verbatim. Every document that says "letter" must be a hit for a
/// query that says "letters", and the two gold turns must lead.
#[test]
fn letters_reaches_every_document_that_says_letter() {
    let home = tempfile::tempdir().expect("tempdir");
    let corpus = gate_corpus(&home.path().join("bm25_index"));

    let hits = corpus.hits("How many letters has Joanna recieved?");
    for gold in ["rejection", "reader"] {
        assert!(
            hits.iter()
                .position(|h| *h == gold)
                .is_some_and(|rank| rank < 4),
            "{gold} must rank in the top four for the letters query, got {hits:?}"
        );
    }
    for stem_match in ["generic", "nate"] {
        assert!(
            hits.contains(&stem_match),
            "{stem_match} says \"letter\" and must be reached by \"letters\", got {hits:?}"
        );
    }
    assert!(
        !hits.contains(&"shooter"),
        "a turn with neither the word nor its stem must not match, got {hits:?}"
    );
}

/// The inverse direction: a singular query reaches a plural document.
#[test]
fn a_singular_query_reaches_a_plural_document() {
    let home = tempfile::tempdir().expect("tempdir");
    let corpus = Corpus::open(
        &home.path().join("bm25_index"),
        &[
            (
                "plural",
                "Joanna: I have received three letters this month.",
                &[],
                &[],
            ),
            ("other", SHOOTER, &[], &[]),
        ],
    );
    let hits = corpus.hits("letter");
    assert_eq!(hits, vec!["plural"], "{hits:?}");
}

/// `tags` was the one field still on the default analyzer. A tag "letter" must
/// be reached by "letters", and a tag "letters" by "letter", without the
/// document's content saying either.
#[test]
fn tags_are_stemmed_in_both_directions() {
    let home = tempfile::tempdir().expect("tempdir");
    let corpus = Corpus::open(
        &home.path().join("bm25_index"),
        &[
            ("singular_tag", SHOOTER, &["letter"], &[]),
            ("plural_tag", TOURNAMENT, &["letters"], &[]),
            ("no_tag", NATE_LETTER, &["conv-42"], &[]),
        ],
    );

    let hits = corpus.hits("letters");
    assert!(
        hits.contains(&"singular_tag") && hits.contains(&"plural_tag"),
        "both tagged turns must be reached by \"letters\", got {hits:?}"
    );
    let hits = corpus.hits("letter");
    assert!(
        hits.contains(&"singular_tag") && hits.contains(&"plural_tag"),
        "both tagged turns must be reached by \"letter\", got {hits:?}"
    );
}

/// `entities` holds proper nouns and must stay literal: stemming would fold
/// "Paris" and "Parison" onto one term and make each a hit for the other.
#[test]
fn entities_are_not_stemmed_so_paris_and_parison_do_not_collide() {
    let home = tempfile::tempdir().expect("tempdir");
    let corpus = Corpus::open(
        &home.path().join("bm25_index"),
        &[
            (
                "paris",
                "We walked along the river at dusk.",
                &[],
                &["Paris"],
            ),
            (
                "parison",
                "The glassblower shaped it at dusk.",
                &[],
                &["Parison"],
            ),
        ],
    );
    let hits = corpus.hits("Paris");
    assert_eq!(hits, vec!["paris"], "{hits:?}");
    let hits = corpus.hits("Parison");
    assert_eq!(hits, vec!["parison"], "{hits:?}");
}

/// Reopening an index built by this binary keeps it: no rebuild, no marker,
/// every document still there.
#[test]
fn reopening_a_current_index_keeps_its_documents() {
    let home = tempfile::tempdir().expect("tempdir");
    let path = home.path().join("bm25_index");
    {
        let corpus = gate_corpus(&path);
        assert!(!corpus.index.rebuild_pending());
        assert_eq!(corpus.index.len(), 6);
    }
    let reopened = BM25Index::new(&path).expect("reopen");
    reopened.reload().expect("reload");
    assert!(
        !reopened.rebuild_pending(),
        "a current index is not rebuilt"
    );
    assert_eq!(reopened.len(), 6, "every document survives the reopen");
}

/// An index whose on-disk schema predates the stemmed `tags` analyzer would
/// otherwise go on indexing and querying unstemmed forever: `Index::open`
/// reads the analyzer names from `meta.json` and ignores the schema the caller
/// built. It must come back empty, stemmed, and flagged for a full replay.
#[test]
fn an_index_with_the_previous_schema_is_rebuilt_empty_and_stemmed() {
    use tantivy::schema::{Schema, STORED, STRING, TEXT};
    use tantivy::{Index, TantivyDocument};

    let home = tempfile::tempdir().expect("tempdir");
    let path = home.path().join("bm25_index");
    {
        let mut old = Schema::builder();
        old.add_text_field("id", STRING | STORED);
        old.add_text_field("content", TEXT | STORED);
        old.add_text_field("tags", TEXT);
        old.add_text_field("entities", TEXT);
        old.add_text_field("stemmed_content", TEXT);
        std::fs::create_dir_all(&path).expect("mkdir");
        let index = Index::create_in_dir(&path, old.build()).expect("old-schema index");
        let schema = index.schema();
        let mut writer = index.writer::<TantivyDocument>(15_000_000).expect("writer");
        let mut doc = TantivyDocument::new();
        doc.add_text(
            schema.get_field("id").unwrap(),
            uuid::Uuid::new_v4().to_string(),
        );
        doc.add_text(schema.get_field("content").unwrap(), NATE_LETTER);
        doc.add_text(schema.get_field("tags").unwrap(), "letter");
        writer.add_document(doc).expect("add");
        writer.commit().expect("commit");
    }

    let rebuilt = BM25Index::new(&path).expect("open rebuilds");
    rebuilt.reload().expect("reload");
    assert!(rebuilt.is_empty(), "the old-schema index must be discarded");
    assert!(
        rebuilt.rebuild_pending(),
        "the projection must be told to replay the whole log"
    );

    // The recreated index stems its tags.
    let id = MemoryId(uuid::Uuid::new_v4());
    rebuilt
        .upsert(&id, SHOOTER, &["letter".to_string()], &[])
        .expect("upsert");
    rebuilt.commit().expect("commit");
    rebuilt.reload().expect("reload");
    let hits = rebuilt.search("letters", 10).expect("search");
    assert_eq!(hits.len(), 1, "a tag \"letter\" is reached by \"letters\"");
    assert_eq!(hits[0].0, id);

    rebuilt.finish_rebuild().expect("finish");
    assert!(!rebuilt.rebuild_pending());
}
