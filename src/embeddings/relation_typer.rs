//! Semantic relation typing of co-mentioned entities.
//!
//! Ported from upstream shodh-memory `relation_typer.rs` (#325, 27819826; on by default since
//! 9f632af5). The sentence holding both mentions is template-normalised, the earlier mention
//! to "x" and the later to "y", embedded with the memory system's own embedder, and matched
//! against embedded exemplar templates ("x lives in y", "x works at y"). The nearest exemplar
//! types the edge when its cosine clears the floor. No model is added and nothing is trained:
//! a relation type is a list of exemplar lines.
//!
//! Three adaptations. Exemplar embeddings are cached per embedder id rather than once per
//! process, because a veld process can hold embedders of different dimensions. The cosine is
//! read on the MiniLM reference scale ([`crate::similarity::scale`]) before it meets the floor,
//! which upstream set on MiniLM. And a negated sentence is never typed.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use crate::embeddings::Embedder;
use crate::graph_memory::RelationType;

struct Exemplar {
    relation: RelationType,
    /// Whether the template's "x", the earlier mention, is the relation's source.
    x_is_source: bool,
    embedding: Vec<f32>,
}

/// `(relation, x_is_source, template)`. An effect-first phrasing ("x was caused by y") carries
/// `x_is_source = false`, so the direction lives in the exemplar.
fn exemplar_specs() -> Vec<(RelationType, bool, &'static str)> {
    use RelationType::*;
    vec![
        (Causes, true, "x caused y"),
        (Causes, true, "x led to y"),
        (Causes, false, "x happened because of y"),
        (Causes, false, "x was caused by y"),
        (WorksAt, true, "x works at y"),
        (WorksAt, true, "x joined y"),
        (Manages, true, "x manages y"),
        (CreatedBy, true, "x created y"),
        (CreatedBy, false, "x was created by y"),
        (Uses, true, "x uses y"),
        (LocatedIn, true, "x lives in y"),
        (LocatedIn, true, "x is located in y"),
        (LocatedIn, true, "x traveled to y"),
        (PartOf, true, "x is part of y"),
        (PartOf, true, "x is a member of y"),
        (DependsOn, true, "x depends on y"),
        (SupersededBy, true, "x was replaced by y"),
        (Knows, true, "x is friends with y"),
        (Knows, true, "x is married to y"),
        (Knows, true, "x met y"),
        (Knows, true, "x talked with y"),
        (Prefers, true, "x likes y"),
        (Prefers, true, "x enjoys y"),
        (Prefers, true, "x loves y"),
        (Teaches, true, "x taught y"),
        (Learned, true, "x learned y"),
        (AssociatedWith, true, "x attended y"),
        (AssociatedWith, true, "x went to y"),
        (AssociatedWith, true, "x participated in y"),
    ]
}

/// Exemplar embeddings per embedder id, built on first use (about thirty short encodes).
#[derive(Default)]
pub struct RelationTyper {
    exemplars: parking_lot::RwLock<HashMap<String, Arc<Vec<Exemplar>>>>,
}

/// The process-wide typer.
pub static RELATION_TYPER: LazyLock<RelationTyper> = LazyLock::new(RelationTyper::default);

impl RelationTyper {
    fn exemplars_for(&self, embedder: &dyn Embedder) -> Arc<Vec<Exemplar>> {
        if let Some(cached) = self.exemplars.read().get(embedder.id()) {
            return Arc::clone(cached);
        }
        let mut built = Vec::new();
        let mut complete = true;
        for (relation, x_is_source, text) in exemplar_specs() {
            match embedder.encode(text) {
                Ok(embedding) => built.push(Exemplar {
                    relation,
                    x_is_source,
                    embedding,
                }),
                Err(e) => {
                    complete = false;
                    tracing::warn!("relation exemplar embed failed ({text}): {e}");
                }
            }
        }
        let built = Arc::new(built);
        // A partial set would type every later write without the failed relations for the life
        // of the process; serve it this once and build again on the next write.
        if !complete {
            return built;
        }
        self.exemplars
            .write()
            .entry(embedder.id().to_string())
            .or_insert_with(|| Arc::clone(&built))
            .clone()
    }

    /// The relation between `name_a` and `name_b` in `text` as `(relation, a_is_source,
    /// cosine)`, the cosine on the reference scale. `None` when either name is missing from
    /// `text`, or no exemplar reaches `min_cosine`.
    pub fn type_relation(
        &self,
        embedder: &dyn Embedder,
        text: &str,
        name_a: &str,
        name_b: &str,
        min_cosine: f32,
    ) -> Option<(RelationType, bool, f32)> {
        let lc = text.to_ascii_lowercase();
        let a = name_a.to_ascii_lowercase();
        let b = name_b.to_ascii_lowercase();
        if a.is_empty() || b.is_empty() {
            return None;
        }
        let pa = lc.find(&a)?;
        let pb = lc.find(&b)?;
        if pa == pb {
            return None;
        }
        // The sentence holding both mentions, scoped as the cue extractor scopes it.
        let (lo, hi) = if pa < pb {
            (pa, pb + b.len())
        } else {
            (pb, pa + a.len())
        };
        let sent_start = lc[..lo]
            .rfind(['.', '!', '?', ';', '\n'])
            .map(|i| i + 1)
            .unwrap_or(0);
        let sent_end = lc[hi..]
            .find(['.', '!', '?', ';', '\n'])
            .map(|i| hi + i)
            .unwrap_or(lc.len());
        let a_first = pa < pb;
        let (x_name, y_name) = if a_first { (&a, &b) } else { (&b, &a) };
        let sentence = &lc[sent_start..sent_end];
        // A negated sentence says the relation does not hold, and its embedding still sits
        // next to the exemplar it negates: "x did not cause y" is nearest "x caused y".
        if sentence
            .split(|c: char| !(c.is_alphanumeric() || c == '\''))
            .any(|word| crate::memory::query_parser::is_negation(word) || word.ends_with("n't"))
        {
            return None;
        }
        // The longer name first, so a name nested in the other is not mangled.
        let normalized = if x_name.len() >= y_name.len() {
            sentence
                .replace(x_name.as_str(), "x")
                .replace(y_name.as_str(), "y")
        } else {
            sentence
                .replace(y_name.as_str(), "y")
                .replace(x_name.as_str(), "x")
        };
        let query = embedder.encode(normalized.trim()).ok()?;
        let exemplars = self.exemplars_for(embedder);
        let (best, cosine) = exemplars
            .iter()
            .map(|ex| {
                let raw = crate::similarity::cosine_similarity(&query, &ex.embedding);
                (ex, crate::similarity::scale::to_reference(raw))
            })
            .max_by(|l, r| l.1.total_cmp(&r.1))?;
        if cosine < min_cosine {
            return None;
        }
        let a_is_source = if a_first {
            best.x_is_source
        } else {
            !best.x_is_source
        };
        Some((best.relation.clone(), a_is_source, cosine))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::minilm::{EmbeddingConfig, MiniLMEmbedder};

    fn embedder() -> MiniLMEmbedder {
        MiniLMEmbedder::new_simplified(EmbeddingConfig::default()).expect("simplified embedder")
    }

    /// Fails its first `failures` encodes, then delegates.
    struct FlakyEmbedder {
        inner: MiniLMEmbedder,
        failures: std::sync::atomic::AtomicUsize,
    }

    impl Embedder for FlakyEmbedder {
        fn encode(&self, text: &str) -> anyhow::Result<Vec<f32>> {
            use std::sync::atomic::Ordering;
            if self
                .failures
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
            {
                anyhow::bail!("embedder unavailable");
            }
            self.inner.encode(text)
        }

        fn dimension(&self) -> usize {
            self.inner.dimension()
        }

        fn id(&self) -> &str {
            self.inner.id()
        }
    }

    #[test]
    fn a_failed_exemplar_encode_is_retried_rather_than_cached() {
        let embedder = FlakyEmbedder {
            inner: embedder(),
            failures: std::sync::atomic::AtomicUsize::new(1),
        };
        let typer = RelationTyper::default();
        let total = exemplar_specs().len();

        assert_eq!(typer.exemplars_for(&embedder).len(), total - 1);
        assert!(
            typer.exemplars.read().is_empty(),
            "a partial set is not cached"
        );

        assert_eq!(typer.exemplars_for(&embedder).len(), total);
        assert_eq!(
            typer.exemplars.read().len(),
            1,
            "the complete set is cached"
        );
    }

    /// The simplified (hash) embedder embeds identical strings identically, so a sentence that
    /// normalises to an exemplar template matches it at cosine 1.0.
    #[test]
    fn an_exact_template_sentence_is_typed_with_its_direction() {
        let embedder = embedder();
        let typer = RelationTyper::default();

        let (rt, a_is_source, cosine) = typer
            .type_relation(&embedder, "Redis caused Outage.", "Redis", "Outage", 0.6)
            .expect("exact template");
        assert_eq!(rt, RelationType::Causes);
        assert!(a_is_source);
        assert!(cosine > 0.99, "{cosine}");

        let (_, a_is_source, _) = typer
            .type_relation(&embedder, "Redis caused Outage.", "Outage", "Redis", 0.6)
            .expect("exact template");
        assert!(!a_is_source, "name_a is the effect");

        let (rt, a_is_source, _) = typer
            .type_relation(
                &embedder,
                "Outage was caused by Redis.",
                "Redis",
                "Outage",
                0.6,
            )
            .expect("effect-first template");
        assert_eq!(rt, RelationType::Causes);
        assert!(a_is_source, "the later-mentioned cause is the source");
    }

    #[test]
    fn a_missing_mention_or_a_cosine_under_the_floor_is_untyped() {
        let embedder = embedder();
        let typer = RelationTyper::default();
        assert!(typer
            .type_relation(&embedder, "Alpha met Beta.", "Alpha", "Epsilon", 0.6)
            .is_none());
        assert!(typer
            .type_relation(&embedder, "Alpha met Beta.", "Alpha", "Beta", 1.01)
            .is_none());
    }

    /// Upstream's typer reads "x did not cause y" as Causes: it is the nearest exemplar.
    #[test]
    fn a_negated_sentence_is_untyped_at_any_floor() {
        let embedder = embedder();
        let typer = RelationTyper::default();
        for sentence in [
            "Redis did not cause Outage.",
            "Redis never caused Outage.",
            "Redis isn't part of Outage.",
        ] {
            assert!(
                typer
                    .type_relation(&embedder, sentence, "Redis", "Outage", -1.0)
                    .is_none(),
                "{sentence}"
            );
        }
        assert!(typer
            .type_relation(&embedder, "Redis caused Outage.", "Redis", "Outage", -1.0)
            .is_some());
    }
}
