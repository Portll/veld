//! What part of the graph a gap analysis covers.

use serde::Serialize;

/// Smallest `max_nodes` honoured: an open triad needs three entities.
pub const MIN_SCOPE_NODES: usize = 3;
/// Largest `max_nodes` honoured.
pub const MAX_SCOPE_NODES: usize = 1_000_000;

/// Filters that narrow an analysis to part of the graph, and a cap on its size.
/// An entity is in scope when it passes every filter given; the analysis runs
/// on the edges between entities in scope.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GapScope {
    /// Entities named `{namespace}:…`.
    pub namespace: Option<String>,
    /// Entities linked to the tag entity `project:{project}`, which is left out.
    pub project: Option<String>,
    /// Entities carrying any of these labels.
    pub entity_types: Vec<String>,
    /// Analyze at most this many entities: the most salient in scope.
    pub max_nodes: Option<usize>,
}

fn normalized(value: Option<String>) -> Option<String> {
    value
        .map(|v| v.trim().to_lowercase())
        .filter(|v| !v.is_empty())
}

impl GapScope {
    /// Filters trimmed and lowercased, empty ones dropped, types sorted and
    /// deduplicated, `max_nodes` clamped to the honoured range.
    pub fn new(
        namespace: Option<String>,
        project: Option<String>,
        entity_types: Vec<String>,
        max_nodes: Option<usize>,
    ) -> Self {
        let mut entity_types: Vec<String> = entity_types
            .into_iter()
            .filter_map(|t| normalized(Some(t)))
            .collect();
        entity_types.sort();
        entity_types.dedup();
        Self {
            namespace: normalized(namespace).map(|n| n.trim_end_matches(':').to_string()),
            project: normalized(project),
            entity_types,
            max_nodes: max_nodes.map(|n| n.clamp(MIN_SCOPE_NODES, MAX_SCOPE_NODES)),
        }
    }

    /// No filter and no cap: the analysis covers every entity.
    pub fn is_whole_graph(&self) -> bool {
        !self.has_filters() && self.max_nodes.is_none()
    }

    pub fn has_filters(&self) -> bool {
        self.namespace.is_some() || self.project.is_some() || !self.entity_types.is_empty()
    }

    /// The report of this scope over `nodes_in_scope` matching entities.
    pub fn report(&self, nodes_in_scope: usize, nodes_analyzed: usize) -> ScopeReport {
        let sampled = nodes_analyzed < nodes_in_scope;
        ScopeReport {
            namespace: self.namespace.clone(),
            project: self.project.clone(),
            entity_types: self.entity_types.clone(),
            max_nodes: self.max_nodes,
            nodes_in_scope,
            nodes_analyzed,
            sampled,
            sample_method: sampled.then_some("top_salience"),
        }
    }
}

/// The scope a result covers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScopeReport {
    pub namespace: Option<String>,
    pub project: Option<String>,
    pub entity_types: Vec<String>,
    pub max_nodes: Option<usize>,
    /// Entities the filters matched; every mirrored entity without filters.
    pub nodes_in_scope: usize,
    /// Entities the analysis ran over.
    pub nodes_analyzed: usize,
    /// Fewer entities were analyzed than matched.
    pub sampled: bool,
    /// How the sample was drawn, when `sampled`.
    pub sample_method: Option<&'static str>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_are_normalized_and_an_empty_scope_is_the_whole_graph() {
        let scope = GapScope::new(
            Some(" File: ".into()),
            Some("".into()),
            vec![
                "Person".into(),
                " person".into(),
                "".into(),
                "Concept".into(),
            ],
            Some(1),
        );
        assert_eq!(scope.namespace.as_deref(), Some("file"));
        assert_eq!(scope.project, None);
        assert_eq!(scope.entity_types, vec!["concept", "person"]);
        assert_eq!(scope.max_nodes, Some(MIN_SCOPE_NODES));
        assert!(GapScope::new(None, Some("  ".into()), vec![], None).is_whole_graph());
    }

    #[test]
    fn a_report_says_whether_it_sampled() {
        let scope = GapScope::new(None, None, vec![], Some(10));
        assert!(scope.report(40, 10).sampled);
        assert_eq!(scope.report(40, 10).sample_method, Some("top_salience"));
        assert!(!scope.report(8, 8).sampled);
    }
}
