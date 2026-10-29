//! The server half of write-time fact extraction: draining every open tenant's queue off
//! the write path and journaling what the drain writes.
//!
//! [`crate::memory::write_time_facts`] holds the queue, the prompt and the store writes.
//! This half needs the manager, which owns the tenants and their intent logs: every fact
//! a pass creates, rewrites or removes is journaled like any other write, so projections,
//! replicas and point-in-time replay see it.

use std::sync::{Arc, LazyLock, Weak};

use dashmap::DashMap;

use super::remember::{journal_stored, SnapshotKind};
use super::state::MultiUserMemoryManager;
use crate::memory::write_time_facts::{extract, FactDrainReport, PreparedJob};
use crate::memory::Bm25Sync;
use crate::user_memory::SharedUserMemory;

/// Jobs one pass runs for one tenant before it moves to the next, so one busy tenant
/// cannot hold the others back and a shutdown is seen between jobs.
pub const JOBS_PER_TENANT_PASS: usize = 16;

/// One drain per tenant at a time, whoever starts it.
static WRITE_TIME_FACT_LOCKS: LazyLock<DashMap<String, Arc<tokio::sync::Mutex<()>>>> =
    LazyLock::new(DashMap::new);

impl MultiUserMemoryManager {
    /// One pass over the open tenants: up to `jobs_per_tenant` jobs for each tenant whose
    /// queue has one due. Returns what each tenant's jobs did. Blocking: each job waits on
    /// the model.
    pub fn drain_write_time_facts(&self, jobs_per_tenant: usize) -> Vec<(String, FactDrainReport)> {
        let due: Vec<(
            String,
            Weak<parking_lot::RwLock<crate::user_memory::UserMemory>>,
        )> = self
            .user_memories
            .iter()
            .filter(|(_, memory)| memory.read().write_time_facts().has_due_work())
            .map(|(user_id, memory)| (user_id.to_string(), Arc::downgrade(&memory)))
            .collect();
        let mut reports = Vec::new();
        for (user_id, tenant) in due {
            if self.writes_closed() {
                break;
            }
            let report = self.drain_tenant_write_time_facts(&user_id, &tenant, jobs_per_tenant);
            if report.jobs > 0 {
                reports.push((user_id, report));
            }
        }
        reports
    }

    /// Run up to `max_jobs` of one tenant's jobs. The tenant is held only to read a window
    /// and to write its facts, never while the model is thinking, so a tenant the cache
    /// evicts meanwhile closes as it would without the extractor; its job is then counted
    /// as dropped.
    fn drain_tenant_write_time_facts(
        &self,
        user_id: &str,
        tenant: &Weak<parking_lot::RwLock<crate::user_memory::UserMemory>>,
        max_jobs: usize,
    ) -> FactDrainReport {
        let lock = WRITE_TIME_FACT_LOCKS
            .entry(user_id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone();
        let mut report = FactDrainReport::default();
        let Ok(_draining) = lock.try_lock() else {
            return report;
        };
        for _ in 0..max_jobs {
            if self.writes_closed() {
                break;
            }
            let Some(memory) = tenant.upgrade() else {
                break;
            };
            let (ticket, prepared, llm) = {
                let guard = memory.read();
                let facts = guard.write_time_facts();
                let (Some(llm), Some(ticket)) = (facts.llm(), facts.take()) else {
                    break;
                };
                let prepared = guard.prepare_write_time_fact_job(&ticket);
                (ticket, prepared, llm)
            };
            drop(memory);
            let window = match prepared {
                PreparedJob::Ready(window) => window,
                PreparedJob::Done(done) => {
                    report.absorb(done);
                    continue;
                }
            };

            let extraction = extract(llm.as_ref(), &window);

            let Some(memory) = tenant.upgrade() else {
                crate::metrics::WRITE_TIME_FACT_FAILURES_TOTAL
                    .with_label_values(&["dropped"])
                    .inc();
                tracing::warn!(
                    user_id,
                    target = %ticket.target().0,
                    "write-time facts: the tenant closed while the model answered"
                );
                break;
            };
            let written = memory.read().apply_write_time_facts(
                ticket,
                &window,
                extraction,
                &llm.llm_version(),
                Bm25Sync::Journaled,
            );
            journal_written(self, &memory, user_id, &written);
            report.absorb(written);
        }
        report
    }
}

/// Journal what one job wrote: a `Remember` for each fact it created, an `Update` for
/// each it rewrote and a `Forget` for each it removed.
fn journal_written(
    state: &MultiUserMemoryManager,
    memory: &SharedUserMemory,
    user_id: &str,
    written: &FactDrainReport,
) {
    journal_stored(
        state,
        memory,
        user_id,
        &written.created_ids,
        SnapshotKind::Remember,
    );
    journal_stored(
        state,
        memory,
        user_id,
        &written.updated_ids,
        SnapshotKind::Update,
    );
    super::crud::journal_bulk_forget(state, user_id, written.removed_ids.iter().cloned());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::test_helpers::TestHarness;
    use crate::intent_log::{IntentLogIter, IntentPayload};
    use crate::memory::llm::{LlmConsolidator, LlmExtractedFact};
    use crate::memory::write_time_facts::{is_write_time_fact, META_SUPERSEDES};
    use crate::memory::{CognitiveDials, Experience, ExperienceType, Memory, MemoryId};
    use chrono::TimeZone;

    /// A deterministic model: where Nate lives, from the one turn it is shown.
    struct HomeCity;

    impl LlmConsolidator for HomeCity {
        fn extract_facts(&self, _: &[String]) -> anyhow::Result<Vec<LlmExtractedFact>> {
            Ok(Vec::new())
        }

        fn extract_with_schema(
            &self,
            snippets: &[String],
            _schema: &str,
        ) -> anyhow::Result<Vec<serde_json::Value>> {
            let city = if snippets.last().is_some_and(|s| s.contains("Denver")) {
                "Denver"
            } else {
                "Austin"
            };
            Ok(vec![serde_json::json!({
                "fact": format!("Nate lives in {city}."),
                "source_turns": [format!("T{}", snippets.len())],
                "entities": ["Nate", city],
                "subject": "Nate",
                "attribute": "home city",
            })])
        }

        fn llm_version(&self) -> String {
            "fake-1".to_string()
        }
    }

    fn snapshots(harness: &TestHarness, user: &str) -> Vec<(&'static str, Memory)> {
        let journal = harness
            .state()
            .user_dir(user)
            .expect("user dir")
            .join("intent.log");
        IntentLogIter::open_for_scan(&journal)
            .expect("scan the journal")
            .filter_map(|record| {
                let record = record.expect("journal frame");
                match crate::intent_log::payload::decode(&record.payload).expect("payload") {
                    IntentPayload::Remember { memory_bincode, .. } => {
                        Some(("remember", memory_bincode))
                    }
                    IntentPayload::Update { memory_bincode, .. } => {
                        Some(("update", memory_bincode))
                    }
                    _ => None,
                }
            })
            .map(|(kind, bytes)| {
                let memory =
                    crate::memory::storage::decode_memory_snapshot(&bytes).expect("snapshot");
                (kind, memory)
            })
            .collect()
    }

    #[test]
    fn a_pass_journals_the_facts_it_writes_and_the_windows_it_closes() {
        let harness = TestHarness::new();
        let manager = harness.state();
        let user = "write-time-facts-journal";
        let memory = manager.get_user_memory(user).expect("tenant");
        {
            let guard = memory.read();
            guard.set_write_time_fact_llm(Some(Arc::new(HomeCity)));
            guard.set_cognitive_dials(CognitiveDials {
                write_time_facts: true,
                write_time_fact_window: 0,
                ..Default::default()
            });
            for (content, month) in [
                ("Nate: I just moved into a flat in Austin.", 1),
                ("Nate: We packed up and moved to Denver.", 6),
            ] {
                guard
                    .remember(
                        Experience {
                            experience_type: ExperienceType::Conversation,
                            content: content.to_string(),
                            tags: vec!["conv-7".to_string()],
                            ..Default::default()
                        },
                        Some(
                            chrono::Utc
                                .with_ymd_and_hms(2022, month, 1, 12, 0, 0)
                                .unwrap(),
                        ),
                    )
                    .expect("remember");
            }
        }
        drop(memory);

        let reports = manager.drain_write_time_facts(JOBS_PER_TENANT_PASS);
        assert_eq!(reports.len(), 1);
        let (drained_user, report) = &reports[0];
        assert_eq!(drained_user, user);
        assert_eq!((report.jobs, report.created, report.superseded), (2, 2, 1));

        let frames = snapshots(&harness, user);
        let facts: Vec<&(&str, Memory)> = frames
            .iter()
            .filter(|(_, m)| is_write_time_fact(m))
            .collect();
        let remembered: Vec<&MemoryId> = facts
            .iter()
            .filter(|(kind, _)| *kind == "remember")
            .map(|(_, m)| &m.id)
            .collect();
        assert_eq!(
            remembered.len(),
            2,
            "each fact is journaled when it is created"
        );

        let austin = facts
            .iter()
            .find(|(_, m)| m.experience.content == "Nate lives in Austin.")
            .map(|(_, m)| m.id.clone())
            .expect("the Austin fact");
        let closed = facts
            .iter()
            .find(|(kind, m)| *kind == "update" && m.id == austin)
            .map(|(_, m)| m)
            .expect("the closed window is journaled as an update");
        assert!(closed.valid_until.is_some());
        let denver = facts
            .iter()
            .find(|(_, m)| m.experience.content == "Nate lives in Denver.")
            .map(|(_, m)| m)
            .expect("the Denver fact");
        assert_eq!(
            denver.experience.metadata.get(META_SUPERSEDES),
            Some(&austin.0.to_string()),
            "the frame carries the fact as written, its link to what it superseded included"
        );
        assert!(
            manager
                .drain_write_time_facts(JOBS_PER_TENANT_PASS)
                .is_empty(),
            "nothing is left to do"
        );
    }
}
