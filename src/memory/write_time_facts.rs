//! Write-time facts: conversation turns rewritten, as they are stored, into
//! self-contained statements that are indexed as memories of their own.
//!
//! A raw chat line rarely matches the question asked about it. "I've had them for three
//! years" shares no words with "how long has Nate had turtles?"; "Nate has had turtles
//! for about three years, as of 23 January 2022" does. When the `write_time_facts` dial
//! is on and an [`LlmConsolidator`] is configured, every stored
//! [`ExperienceType::Conversation`] memory is queued here. A per-user background worker
//! shows the model that turn and up to `write_time_fact_window` earlier turns of the same
//! conversation, and stores each statement it returns as a [`RecordKind::Learning`]
//! record whose `experience.related_memories` are the turns the statement came from.
//! Recall then brings those turns into the candidate pool whenever the statement ranks
//! ([`expand_fact_sources`]).
//!
//! `remember` only queues: it never waits on the model. Nothing here runs without an
//! endpoint (`VELD_LLM_ENDPOINT`) or a model tape holding LLM outputs, so the offline
//! default is unchanged.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Condvar, Mutex, RwLock};
use sha2::{Digest, Sha256};

use super::facets::RecordKind;
use super::llm::{HttpLlmConsolidator, LlmConsolidator};
use super::types::{ChangeType, Experience, ExperienceType, Memory, MemoryId};
use super::{MemorySystem, UpsertProvenance};
use crate::metrics::{
    WRITE_TIME_FACTS_WRITTEN_TOTAL, WRITE_TIME_FACT_FAILURES_TOTAL, WRITE_TIME_FACT_QUEUE_DEPTH,
};

/// Version of the extraction prompt and schema below. Recorded on every fact and part of
/// its external id, so a fact can always be traced to the text that asked for it. Bump it
/// whenever [`INSTRUCTIONS`], a field description, the turn format or the schema changes.
pub const PROMPT_VERSION: &str = "wtf-1";

/// Prefix of every write-time fact's external id: `wtfact:<version>:<window>:<index>`.
pub const EXTERNAL_ID_PREFIX: &str = "wtfact:";

/// `experience.metadata` key holding [`PROMPT_VERSION`].
pub const META_PROMPT_VERSION: &str = "write_time_fact.prompt_version";

/// `experience.metadata` key holding the model that produced the fact.
pub const META_MODEL: &str = "write_time_fact.model";

/// Largest `write_time_fact_window` the dial accepts.
pub const MAX_WINDOW: usize = 16;

/// Facts kept from one window. A window's facts are numbered, so this also bounds how
/// many stale numbers a re-extraction has to clear.
pub const MAX_FACTS_PER_WINDOW: usize = 8;

/// Source turns one recall may add to the candidate pool.
pub const MAX_EXPANSION: usize = 20;

const MAX_FACT_CHARS: usize = 600;
const MAX_FACT_ENTITIES: usize = 16;

/// Turns remembered per conversation. Windows are fixed when a turn is queued, so only
/// the most recent turns are ever needed.
const TRACKED_TURNS: usize = MAX_WINDOW * 2;

/// Conversations whose recent turns are kept in memory; the least recently written is
/// dropped first and re-read from the store if it continues.
const TRACKED_CONVERSATIONS: usize = 4096;

/// Key-value marker written with the first fact, so recall skips expansion entirely on a
/// store that has never held one.
const FACTS_PRESENT_KEY: &[u8] = b"write_time_facts:present";

/// `changed_by` on a fact's revision history.
const CHANGED_BY: &str = "write-time-facts";

// ---------------------------------------------------------------------------
// The prompt. Every word the model reads is in this block.
// ---------------------------------------------------------------------------

/// What the model is asked to do, sent as the schema's description.
const INSTRUCTIONS: &str = "You are reading consecutive turns of one conversation, oldest \
first. Each turn is labelled T1, T2, ... and carries the date and time it was said; the \
speaker's name starts the turn. The last turn is marked TARGET. Return every durable fact \
the TARGET turn states, each as one element. Write each fact so that a reader who never \
saw the conversation understands it: replace every pronoun and vague reference with the \
name of the person, place or thing it refers to, using the earlier turns to resolve it. \
Turn every relative time into an absolute one computed from the date of the turn it was \
said in: \"yesterday\" said on Sunday 23 January 2022 becomes 22 January 2022, \"last \
week\" becomes the week before that date, and a duration such as \"for three years\" \
becomes \"for about three years, as of 23 January 2022\". Use the earlier turns only to \
resolve references; never return a fact that only an earlier turn states. Skip \
greetings, thanks, questions and small talk. Return an empty array when the TARGET turn \
states no fact.";

const FACT_FIELD: &str = "One self-contained sentence: names instead of pronouns, \
absolute dates instead of relative ones.";

const SOURCE_TURNS_FIELD: &str = "Labels of the turns this fact comes from: always the \
TARGET, plus every earlier turn needed to understand it.";

const ENTITIES_FIELD: &str = "The people, places, organisations, things and events the \
fact names, as written in the fact.";

/// One turn as the model reads it: `T3 (TARGET), said Sunday 23 January 2022, 19:31 UTC:
/// Nate: I've had them for three years.`
fn format_turn(turn: &WindowTurn, is_target: bool) -> String {
    let marker = if is_target { " (TARGET)" } else { "" };
    format!(
        "{}{marker}, said {}: {}",
        turn.label,
        turn.created_at.format("%A %-d %B %Y, %H:%M UTC"),
        turn.content
    )
}

/// The JSON Schema one extracted fact must satisfy. The turn labels are an enum, so a
/// source the window does not hold is a schema violation, not a guess to repair.
fn schema_for(window: &FactWindow) -> String {
    let labels: Vec<&str> = window.turns().map(|t| t.label.as_str()).collect();
    serde_json::json!({
        "title": "WriteTimeFact",
        "description": INSTRUCTIONS,
        "type": "object",
        "additionalProperties": false,
        "required": ["fact", "source_turns", "entities"],
        "properties": {
            "fact": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_FACT_CHARS,
                "description": FACT_FIELD,
            },
            "source_turns": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": true,
                "items": { "type": "string", "enum": labels },
                "description": SOURCE_TURNS_FIELD,
            },
            "entities": {
                "type": "array",
                "items": { "type": "string" },
                "description": ENTITIES_FIELD,
            },
        },
    })
    .to_string()
}

/// The `(snippets, schema)` pair passed to [`LlmConsolidator::extract_with_schema`].
pub fn request_for(window: &FactWindow) -> (Vec<String>, String) {
    let target = window.target_index();
    let snippets = window
        .turns()
        .enumerate()
        .map(|(i, t)| format_turn(t, i == target))
        .collect();
    (snippets, schema_for(window))
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

/// One turn shown to the model.
#[derive(Debug, Clone)]
pub struct WindowTurn {
    pub id: MemoryId,
    /// `T1`, `T2`, ... in conversation order.
    pub label: String,
    pub created_at: DateTime<Utc>,
    pub content: String,
}

/// The turns one extraction reads: earlier turns of the conversation, then the target.
#[derive(Debug, Clone)]
pub struct FactWindow {
    context: Vec<WindowTurn>,
    target: WindowTurn,
}

impl FactWindow {
    /// Label `memories` (oldest first, the target last) as `T1..Tn`. `None` when empty.
    pub fn from_turns(memories: &[Memory]) -> Option<Self> {
        let mut turns: Vec<WindowTurn> = memories
            .iter()
            .enumerate()
            .map(|(i, m)| WindowTurn {
                id: m.id.clone(),
                label: format!("T{}", i + 1),
                created_at: m.created_at,
                content: m.experience.content.clone(),
            })
            .collect();
        let target = turns.pop()?;
        Some(Self {
            context: turns,
            target,
        })
    }

    /// Every turn in conversation order, the target last.
    pub fn turns(&self) -> impl Iterator<Item = &WindowTurn> {
        self.context.iter().chain(std::iter::once(&self.target))
    }

    pub fn target(&self) -> &WindowTurn {
        &self.target
    }

    fn target_index(&self) -> usize {
        self.context.len()
    }

    fn index_of(&self, label: &str) -> Option<usize> {
        self.turns().position(|t| t.label == label)
    }

    /// Identifies the window across runs: the prompt version and every turn's id.
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(PROMPT_VERSION.as_bytes());
        for t in self.turns() {
            h.update(b"\n");
            h.update(t.id.0.as_bytes());
        }
        hex::encode(&h.finalize()[..16])
    }
}

/// The external id of the `index`th fact of the window with digest `digest`. The same
/// window under the same prompt always maps to the same ids, so extracting it again
/// updates its facts instead of adding to them.
pub fn external_id(digest: &str, index: usize) -> String {
    format!("{EXTERNAL_ID_PREFIX}{PROMPT_VERSION}:{digest}:{index}")
}

/// The conversation a turn belongs to: its episode id, else its first `conv-*` tag.
pub fn conversation_key(experience: &Experience) -> Option<String> {
    let episode = experience
        .context
        .as_ref()
        .and_then(|c| c.episode.episode_id.as_deref())
        .map(str::trim)
        .filter(|e| !e.is_empty());
    if let Some(episode) = episode {
        return Some(format!("episode:{episode}"));
    }
    experience
        .tags
        .iter()
        .find(|t| t.starts_with("conv-"))
        .map(|t| format!("tag:{t}"))
}

/// Whether `memory` is a write-time fact rather than a turn or any other record.
pub fn is_write_time_fact(memory: &Memory) -> bool {
    memory.kind == RecordKind::Learning
        && memory
            .external_id
            .as_deref()
            .is_some_and(|e| e.starts_with(EXTERNAL_ID_PREFIX))
}

// ---------------------------------------------------------------------------
// Parsing the model's answer
// ---------------------------------------------------------------------------

/// One statement the model returned that satisfied the schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedFact {
    pub text: String,
    /// The turns it came from, in conversation order. Always includes the target.
    pub sources: Vec<MemoryId>,
    pub entities: Vec<String>,
}

/// Keep the elements of `values` that satisfy [`schema_for`]; return them with the
/// number rejected. Duplicate statements collapse to the first, and at most
/// [`MAX_FACTS_PER_WINDOW`] are kept.
pub fn parse_facts(
    values: &[serde_json::Value],
    window: &FactWindow,
) -> (Vec<ExtractedFact>, usize) {
    let mut facts = Vec::new();
    let mut seen = HashSet::new();
    let mut rejected = 0;
    for value in values {
        match parse_fact(value, window) {
            Some(fact) => {
                if facts.len() < MAX_FACTS_PER_WINDOW && seen.insert(fact.text.to_lowercase()) {
                    facts.push(fact);
                }
            }
            None => rejected += 1,
        }
    }
    (facts, rejected)
}

fn parse_fact(value: &serde_json::Value, window: &FactWindow) -> Option<ExtractedFact> {
    const FIELDS: [&str; 3] = ["fact", "source_turns", "entities"];
    let obj = value.as_object()?;
    if obj.len() != FIELDS.len() || !FIELDS.iter().all(|f| obj.contains_key(*f)) {
        return None;
    }

    let text = obj["fact"]
        .as_str()?
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if text.is_empty() || text.chars().count() > MAX_FACT_CHARS {
        return None;
    }

    let labels = obj["source_turns"].as_array()?;
    if labels.is_empty() {
        return None;
    }
    let mut picked = HashSet::new();
    for label in labels {
        picked.insert(window.index_of(label.as_str()?)?);
    }
    // The statement was extracted from the target by construction, so the target is a
    // source whether or not the model listed it.
    picked.insert(window.target_index());
    let sources = window
        .turns()
        .enumerate()
        .filter(|(i, _)| picked.contains(i))
        .map(|(_, t)| t.id.clone())
        .collect();

    let mut entities = Vec::new();
    let mut seen = HashSet::new();
    for entity in obj["entities"].as_array()? {
        let entity = entity.as_str()?.trim();
        if !entity.is_empty()
            && entities.len() < MAX_FACT_ENTITIES
            && seen.insert(entity.to_lowercase())
        {
            entities.push(entity.to_string());
        }
    }

    Some(ExtractedFact {
        text,
        sources,
        entities,
    })
}

/// Ask `llm` for the facts of `window`: the facts that satisfied the schema and the
/// number of elements that did not.
pub fn extract(
    llm: &dyn LlmConsolidator,
    window: &FactWindow,
) -> Result<(Vec<ExtractedFact>, usize)> {
    let (snippets, schema) = request_for(window);
    let values = llm.extract_with_schema(&snippets, &schema)?;
    Ok(parse_facts(&values, window))
}

// ---------------------------------------------------------------------------
// The per-user queue
// ---------------------------------------------------------------------------

/// A turn waiting for extraction, with its window fixed when it was queued.
#[derive(Debug, Clone)]
struct Job {
    target: MemoryId,
    /// Earlier turns of the conversation, oldest first.
    context: Vec<MemoryId>,
}

#[derive(Debug, Clone)]
struct TurnRef {
    id: MemoryId,
    created_at: DateTime<Utc>,
    /// 0 for a turn stored before this process started, 1 for one stored since: at equal
    /// times the older process wrote first.
    epoch: u8,
    /// The episode sequence number of an older turn, the write order of a newer one.
    order: u64,
}

impl TurnRef {
    fn sort_key(&self) -> (DateTime<Utc>, u8, u64, uuid::Uuid) {
        (self.created_at, self.epoch, self.order, self.id.0)
    }
}

#[derive(Debug, Default)]
struct Conversation {
    /// Ordered by [`TurnRef::sort_key`]: creation time, then write order.
    turns: Vec<TurnRef>,
    touched: u64,
}

#[derive(Default)]
struct QueueState {
    pending: VecDeque<Job>,
    /// A job is being processed by the worker or a drain. One at a time keeps facts in
    /// queue order, so a drained run writes them in the same order every time.
    busy: bool,
    holds: usize,
    closed: bool,
    worker_started: bool,
    writes: u64,
    conversations: HashMap<String, Conversation>,
}

/// One user's write-time fact extractor: its model, its queue and the recent turns of the
/// conversations it is following.
pub struct WriteTimeFacts {
    llm: RwLock<Option<Arc<dyn LlmConsolidator>>>,
    state: Mutex<QueueState>,
    changed: Condvar,
    facts_present: AtomicBool,
}

impl WriteTimeFacts {
    pub(crate) fn new(facts_present: bool) -> Self {
        Self {
            llm: RwLock::new(None),
            state: Mutex::new(QueueState::default()),
            changed: Condvar::new(),
            facts_present: AtomicBool::new(facts_present),
        }
    }

    /// The model facts are extracted with; `None` leaves the extractor off.
    pub fn llm(&self) -> Option<Arc<dyn LlmConsolidator>> {
        self.llm.read().clone()
    }

    pub fn is_configured(&self) -> bool {
        self.llm.read().is_some()
    }

    /// Turns queued and not yet taken.
    pub fn pending(&self) -> usize {
        self.state.lock().pending.len()
    }

    /// Whether this store has ever held a write-time fact.
    pub fn facts_present(&self) -> bool {
        self.facts_present.load(Ordering::Relaxed)
    }

    /// Keep the background worker from taking jobs until the guard drops, so a caller can
    /// queue a whole corpus and drain it itself in queue order.
    pub fn hold(self: &Arc<Self>) -> FactHold {
        self.state.lock().holds += 1;
        FactHold {
            facts: Arc::clone(self),
        }
    }

    /// Wait until no job is queued or running. `false` when `timeout` passed first.
    pub fn wait_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut st = self.state.lock();
        while !st.pending.is_empty() || st.busy {
            if self.changed.wait_until(&mut st, deadline).timed_out() {
                return st.pending.is_empty() && !st.busy;
            }
        }
        true
    }

    fn push(&self, job: Job) {
        let mut st = self.state.lock();
        if st.closed {
            return;
        }
        st.pending.push_back(job);
        WRITE_TIME_FACT_QUEUE_DEPTH.inc();
        drop(st);
        self.changed.notify_all();
    }

    /// The next job for the background worker; `None` once the store is closing.
    fn next_for_worker(&self) -> Option<Job> {
        let mut st = self.state.lock();
        loop {
            if st.closed {
                return None;
            }
            if st.holds == 0 && !st.busy {
                if let Some(job) = st.pending.pop_front() {
                    st.busy = true;
                    WRITE_TIME_FACT_QUEUE_DEPTH.dec();
                    return Some(job);
                }
            }
            self.changed.wait(&mut st);
        }
    }

    /// The next job for a caller draining the queue itself; `None` when it is empty.
    fn next_for_drain(&self) -> Option<Job> {
        let mut st = self.state.lock();
        while st.busy {
            self.changed.wait(&mut st);
        }
        let job = st.pending.pop_front()?;
        st.busy = true;
        WRITE_TIME_FACT_QUEUE_DEPTH.dec();
        Some(job)
    }

    fn finish(&self) {
        self.state.lock().busy = false;
        self.changed.notify_all();
    }

    /// Stop the worker and drop what is queued. Called when the store is dropped.
    pub(crate) fn close(&self) {
        let mut st = self.state.lock();
        st.closed = true;
        WRITE_TIME_FACT_QUEUE_DEPTH.sub(st.pending.len() as i64);
        st.pending.clear();
        drop(st);
        self.changed.notify_all();
    }

    fn mark_present(&self) -> bool {
        !self.facts_present.swap(true, Ordering::Relaxed)
    }
}

/// See [`WriteTimeFacts::hold`].
pub struct FactHold {
    facts: Arc<WriteTimeFacts>,
}

impl Drop for FactHold {
    fn drop(&mut self) {
        self.facts.state.lock().holds -= 1;
        self.facts.changed.notify_all();
    }
}

/// What a drain, or one job, did.
#[derive(Debug, Clone, Default)]
pub struct FactDrainReport {
    pub jobs: usize,
    pub created: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub removed: usize,
    /// Elements the model returned that broke the schema.
    pub rejected: usize,
    /// Model calls and writes that failed.
    pub failures: usize,
    /// Every fact the jobs produced, created, updated or unchanged, in write order.
    pub fact_ids: Vec<MemoryId>,
}

impl FactDrainReport {
    fn absorb(&mut self, other: FactDrainReport) {
        self.jobs += other.jobs;
        self.created += other.created;
        self.updated += other.updated;
        self.unchanged += other.unchanged;
        self.removed += other.removed;
        self.rejected += other.rejected;
        self.failures += other.failures;
        self.fact_ids.extend(other.fact_ids);
    }
}

enum StoreOutcome {
    Created,
    Updated,
    Unchanged,
}

/// Build the extractor's model from the environment, or `None` to leave it off.
///
/// Live: `VELD_LLM_ENDPOINT` (with `VELD_LLM_MODEL`, `VELD_LLM_API_KEY`), the variables
/// [`HttpLlmConsolidator::from_env`] reads. Replaying a model tape that holds LLM
/// outputs: the model the tape names, answered from the tape and never contacted, so a
/// run recorded against a local model replays where there is none. An endpoint set
/// while replaying a tape without LLM outputs is reported as a tape problem rather than
/// called.
pub fn consolidator_from_env() -> Result<Option<Arc<dyn LlmConsolidator>>> {
    use crate::embeddings::model_tape;
    let endpoint_set = std::env::var("VELD_LLM_ENDPOINT").is_ok_and(|v| !v.trim().is_empty());
    let recorded = model_tape::recorded_llm_model();
    if !endpoint_set && recorded.is_none() {
        return Ok(None);
    }
    let llm = if model_tape::replaying() {
        let model = std::env::var("VELD_LLM_MODEL")
            .ok()
            .filter(|m| !m.trim().is_empty())
            .or(recorded)
            .unwrap_or_else(|| HttpLlmConsolidator::DEFAULT_MODEL.to_string());
        HttpLlmConsolidator::new(String::new(), String::new(), model)?
    } else {
        match HttpLlmConsolidator::from_env()? {
            Some(llm) => llm,
            None => return Ok(None),
        }
    };
    model_tape::register_llm(llm.model());
    tracing::info!(
        model = %llm.llm_version(),
        replaying = model_tape::replaying(),
        "write-time facts: LLM configured; the write_time_facts dial turns extraction on"
    );
    Ok(Some(Arc::new(llm)))
}

/// Start `system`'s background extractor if it has a model. The worker holds only a weak
/// reference, and none while it waits on the model, so it never keeps a store open; it
/// exits when the store is dropped.
pub fn start_worker<T>(system: &Arc<RwLock<T>>)
where
    T: Deref<Target = MemorySystem> + Send + Sync + 'static,
{
    let facts = Arc::clone(&system.read().write_time_facts);
    if !facts.is_configured() {
        return;
    }
    {
        let mut st = facts.state.lock();
        if st.worker_started || st.closed {
            return;
        }
        st.worker_started = true;
    }
    let weak = Arc::downgrade(system);
    let worker_facts = Arc::clone(&facts);
    let spawned = std::thread::Builder::new()
        .name("veld-write-time-facts".to_string())
        .spawn(move || worker_loop(&worker_facts, &weak));
    if let Err(e) = spawned {
        tracing::error!("write-time facts: could not start the worker: {e}");
        facts.state.lock().worker_started = false;
    }
}

fn worker_loop<T>(facts: &WriteTimeFacts, system: &Weak<RwLock<T>>)
where
    T: Deref<Target = MemorySystem>,
{
    while let Some(job) = facts.next_for_worker() {
        let alive = run_detached(facts, system, &job);
        facts.finish();
        if !alive {
            return;
        }
    }
}

/// Run one job, upgrading the weak reference only to read the window and to write the
/// facts. `false` when the store has been dropped.
fn run_detached<T>(facts: &WriteTimeFacts, system: &Weak<RwLock<T>>, job: &Job) -> bool
where
    T: Deref<Target = MemorySystem>,
{
    let window = {
        let Some(sys) = system.upgrade() else {
            return false;
        };
        let guard = sys.read();
        guard.fact_window(job)
    };
    let window = match window {
        Ok(Some(w)) => w,
        Ok(None) => return true,
        Err(e) => {
            fail(
                "store",
                &format!("reading the window of {}: {e:#}", job.target.0),
            );
            return true;
        }
    };
    let Some(llm) = facts.llm() else {
        return true;
    };
    let extraction = extract(llm.as_ref(), &window);
    let Some(sys) = system.upgrade() else {
        return false;
    };
    let guard = sys.read();
    guard.apply_extraction(&window, extraction, &llm.llm_version());
    true
}

fn fail(reason: &str, detail: &str) {
    WRITE_TIME_FACT_FAILURES_TOTAL
        .with_label_values(&[reason])
        .inc();
    tracing::warn!(reason, "write-time facts: {detail}");
}

// ---------------------------------------------------------------------------
// Recall expansion
// ---------------------------------------------------------------------------

/// What [`expand_fact_sources`] did to one candidate list.
#[derive(Debug, Default, PartialEq)]
pub struct FactExpansion {
    /// The source turns of every ranked fact, in the facts' rank order: the fact leg,
    /// whether or not each turn was already in the pool.
    pub leg: Vec<MemoryId>,
    /// Source turns that were not in the pool and were put into it.
    pub injected: usize,
}

/// Bring the source turns of every fact ranked in `ranked[..budget]` into the pool.
///
/// `ranked` is sorted by score, best first, and `budget` is how much of it survives the
/// truncation that follows. A source turn already inside the budget stays where it is;
/// any other goes right after its fact at the fact's score, moved up if it sat below the
/// budget, so the list stays sorted. At most `max_injected` are added, and the caller
/// widens its truncation by [`FactExpansion::injected`]. The facts keep their places.
pub fn expand_fact_sources(
    ranked: &mut Vec<(MemoryId, f32)>,
    budget: usize,
    max_injected: usize,
    sources_of: impl Fn(&MemoryId) -> Option<Vec<MemoryId>>,
) -> FactExpansion {
    let budget = budget.min(ranked.len());
    let mut pool: HashSet<MemoryId> = ranked[..budget].iter().map(|(id, _)| id.clone()).collect();
    let mut leg = Vec::new();
    let mut in_leg = HashSet::new();
    let mut after: HashMap<usize, Vec<MemoryId>> = HashMap::new();
    let mut injected = 0;
    for (rank, (id, _)) in ranked[..budget].iter().enumerate() {
        let Some(sources) = sources_of(id) else {
            continue;
        };
        for source in sources {
            if in_leg.insert(source.clone()) {
                leg.push(source.clone());
            }
            if injected < max_injected && pool.insert(source.clone()) {
                after.entry(rank).or_default().push(source);
                injected += 1;
            }
        }
    }
    if injected == 0 {
        return FactExpansion { leg, injected };
    }

    let moved: HashSet<MemoryId> = after.values().flatten().cloned().collect();
    let mut expanded = Vec::with_capacity(ranked.len() + injected);
    for (rank, (id, score)) in ranked.drain(..).enumerate() {
        if rank >= budget {
            if !moved.contains(&id) {
                expanded.push((id, score));
            }
            continue;
        }
        expanded.push((id, score));
        if let Some(sources) = after.remove(&rank) {
            expanded.extend(sources.into_iter().map(|s| (s, score)));
        }
    }
    *ranked = expanded;
    FactExpansion { leg, injected }
}

// ---------------------------------------------------------------------------
// MemorySystem integration
// ---------------------------------------------------------------------------

/// Whether the store holds a write-time fact, from its key-value marker.
pub(crate) fn facts_present_in(kv: &dyn crate::storage::KeyValueStore) -> bool {
    matches!(kv.get_bytes(FACTS_PRESENT_KEY), Ok(Some(_)))
}

impl MemorySystem {
    /// Give this store's extractor a model, or take it away with `None`. Extraction runs
    /// only while the `write_time_facts` dial is also on.
    pub fn set_write_time_fact_llm(&self, llm: Option<Arc<dyn LlmConsolidator>>) {
        *self.write_time_facts.llm.write() = llm;
    }

    /// The extractor's queue, for a caller that must wait for it or hold it.
    pub fn write_time_facts(&self) -> Arc<WriteTimeFacts> {
        Arc::clone(&self.write_time_facts)
    }

    /// The source turns of `id` when it is a write-time fact; `None` for anything else.
    pub(crate) fn write_time_fact_sources(&self, id: &MemoryId) -> Option<Vec<MemoryId>> {
        let cached = self
            .working_memory
            .read()
            .get(id)
            .or_else(|| self.session_memory.read().get(id));
        let sources = match cached {
            Some(memory) => {
                is_write_time_fact(&memory).then(|| memory.experience.related_memories.clone())
            }
            None => self
                .long_term_memory
                .get_memory(id)
                .ok()
                .flatten()
                .filter(is_write_time_fact)
                .map(|memory| memory.experience.related_memories),
        };
        sources.filter(|s| !s.is_empty())
    }

    /// Queue `memory` for extraction when it is a conversation turn, the dial is on and a
    /// model is configured. Called from the create path under the write lock, which makes
    /// the queue order the write order.
    pub(crate) fn queue_write_time_facts(&self, memory: &Memory) {
        if memory.experience.experience_type != ExperienceType::Conversation
            || !self.write_time_facts.is_configured()
        {
            return;
        }
        let window = {
            let dials = self.cognitive_dials.read();
            if !dials.write_time_facts {
                return;
            }
            dials.write_time_fact_window.min(MAX_WINDOW)
        };
        let context = match conversation_key(&memory.experience) {
            Some(key) => self.track_turn(&key, memory, window),
            None => Vec::new(),
        };
        self.write_time_facts.push(Job {
            target: memory.id.clone(),
            context,
        });
    }

    /// Record `memory` as the newest turn of conversation `key` and return the `window`
    /// turns before it, oldest first.
    fn track_turn(&self, key: &str, memory: &Memory, window: usize) -> Vec<MemoryId> {
        let known = self
            .write_time_facts
            .state
            .lock()
            .conversations
            .contains_key(key);
        // A conversation first seen in this process may have begun in an earlier one.
        let earlier = if known {
            Vec::new()
        } else {
            self.stored_turns(key, &memory.id)
        };

        let mut st = self.write_time_facts.state.lock();
        st.writes += 1;
        let writes = st.writes;
        // Re-checked under the lock: a writer outside the write lock may have started
        // following the conversation since, and then holds everything `earlier` would add.
        let first = !st.conversations.contains_key(key);
        let conversation = st.conversations.entry(key.to_string()).or_default();
        if first {
            conversation.turns.extend(earlier);
            conversation.turns.sort_by_key(TurnRef::sort_key);
        }
        conversation.touched = writes;
        let turn = TurnRef {
            id: memory.id.clone(),
            created_at: memory.created_at,
            epoch: 1,
            order: writes,
        };
        let at = conversation
            .turns
            .partition_point(|t| t.sort_key() < turn.sort_key());
        let context = conversation.turns[at.saturating_sub(window)..at]
            .iter()
            .map(|t| t.id.clone())
            .collect();
        conversation.turns.insert(at, turn);
        let excess = conversation.turns.len().saturating_sub(TRACKED_TURNS);
        conversation.turns.drain(..excess);

        if st.conversations.len() > TRACKED_CONVERSATIONS {
            let stalest = st
                .conversations
                .iter()
                .min_by_key(|(_, c)| c.touched)
                .map(|(k, _)| k.clone());
            if let Some(stalest) = stalest {
                st.conversations.remove(&stalest);
            }
        }
        context
    }

    /// Turns of conversation `key` already in the store, other than `exclude`.
    fn stored_turns(&self, key: &str, exclude: &MemoryId) -> Vec<TurnRef> {
        use super::storage::SearchCriteria;
        let criteria = if let Some(episode) = key.strip_prefix("episode:") {
            SearchCriteria::ByEpisode(episode.to_string())
        } else if let Some(tag) = key.strip_prefix("tag:") {
            SearchCriteria::ByTags(vec![tag.to_string()])
        } else {
            return Vec::new();
        };
        let found = match self.long_term_memory.search(criteria) {
            Ok(found) => found,
            Err(e) => {
                fail("store", &format!("reading earlier turns of {key}: {e:#}"));
                return Vec::new();
            }
        };
        found
            .into_iter()
            .filter(|m| {
                m.id != *exclude
                    && m.experience.experience_type == ExperienceType::Conversation
                    && conversation_key(&m.experience).as_deref() == Some(key)
            })
            .map(|m| TurnRef {
                order: m
                    .experience
                    .context
                    .as_ref()
                    .and_then(|c| c.episode.sequence_number)
                    .map_or(0, u64::from),
                id: m.id,
                created_at: m.created_at,
                epoch: 0,
            })
            .collect()
    }

    /// Load the window of `job`. `None` when the target no longer exists; an earlier turn
    /// that has gone is left out.
    fn fact_window(&self, job: &Job) -> Result<Option<FactWindow>> {
        let Some(target) = self.long_term_memory.get_memory(&job.target)? else {
            return Ok(None);
        };
        let mut turns = Vec::with_capacity(job.context.len() + 1);
        for id in &job.context {
            if let Some(turn) = self.long_term_memory.get_memory(id)? {
                turns.push(turn);
            }
        }
        turns.push(target);
        Ok(FactWindow::from_turns(&turns))
    }

    /// Run every queued job on this thread, in queue order, and report what they did.
    /// Waits for a job the worker is running first. A run that holds the queue
    /// ([`WriteTimeFacts::hold`]) and then drains writes the same facts in the same order
    /// every time the model answers the same way.
    pub fn drain_write_time_facts(&self) -> FactDrainReport {
        let facts = Arc::clone(&self.write_time_facts);
        let mut report = FactDrainReport::default();
        while let Some(job) = facts.next_for_drain() {
            report.absorb(self.run_write_time_fact_job(&facts, &job));
            facts.finish();
        }
        report
    }

    fn run_write_time_fact_job(&self, facts: &WriteTimeFacts, job: &Job) -> FactDrainReport {
        let window = match self.fact_window(job) {
            Ok(Some(w)) => w,
            Ok(None) => return FactDrainReport::default(),
            Err(e) => {
                fail(
                    "store",
                    &format!("reading the window of {}: {e:#}", job.target.0),
                );
                return FactDrainReport {
                    jobs: 1,
                    failures: 1,
                    ..Default::default()
                };
            }
        };
        let Some(llm) = facts.llm() else {
            return FactDrainReport::default();
        };
        let extraction = extract(llm.as_ref(), &window);
        self.apply_extraction(&window, extraction, &llm.llm_version())
    }

    /// Store what the model returned for `window`. A failed call changes nothing already
    /// stored; a successful one replaces the window's facts, removing any numbered past
    /// what it returned this time.
    fn apply_extraction(
        &self,
        window: &FactWindow,
        extraction: Result<(Vec<ExtractedFact>, usize)>,
        model: &str,
    ) -> FactDrainReport {
        let mut report = FactDrainReport {
            jobs: 1,
            ..Default::default()
        };
        let (facts, rejected) = match extraction {
            Ok(parsed) => parsed,
            Err(e) => {
                fail(
                    "llm",
                    &format!("extracting from {}: {e:#}", window.target().id.0),
                );
                report.failures = 1;
                return report;
            }
        };
        if rejected > 0 {
            WRITE_TIME_FACT_FAILURES_TOTAL
                .with_label_values(&["invalid"])
                .inc_by(rejected as u64);
            report.rejected = rejected;
        }

        let digest = window.digest();
        for (index, fact) in facts.iter().enumerate() {
            match self.store_write_time_fact(window, &digest, index, fact, model) {
                Ok((id, outcome)) => {
                    let label = match outcome {
                        StoreOutcome::Created => {
                            report.created += 1;
                            "created"
                        }
                        StoreOutcome::Updated => {
                            report.updated += 1;
                            "updated"
                        }
                        StoreOutcome::Unchanged => {
                            report.unchanged += 1;
                            "unchanged"
                        }
                    };
                    WRITE_TIME_FACTS_WRITTEN_TOTAL
                        .with_label_values(&[label])
                        .inc();
                    report.fact_ids.push(id);
                }
                Err(e) => {
                    fail("store", &format!("storing fact {index} of {digest}: {e:#}"));
                    report.failures += 1;
                }
            }
        }
        for index in facts.len()..MAX_FACTS_PER_WINDOW {
            match self.remove_write_time_fact(&external_id(&digest, index)) {
                Ok(true) => {
                    WRITE_TIME_FACTS_WRITTEN_TOTAL
                        .with_label_values(&["removed"])
                        .inc();
                    report.removed += 1;
                }
                Ok(false) => {}
                Err(e) => {
                    fail(
                        "store",
                        &format!("removing fact {index} of {digest}: {e:#}"),
                    );
                    report.failures += 1;
                }
            }
        }
        report
    }

    fn store_write_time_fact(
        &self,
        window: &FactWindow,
        digest: &str,
        index: usize,
        fact: &ExtractedFact,
        model: &str,
    ) -> Result<(MemoryId, StoreOutcome)> {
        let eid = external_id(digest, index);
        if let Some(existing) = self.long_term_memory.find_by_external_id(&eid)? {
            let same = existing.experience.content == fact.text
                && existing.experience.related_memories == fact.sources
                && existing.experience.entities == fact.entities
                && existing
                    .experience
                    .metadata
                    .get(META_MODEL)
                    .map(String::as_str)
                    == Some(model);
            if same {
                return Ok((existing.id, StoreOutcome::Unchanged));
            }
        }

        let metadata = HashMap::from([
            (META_PROMPT_VERSION.to_string(), PROMPT_VERSION.to_string()),
            (META_MODEL.to_string(), model.to_string()),
        ]);
        let experience = Experience {
            experience_type: ExperienceType::Learning,
            content: fact.text.clone(),
            entities: fact.entities.clone(),
            tags: fact.entities.clone(),
            related_memories: fact.sources.clone(),
            metadata: metadata.clone(),
            ..Default::default()
        };
        let provenance = UpsertProvenance {
            // A fact is dated by the turn that stated it.
            created_at: Some(window.target().created_at),
            kind: Some(RecordKind::Learning),
            ..Default::default()
        };
        let (id, updated) = self.upsert_with_provenance(
            eid,
            experience,
            ChangeType::ContentUpdated,
            Some(CHANGED_BY.to_string()),
            Some(format!("re-extracted under {PROMPT_VERSION}")),
            provenance,
        )?;

        if updated {
            // The upsert update path rewrites content, entities and tags; the links to
            // the source turns and the provenance are this record's own.
            let _write_guard = self.write_lock.lock();
            let mut stored = self.long_term_memory.get(&id)?;
            stored.experience.related_memories = fact.sources.clone();
            stored.experience.metadata.extend(metadata);
            self.update_memory(&stored)?;
        }

        if self.write_time_facts.mark_present() {
            if let Err(e) = self
                .long_term_memory
                .key_value_store()
                .put_bytes(FACTS_PRESENT_KEY, b"1")
            {
                tracing::warn!("write-time facts: could not persist the presence marker: {e}");
            }
        }

        let outcome = if updated {
            StoreOutcome::Updated
        } else {
            StoreOutcome::Created
        };
        Ok((id, outcome))
    }

    /// Remove the fact with external id `eid`. `false` when there is none.
    fn remove_write_time_fact(&self, eid: &str) -> Result<bool> {
        let Some(existing) = self.long_term_memory.find_by_external_id(eid)? else {
            return Ok(false);
        };
        if !is_write_time_fact(&existing) {
            return Ok(false);
        }
        let _write_guard = self.write_lock.lock();
        self.forget(super::types::ForgetCriteria::ById(existing.id))?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::llm::LlmExtractedFact;
    use crate::memory::{CognitiveDials, MemoryConfig};
    use chrono::TimeZone;
    use serde_json::json;
    use tempfile::TempDir;

    type Script = dyn Fn(&[String]) -> Vec<serde_json::Value> + Send + Sync;

    /// Answers each request with what its script returns, and keeps every request. The
    /// same shape as the `FakeLlm` in the tests of `memory::llm`.
    struct ScriptedLlm {
        script: Mutex<Box<Script>>,
        requests: Mutex<Vec<(Vec<String>, String)>>,
    }

    impl ScriptedLlm {
        fn new(
            script: impl Fn(&[String]) -> Vec<serde_json::Value> + Send + Sync + 'static,
        ) -> Arc<Self> {
            Arc::new(Self {
                script: Mutex::new(Box::new(script)),
                requests: Mutex::new(Vec::new()),
            })
        }

        fn rescript(
            &self,
            script: impl Fn(&[String]) -> Vec<serde_json::Value> + Send + Sync + 'static,
        ) {
            *self.script.lock() = Box::new(script);
        }
    }

    impl LlmConsolidator for ScriptedLlm {
        fn extract_facts(&self, _: &[String]) -> Result<Vec<LlmExtractedFact>> {
            Ok(Vec::new())
        }

        fn extract_with_schema(
            &self,
            snippets: &[String],
            schema_json: &str,
        ) -> Result<Vec<serde_json::Value>> {
            self.requests
                .lock()
                .push((snippets.to_vec(), schema_json.to_string()));
            let script = self.script.lock();
            Ok((**script)(snippets))
        }

        fn llm_version(&self) -> String {
            "scripted-1".to_string()
        }
    }

    /// One fact about the target of each request, which is always the last turn.
    fn one_fact_per_turn(snippets: &[String]) -> Vec<serde_json::Value> {
        let target = format!("T{}", snippets.len());
        vec![json!({
            "fact": format!("Statement drawn from turn {}.", snippets.len()),
            "source_turns": [target],
            "entities": ["Nate"],
        })]
    }

    fn at(day: u32, minute: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2022, 1, day, 19, minute, 0).unwrap()
    }

    fn turn(content: &str, created_at: DateTime<Utc>) -> Memory {
        Memory::new(
            MemoryId(uuid::Uuid::new_v4()),
            Experience {
                experience_type: ExperienceType::Conversation,
                content: content.to_string(),
                tags: vec!["conv-42".to_string()],
                ..Default::default()
            },
            0.5,
            None,
            None,
            None,
            Some(created_at),
        )
    }

    fn two_turn_window() -> FactWindow {
        FactWindow::from_turns(&[
            turn("Joanna: How are your turtles doing?", at(23, 30)),
            turn(
                "Nate: Great! I've had them for three years now.",
                at(23, 31),
            ),
        ])
        .expect("window")
    }

    fn system(window: usize) -> (MemorySystem, TempDir) {
        let tmp = TempDir::new().expect("temp dir");
        let sys = MemorySystem::new(
            MemoryConfig {
                storage_path: tmp.path().to_path_buf(),
                ..Default::default()
            },
            None,
        )
        .expect("memory system");
        sys.set_cognitive_dials(CognitiveDials {
            write_time_facts: true,
            write_time_fact_window: window,
            ..Default::default()
        });
        (sys, tmp)
    }

    fn remember_turn(sys: &MemorySystem, content: &str, created_at: DateTime<Utc>) -> MemoryId {
        sys.remember(
            Experience {
                experience_type: ExperienceType::Conversation,
                content: content.to_string(),
                tags: vec!["conv-42".to_string()],
                ..Default::default()
            },
            Some(created_at),
        )
        .expect("remember")
    }

    fn stored_facts(sys: &MemorySystem) -> Vec<Memory> {
        let mut facts: Vec<Memory> = sys
            .long_term_memory
            .get_all()
            .expect("all memories")
            .into_iter()
            .filter(is_write_time_fact)
            .collect();
        facts.sort_by(|a, b| a.external_id.cmp(&b.external_id));
        facts
    }

    fn queued(sys: &MemorySystem) -> Vec<(MemoryId, Vec<MemoryId>)> {
        sys.write_time_facts
            .state
            .lock()
            .pending
            .iter()
            .map(|j| (j.target.clone(), j.context.clone()))
            .collect()
    }

    #[test]
    fn the_request_labels_turns_dates_them_and_marks_the_target() {
        let window = two_turn_window();
        let (snippets, schema) = request_for(&window);
        assert_eq!(
            snippets,
            vec![
                "T1, said Sunday 23 January 2022, 19:30 UTC: Joanna: How are your turtles doing?"
                    .to_string(),
                "T2 (TARGET), said Sunday 23 January 2022, 19:31 UTC: Nate: Great! I've had \
                 them for three years now."
                    .to_string(),
            ]
        );

        let schema: serde_json::Value = serde_json::from_str(&schema).expect("schema is JSON");
        assert_eq!(schema["description"], INSTRUCTIONS);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["required"],
            json!(["fact", "source_turns", "entities"])
        );
        assert_eq!(
            schema["properties"]["source_turns"]["items"]["enum"],
            json!(["T1", "T2"])
        );
        assert_eq!(schema["properties"]["fact"]["maxLength"], MAX_FACT_CHARS);
    }

    #[test]
    fn only_elements_that_satisfy_the_schema_are_kept() {
        let window = two_turn_window();
        let ids: Vec<MemoryId> = window.turns().map(|t| t.id.clone()).collect();
        let values = vec![
            json!({
                "fact": "  Nate has had turtles for about three years,\n as of 23 January 2022. ",
                "source_turns": ["T1"],
                "entities": ["Nate", " turtles ", "nate", ""],
            }),
            // The same statement again, cased differently.
            json!({
                "fact": "nate has had turtles for about three years, as of 23 january 2022.",
                "source_turns": ["T2"],
                "entities": [],
            }),
            json!({ "fact": "Missing a field.", "source_turns": ["T2"] }),
            json!({
                "fact": "An extra field.",
                "source_turns": ["T2"],
                "entities": [],
                "confidence": 1,
            }),
            json!({ "fact": "An unknown turn.", "source_turns": ["T9"], "entities": [] }),
            json!({ "fact": "No turns.", "source_turns": [], "entities": [] }),
            json!({ "fact": "   ", "source_turns": ["T2"], "entities": [] }),
            json!({ "fact": "A bad entity.", "source_turns": ["T2"], "entities": [3] }),
            json!("not an object"),
        ];
        let (facts, rejected) = parse_facts(&values, &window);
        assert_eq!(rejected, 7);
        assert_eq!(
            facts,
            vec![ExtractedFact {
                text: "Nate has had turtles for about three years, as of 23 January 2022."
                    .to_string(),
                // T1 as listed, and the target T2 although the model left it out.
                sources: ids,
                entities: vec!["Nate".to_string(), "turtles".to_string()],
            }]
        );
    }

    #[test]
    fn at_most_the_per_window_cap_is_kept() {
        let window = two_turn_window();
        let values: Vec<serde_json::Value> = (0..MAX_FACTS_PER_WINDOW + 3)
            .map(
                |i| json!({ "fact": format!("Fact {i}."), "source_turns": ["T2"], "entities": [] }),
            )
            .collect();
        let (facts, rejected) = parse_facts(&values, &window);
        assert_eq!(facts.len(), MAX_FACTS_PER_WINDOW);
        assert_eq!(rejected, 0);
    }

    #[test]
    fn an_episode_id_names_the_conversation_before_a_tag() {
        let mut exp = Experience {
            tags: vec!["Nate".to_string(), "conv-42".to_string()],
            ..Default::default()
        };
        assert_eq!(conversation_key(&exp).as_deref(), Some("tag:conv-42"));
        let mut ctx = crate::memory::session_detect::rich_context_with_session();
        ctx.episode.episode_id = Some("ep-7".to_string());
        exp.context = Some(ctx);
        assert_eq!(conversation_key(&exp).as_deref(), Some("episode:ep-7"));
        exp.context = None;
        exp.tags = vec!["Nate".to_string()];
        assert_eq!(conversation_key(&exp), None);
    }

    #[test]
    fn the_window_digest_depends_on_the_turns_and_their_order() {
        let a = turn("A", at(23, 1));
        let b = turn("B", at(23, 2));
        let ab = FactWindow::from_turns(&[a.clone(), b.clone()]).expect("window");
        let ba = FactWindow::from_turns(&[b, a.clone()]).expect("window");
        let a_only = FactWindow::from_turns(&[a]).expect("window");
        assert_eq!(ab.digest(), ab.clone().digest());
        assert_ne!(ab.digest(), ba.digest());
        assert_ne!(ab.digest(), a_only.digest());
        let eid = external_id(&ab.digest(), 3);
        assert!(eid.starts_with("wtfact:wtf-1:"), "{eid}");
        assert!(eid.ends_with(":3"), "{eid}");
    }

    #[test]
    fn nothing_is_queued_while_the_dial_is_off_or_no_model_is_configured() {
        let (sys, _tmp) = system(4);
        remember_turn(&sys, "Nate: no model yet.", at(21, 1));
        assert_eq!(sys.write_time_facts.pending(), 0);

        sys.set_write_time_fact_llm(Some(ScriptedLlm::new(|_| Vec::new())));
        sys.set_cognitive_dials(CognitiveDials::default());
        remember_turn(&sys, "Nate: the dial is off.", at(21, 2));
        assert_eq!(sys.write_time_facts.pending(), 0);

        sys.set_cognitive_dials(CognitiveDials {
            write_time_facts: true,
            ..Default::default()
        });
        sys.remember(
            Experience {
                experience_type: ExperienceType::Observation,
                content: "Not a conversation turn.".to_string(),
                ..Default::default()
            },
            Some(at(21, 3)),
        )
        .expect("remember");
        assert_eq!(sys.write_time_facts.pending(), 0);

        remember_turn(&sys, "Nate: now it runs.", at(21, 4));
        assert_eq!(sys.write_time_facts.pending(), 1);
    }

    #[test]
    fn a_window_holds_the_earlier_turns_of_its_conversation_in_write_order() {
        let (sys, _tmp) = system(2);
        sys.set_write_time_fact_llm(Some(ScriptedLlm::new(|_| Vec::new())));
        // One timestamp for the whole session, as LoCoMo stores it, so only the write
        // order tells the turns apart.
        let t1 = remember_turn(&sys, "Nate: one.", at(21, 0));
        let t2 = remember_turn(&sys, "Joanna: two.", at(21, 0));
        let t3 = remember_turn(&sys, "Nate: three.", at(21, 0));
        let t4 = remember_turn(&sys, "Joanna: four.", at(21, 0));
        assert_eq!(
            queued(&sys),
            vec![
                (t1.clone(), vec![]),
                (t2.clone(), vec![t1.clone()]),
                (t3.clone(), vec![t1, t2.clone()]),
                (t4, vec![t2, t3]),
            ]
        );
    }

    #[test]
    fn remember_returns_before_the_model_answers() {
        let (sys, _tmp) = system(4);
        let llm = ScriptedLlm::new(one_fact_per_turn);
        sys.set_write_time_fact_llm(Some(llm.clone()));
        remember_turn(
            &sys,
            "Nate: I won my first video game tournament.",
            at(21, 5),
        );
        assert!(llm.requests.lock().is_empty());
        assert_eq!(sys.write_time_facts.pending(), 1);
        assert!(stored_facts(&sys).is_empty());
    }

    #[test]
    fn facts_are_learning_records_linked_to_their_turns() {
        let (sys, _tmp) = system(4);
        let llm = ScriptedLlm::new(|snippets| {
            if snippets.len() < 2 {
                return Vec::new();
            }
            vec![json!({
                "fact": "Nate has had turtles for about three years, as of 21 January 2022.",
                "source_turns": ["T1", "T2"],
                "entities": ["Nate", "turtles"],
            })]
        });
        sys.set_write_time_fact_llm(Some(llm.clone()));
        let question = remember_turn(&sys, "Joanna: How are your turtles?", at(21, 10));
        let answer = remember_turn(&sys, "Nate: I've had them for three years now.", at(21, 11));
        assert!(!sys.write_time_facts().facts_present());

        let report = sys.drain_write_time_facts();
        assert_eq!(report.jobs, 2);
        assert_eq!(
            (report.created, report.failures, report.rejected),
            (1, 0, 0)
        );
        assert_eq!(llm.requests.lock().len(), 2);
        assert!(sys.write_time_facts().facts_present());

        let facts = stored_facts(&sys);
        assert_eq!(facts.len(), 1);
        let fact = &facts[0];
        assert_eq!(report.fact_ids, vec![fact.id.clone()]);
        assert_eq!(fact.kind, RecordKind::Learning);
        assert_eq!(fact.experience.experience_type, ExperienceType::Learning);
        assert_eq!(fact.experience.related_memories, vec![question, answer]);
        assert_eq!(fact.created_at, at(21, 11));
        assert_eq!(
            fact.experience
                .metadata
                .get(META_PROMPT_VERSION)
                .map(String::as_str),
            Some(PROMPT_VERSION)
        );
        assert_eq!(
            fact.experience.metadata.get(META_MODEL).map(String::as_str),
            Some("scripted-1")
        );
        assert!(fact
            .external_id
            .as_deref()
            .is_some_and(|e| e.starts_with(EXTERNAL_ID_PREFIX) && e.ends_with(":0")));
    }

    #[test]
    fn extracting_a_window_again_updates_its_facts_instead_of_adding_to_them() {
        let (sys, _tmp) = system(4);
        let llm = ScriptedLlm::new(|snippets| {
            let target = format!("T{}", snippets.len());
            vec![
                json!({ "fact": "Nate adopted two turtles.", "source_turns": [target], "entities": ["Nate"] }),
                json!({ "fact": "Nate keeps the turtles in a tank.", "source_turns": [target], "entities": ["Nate"] }),
            ]
        });
        sys.set_write_time_fact_llm(Some(llm.clone()));
        let id = remember_turn(
            &sys,
            "Nate: I adopted two turtles, they live in a tank.",
            at(22, 0),
        );
        let job = sys.write_time_facts.state.lock().pending[0].clone();

        let first = sys.drain_write_time_facts();
        assert_eq!(first.created, 2);
        let before = stored_facts(&sys);
        assert_eq!(before.len(), 2);

        // The same window and the same answer: nothing is written.
        sys.write_time_facts.push(job.clone());
        let same = sys.drain_write_time_facts();
        assert_eq!((same.created, same.updated, same.unchanged), (0, 0, 2));
        let after: Vec<MemoryId> = stored_facts(&sys).into_iter().map(|m| m.id).collect();
        assert_eq!(
            after,
            before.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
        );

        // A different answer for the same window: the first fact is rewritten in place
        // and the second, which the model no longer returns, is removed.
        llm.rescript(|_| {
            vec![json!({
                "fact": "Nate adopted two turtles in January 2022.",
                "source_turns": ["T1"],
                "entities": ["Nate", "turtles"],
            })]
        });
        sys.write_time_facts.push(job);
        let changed = sys.drain_write_time_facts();
        assert_eq!(
            (changed.created, changed.updated, changed.removed),
            (0, 1, 1)
        );
        let facts = stored_facts(&sys);
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].id, before[0].id);
        assert_eq!(
            facts[0].experience.content,
            "Nate adopted two turtles in January 2022."
        );
        assert_eq!(facts[0].experience.related_memories, vec![id]);
        assert_eq!(facts[0].kind, RecordKind::Learning);
    }

    #[test]
    fn a_failed_extraction_leaves_the_stored_facts_alone() {
        struct Failing;
        impl LlmConsolidator for Failing {
            fn extract_facts(&self, _: &[String]) -> Result<Vec<LlmExtractedFact>> {
                Ok(Vec::new())
            }
            fn extract_with_schema(&self, _: &[String], _: &str) -> Result<Vec<serde_json::Value>> {
                Err(anyhow::anyhow!("LLM returned non-JSON-array payload"))
            }
        }

        let (sys, _tmp) = system(4);
        sys.set_write_time_fact_llm(Some(ScriptedLlm::new(one_fact_per_turn)));
        remember_turn(&sys, "Nate: I won my first tournament.", at(22, 1));
        let job = sys.write_time_facts.state.lock().pending[0].clone();
        assert_eq!(sys.drain_write_time_facts().created, 1);

        sys.set_write_time_fact_llm(Some(Arc::new(Failing)));
        sys.write_time_facts.push(job);
        let report = sys.drain_write_time_facts();
        assert_eq!((report.failures, report.removed), (1, 0));
        assert_eq!(stored_facts(&sys).len(), 1);
    }

    #[test]
    fn a_held_queue_waits_for_its_holder_to_drain_it() {
        let (sys, _tmp) = system(4);
        sys.set_write_time_fact_llm(Some(ScriptedLlm::new(one_fact_per_turn)));
        let facts = sys.write_time_facts();
        let hold = facts.hold();
        remember_turn(&sys, "Nate: one.", at(22, 2));
        remember_turn(&sys, "Nate: two.", at(22, 3));
        assert_eq!(facts.pending(), 2);
        assert!(!facts.wait_idle(Duration::from_millis(10)));
        let report = sys.drain_write_time_facts();
        drop(hold);
        assert_eq!((report.jobs, report.created), (2, 2));
        assert!(facts.wait_idle(Duration::from_millis(10)));
    }

    #[test]
    fn the_background_worker_drains_the_queue_and_stops_with_the_store() {
        let (sys, _tmp) = system(4);
        sys.set_write_time_fact_llm(Some(ScriptedLlm::new(one_fact_per_turn)));
        let shared = Arc::new(RwLock::new(SystemHandle(sys)));
        start_worker(&shared);
        remember_turn(&shared.read(), "Nate: one.", at(22, 4));
        remember_turn(&shared.read(), "Nate: two.", at(22, 5));
        let facts = shared.read().write_time_facts();
        assert!(facts.wait_idle(Duration::from_secs(60)));
        assert_eq!(stored_facts(&shared.read()).len(), 2);

        drop(shared);
        assert!(facts.state.lock().closed);
    }

    fn id(n: u128) -> MemoryId {
        MemoryId(uuid::Uuid::from_u128(n))
    }

    #[test]
    fn a_ranked_fact_brings_its_source_turns_into_the_pool_after_it() {
        let (fact, a, b, c, s1, s2) = (id(1), id(2), id(3), id(4), id(5), id(6));
        let ranked = vec![
            (fact.clone(), 0.9),
            (a.clone(), 0.8),
            (b.clone(), 0.7),
            (c.clone(), 0.6),
            (s2.clone(), 0.5),
        ];
        let sources = |m: &MemoryId| (*m == fact).then(|| vec![s1.clone(), a.clone(), s2.clone()]);

        let mut expanded = ranked.clone();
        let expansion = expand_fact_sources(&mut expanded, 3, MAX_EXPANSION, sources);
        // s1 was nowhere, s2 sat below the budget, a was already in the pool.
        assert_eq!(
            expansion,
            FactExpansion {
                leg: vec![s1.clone(), a.clone(), s2.clone()],
                injected: 2,
            }
        );
        assert_eq!(
            expanded,
            vec![
                (fact.clone(), 0.9),
                (s1.clone(), 0.9),
                (s2.clone(), 0.9),
                (a.clone(), 0.8),
                (b.clone(), 0.7),
                (c.clone(), 0.6),
            ]
        );

        // The cap holds: s2 keeps its place below the budget.
        let mut capped = ranked.clone();
        let expansion = expand_fact_sources(&mut capped, 3, 1, sources);
        assert_eq!(expansion.injected, 1);
        assert_eq!(capped[1], (s1, 0.9));
        assert_eq!(capped.last(), Some(&(s2, 0.5)));

        // No fact in the budget: nothing moves and the leg is empty.
        let mut untouched = ranked.clone();
        let expansion = expand_fact_sources(&mut untouched, 3, MAX_EXPANSION, |_| None);
        assert_eq!(expansion, FactExpansion::default());
        assert_eq!(untouched, ranked);
    }

    #[test]
    fn recall_expands_a_ranked_fact_into_its_source_turns() {
        let (sys, _tmp) = system(4);
        sys.set_write_time_fact_llm(Some(ScriptedLlm::new(|snippets| {
            if snippets.len() < 2 {
                return Vec::new();
            }
            vec![json!({
                "fact": "Nate has kept pet turtles for about three years, as of 21 January 2022.",
                "source_turns": ["T1", "T2"],
                "entities": ["Nate", "turtles"],
            })]
        })));
        remember_turn(&sys, "Joanna: How are the little shelled ones?", at(21, 20));
        let answer = remember_turn(&sys, "Nate: Great, three years now with them.", at(21, 21));
        sys.drain_write_time_facts();
        let fact = stored_facts(&sys).remove(0);
        assert_eq!(sys.write_time_fact_sources(&answer), None);
        assert!(sys
            .write_time_fact_sources(&fact.id)
            .is_some_and(|s| s.contains(&answer)));

        let query = crate::memory::types::Query {
            query_text: Some("How long has Nate kept pet turtles?".to_string()),
            max_results: 10,
            ..Default::default()
        };
        let fact_stage = |sys: &MemorySystem| {
            crate::memory::gold_funnel::begin(HashSet::from([answer.clone()]));
            let results = sys.recall(&query).expect("recall");
            let stages = crate::memory::gold_funnel::take().unwrap_or_default();
            let stage = stages
                .into_iter()
                .find(|(name, _)| name == "fact")
                .map(|(_, rank)| rank);
            (results, stage)
        };

        let (results, stage) = fact_stage(&sys);
        assert!(
            results.iter().any(|m| m.id == fact.id),
            "the fact itself stays in the results"
        );
        assert_eq!(
            stage,
            Some(Some(1)),
            "the answer turn is the fact's second source"
        );

        sys.set_cognitive_dials(CognitiveDials {
            write_time_facts: true,
            write_time_fact_expansion: false,
            ..Default::default()
        });
        let (_, stage) = fact_stage(&sys);
        assert_eq!(stage, None, "no fact leg with the dial off");
    }

    /// `start_worker` takes what the server shares, a lock around something that derefs
    /// to the store.
    struct SystemHandle(MemorySystem);

    impl Deref for SystemHandle {
        type Target = MemorySystem;
        fn deref(&self) -> &MemorySystem {
            &self.0
        }
    }
}
