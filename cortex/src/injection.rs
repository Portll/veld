//! Injection Module - Inject activated memories into Claude's context
//!
//! This module takes the activated memories and injects them into
//! the system prompt, making them part of Claude's working memory.

use crate::types::{ClaudeRequest, SurfacedMemory, SystemBlock, SystemContent};

/// Inject memories into the Claude request
///
/// Memories are added as a special block in the system prompt,
/// making them available to Claude without modifying the user's messages.
pub fn inject_memories(request: &mut ClaudeRequest, memories: &[SurfacedMemory]) {
    if memories.is_empty() {
        return;
    }

    // Format memories for injection
    let memory_block = format_memory_block(memories);

    // Inject into system prompt
    request.system = Some(match request.system.take() {
        Some(existing) => {
            SystemContent::Blocks(existing.into_blocks_with_injection(&memory_block))
        }
        None => SystemContent::Blocks(vec![SystemBlock::Text {
            text: memory_block,
            cache_control: None,
        }]),
    });
}

/// Format memories into an injection block
fn format_memory_block(memories: &[SurfacedMemory]) -> String {
    let formatted: Vec<String> = memories
        .iter()
        .enumerate()
        .map(|(i, m)| {
            format!(
                "[{}] ({}) {:.0}%: {}",
                i + 1,
                m.memory_type,
                m.score * 100.0,
                m.content
            )
        })
        .collect();

    format!(
        r#"<shodh-context relevance="proactive">
The following memories from past interactions may be relevant:

{}

Use these memories to provide contextual, personalized responses.
If a memory contradicts the current request, prioritize the user's current intent.
</shodh-context>"#,
        formatted.join("\n")
    )
}

/// Format memories with metadata for detailed injection
#[allow(dead_code)]
pub fn format_memory_block_detailed(memories: &[SurfacedMemory]) -> String {
    let formatted: Vec<String> = memories
        .iter()
        .enumerate()
        .map(|(i, m)| {
            format!(
                r#"  <memory index="{}" type="{}" relevance="{:.0}%">
    {}
  </memory>"#,
                i + 1,
                m.memory_type,
                m.score * 100.0,
                m.content
            )
        })
        .collect();

    format!(
        r#"<shodh-context relevance="proactive" count="{}">
{}
</shodh-context>"#,
        memories.len(),
        formatted.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_memory_block() {
        let memories = vec![
            SurfacedMemory {
                id: "1".to_string(),
                content: "User prefers Rust".to_string(),
                memory_type: "Learning".to_string(),
                score: 0.85,
            },
            SurfacedMemory {
                id: "2".to_string(),
                content: "Working on shodh-memory".to_string(),
                memory_type: "Context".to_string(),
                score: 0.72,
            },
        ];

        let block = format_memory_block(&memories);
        assert!(block.contains("shodh-context"));
        assert!(block.contains("User prefers Rust"));
        assert!(block.contains("85%"));
    }

    #[test]
    fn test_inject_into_empty_system() {
        use std::collections::HashMap;

        let mut request = ClaudeRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![],
            system: None,
            max_tokens: 1000,
            stream: false,
            tools: vec![],
            extra: HashMap::new(),
        };

        let memories = vec![SurfacedMemory {
            id: "1".to_string(),
            content: "Test memory".to_string(),
            memory_type: "Context".to_string(),
            score: 0.9,
        }];

        inject_memories(&mut request, &memories);

        assert!(request.system.is_some());
        let text = request.system.unwrap().as_text();
        assert!(text.contains("Test memory"));
    }

    #[test]
    fn test_inject_into_existing_system() {
        use std::collections::HashMap;

        let mut request = ClaudeRequest {
            model: "claude-3-opus".to_string(),
            messages: vec![],
            system: Some(SystemContent::Text("You are a helpful assistant.".to_string())),
            max_tokens: 1000,
            stream: false,
            tools: vec![],
            extra: HashMap::new(),
        };

        let memories = vec![SurfacedMemory {
            id: "1".to_string(),
            content: "Test memory".to_string(),
            memory_type: "Context".to_string(),
            score: 0.9,
        }];

        inject_memories(&mut request, &memories);

        let text = request.system.unwrap().as_text();
        assert!(text.contains("helpful assistant"));
        assert!(text.contains("Test memory"));
    }
}
