- production grade code only, no TODOs, no placeholders, no mocks, no stubs
- understand context, take a bird's eye view, understand architecture and data flow before fixing anything
- dont build, dont trunk serve, I will run it in background
- DO NOT add "Generated with Claude Code" signature or "Co-Authored-By" lines to git commits - clean commit messages only

# Memory Usage
- At session start, call proactive_context to surface relevant memories
- Check list_todos to see pending work and continue where you left off
- Use recall before answering questions about past work, decisions, or patterns
- Use remember to store important decisions, learnings, and context worth preserving
- Use add_todo to track work that spans sessions
- Complete todos (complete_todo) as you finish them
