"""
Veld - Agentic Memory + CrewAI Integration Example

An account-management crew that keeps what it learns about clients in veld.
Run it twice: the first run stores the client's preferences and the agent's
result; the second run recalls them before the task, without being told.

Requirements:
    pip install "veld[crewai]"

Setup:
    1. Start a veld server: veld server   (listens on http://127.0.0.1:3030)
    2. export VELD_API_KEY=<a key the server accepts>
    3. export OPENAI_API_KEY=<key>, or set CREWAI_LLM to another CrewAI model string
    4. python veld_crewai.py
"""

import os

from crewai import Agent, Crew, Task

from veld.integrations.crewai import VeldCrewMemory


def main() -> None:
    memory = VeldCrewMemory(
        server_url=os.environ.get("VELD_API_URL", "http://127.0.0.1:3030"),
        user_id=os.environ.get("VELD_USER_ID", "crewai-example"),
        # Everything this crew stores lands under this scope, and its recalls stay inside it.
        root_scope="/crew/account-team",
        memory_type="Learning",
        # Veld records every memory written over its HTTP API as untrusted, this
        # crew's own included, and recall withholds untrusted memories by default.
        # This crew reads back what it wrote, so it opts in; each recalled memory
        # reaches the agent's prompt labelled "source: untrusted (AiGenerated)".
        trusted_only=False,
    )

    # Facts can be stored directly, outside any task.
    memory.remember(
        "Acme Corp wants a one-page status report every Friday, with risks listed first.",
        scope="client/acme",
        categories=["preferences"],
    )

    llm = os.environ.get("CREWAI_LLM", "gpt-5.4-mini")
    manager = Agent(
        role="Account manager",
        goal="Write client updates that match what each client has asked for",
        backstory="You manage a portfolio of enterprise clients and keep their preferences in mind.",
        llm=llm,
    )
    update = Task(
        description="Draft this week's status update for Acme Corp. The project is on track; "
        "the data migration slipped two days.",
        expected_output="A status update in the format Acme Corp asked for.",
        agent=manager,
    )

    # With memory set, the agent recalls from veld before the task and saves its
    # result after it. The crew also gets "Search memory" and "Save to memory" tools.
    crew = Crew(agents=[manager], tasks=[update], memory=memory)
    result = crew.kickoff()
    print(result.raw)

    print("\nWhat the crew can recall about Acme:")
    for match in memory.recall("Acme Corp reporting preferences", limit=5):
        print(match.format())

    memory.close()


if __name__ == "__main__":
    main()
