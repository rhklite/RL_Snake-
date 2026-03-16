Run the RL pre-flight verification before launching a training run.

Read the skill at `~/.cursor/skills/rl-preflight/SKILL.md` and follow its steps. The skill will:

1. Read hypothesis.md to understand design intent
2. Diff the code to find implementation changes
3. Automatically verify implementation matches intent (PASS/MISMATCH report)
4. Generate a Gemini handoff prompt for cross-family critique
5. Display a workflow transition banner with next steps
