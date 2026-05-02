---
on:
  issues:
    types: [labeled]
    names: ["todo"]
  roles: [admin, maintainer, write]

permissions:
  contents: read
  issues: read
  pull-requests: read

engine: copilot
timeout-minutes: 5

tools:
  github:
    allowed: [issue_read]
  edit:
  bash: [grep, cat, awk, sed]

safe-outputs:
  create-pull-request:
    title-prefix: "[todo] "
    labels: [automation, todo-list]
    draft: false
---

# Append Labeled Todo Issue to Checklist

A repo contributor has applied the `todo` label to issue
**#${{ github.event.issue.number }}** — `${{ github.event.issue.title }}`.

Update `.copilot/todo.md` so the checklist mirrors the new label.

## Steps

1. Use the `issue_read` tool to fetch issue **#${{ github.event.issue.number }}** in repository `${{ github.repository }}`. From the response, capture:
   - `number` (the issue number — used only for the hidden idempotency marker and the PR `Closes` keyword)
   - `title`
   - `body` (the issue description; trim and treat empty/whitespace as "no summary")

2. Read `.copilot/todo.md`. If the file is missing, create it from this template:

   ````markdown
   # Todo

   Auto-managed checklist of issues labeled `todo`.

   ## Open

   <!-- todo-list:start -->
   <!-- todo-list:end -->
   ````

3. **Idempotency:** if the file already contains the hidden marker `<!-- issue:<number> -->`, make no changes and stop. The PR step will be skipped because the diff is empty.

4. Otherwise, insert a new entry immediately before the `<!-- todo-list:end -->` marker. Do not modify any other line, do not reorder existing entries, do not touch sections outside the markers.

   The first line is always:

   ```
   - [ ] **<title>** <!-- issue:<number> -->
   ```

   If — and only if — the issue body contains non-whitespace text, append a second line directly underneath:

   ```
     <summary>
   ```

   where `<summary>` is the issue body collapsed to a single line (replace newlines with spaces, trim whitespace). If the issue body is empty, missing, or whitespace-only, **do not** add the summary line and **do not** add a blank/indented placeholder line — the entry is exactly one line.

5. Commit only `.copilot/todo.md`.
   - PR **title**: `Add todo: <title>`
   - PR **body** must include the line `Closes #<number>` on its own line so merging the PR closes the source issue. Keep the rest of the body brief.
