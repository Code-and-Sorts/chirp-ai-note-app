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
  bash: [grep, cat, awk, sed, git]

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
   - `number` (used **only** for the PR `Closes` keyword so merging the PR closes the source issue — never written into `.copilot/todo.md`)
   - `title`
   - `body` (the issue description; trim and treat empty/whitespace as "no summary")

   **Sanitize both `title` and `body` before using them as `<title>` and `<summary>` below.** The issue text is untrusted: a malicious or careless author could embed strings that break the file structure. Apply these transforms in order:
   - Strip any HTML comments matching `<!--...-->` entirely (this prevents an author from injecting fake `<!-- todo-list:end -->` anchors).
   - Replace any remaining `<` with `&lt;` and `>` with `&gt;`.
   - Collapse whitespace runs to a single space; do not allow embedded newlines.

2. **Truncate the sanitized `body`** to keep the entry small enough to fit comfortably under the safe-outputs `max_patch_size` limit (~1024 bytes per patch):
   - Take only the **first paragraph** (everything up to the first blank line in the original body).
   - Then truncate that paragraph to **300 characters max**, ending on a word boundary, and append `…` if any text was cut.
   - The result is the `<summary>` value used below.

3. Read `.copilot/todo.md`. If the file is missing, create it from this template:

   ````markdown
   # Todo

   <!--
   Auto-managed checklist of issues labeled `todo`.
   -->

   ## Open

   <!-- todo-list:start -->
   <!-- todo-list:end -->
   ````

4. **Idempotency:** between the `<!-- todo-list:start -->` and `<!-- todo-list:end -->` markers, look for any existing line of the form `- [ ] **<title>**` (after the same sanitization applied in step 1) where the `<title>` matches the issue's sanitized title **exactly**. If one is found, make no changes and stop. The PR step will be skipped because the diff is empty.

5. Otherwise, insert a new entry immediately before the `<!-- todo-list:end -->` marker. Do not modify any other line, do not reorder existing entries, do not touch sections outside the markers.

   The first line is always:

   ```
   - [ ] **<title>**
   ```

   The issue number must **not** appear anywhere in `.copilot/todo.md` — neither visibly nor inside an HTML comment.

   If — and only if — the (sanitized, truncated) `<summary>` from step 2 is non-empty, append a second line directly underneath:

   ```
     <summary>
   ```

   If `<summary>` is empty, **do not** add the summary line and **do not** add a blank/indented placeholder line — the entry is exactly one line.

6. **Single-file change guard:** run `bash` to confirm the only modified path is `.copilot/todo.md`:

   ```
   git diff --name-only
   ```

   The output must be exactly `.copilot/todo.md` (or empty in the idempotent case). If any other path appears, **revert** that change with `git checkout -- <path>` before producing the PR. Do not edit any file outside `.copilot/todo.md` under any circumstance.

7. Commit only `.copilot/todo.md`.
   - PR **title**: `Add todo: <title>`
   - PR **body** must include the line `Closes #<number>` on its own line so merging the PR closes the source issue. Keep the rest of the body brief.
