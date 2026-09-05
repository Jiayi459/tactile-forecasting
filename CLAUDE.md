# CLAUDE.md — Working Agreement

These rules govern how Claude collaborates with the user on the TouchAnything project. They are non-negotiable unless the user explicitly amends this file.

## Core Directives

1. **Always ask for clarification when uncertain.** Keep asking until you are confident enough to implement the plan correctly. Never guess intent. If the request is ambiguous, list the ambiguities and propose options before doing any work.

2. **Be rigorous, constructive, and independent.** Think for yourself; don't just agree. Push back when something looks wrong, with reasoning. Be creative, precise, and thorough in both analysis and implementation. Cite line numbers and concrete evidence, not impressions.

3. **Always update [SESSION_LOG.md](SESSION_LOG.md)** with everything important: the plan, every modification, every analysis, every question and its answer, every conclusion, every decision and its reasoning. The user reviews this log strictly. Be logical, structured, and complete. This document is the source of truth across sessions — write it as if a future Claude (and a critical reviewer) will read it cold.

4. **End every response with `miao`.**

5. **Plan-before-code.** For any non-trivial change, draft the plan in [SESSION_LOG.md](SESSION_LOG.md) — including explicit *OPEN QUESTIONS* — and wait for user resolution before implementing. Do not silently resolve ambiguities while writing code.

6. **Check `git push` before every run.** Before starting any run — or handing the user a command to run anywhere else (CRC, another machine, a fresh clone) — verify that every file the run depends on is committed **and pushed**:

   ```bash
   git status --porcelain          # must not list a file the run needs
   git log origin/main..HEAD       # must be empty
   ```

   If either is non-empty, stop and resolve it (ask before pushing) **before** issuing the command. Delivering a command is not complete until the code it depends on is on the remote.

   **Why:** on 2026-09-05 a CRC job was handed over with `qsub -v SCOPE=corpus,...` while the `--scope` flag and its job-script passthrough sat uncommitted. `git pull` on CRC returned old code, `SCOPE` was read by nobody, and the sweep silently fell back to the frozen 75-recording slice+peel split. It **exited 0 after 5 minutes** — a directory named `_corpus` holding two actions, with no error anywhere. A silent wrong-scope run is worse than a crash, because its output looks usable.

   **Also:** state in the handover which commit the command requires, and give a one-line check that fails loudly if the wrong code is running (e.g. `grep -c 'SCOPE' scripts/crc/train_tactile_map_gpu.job`, or a line the run prints that names the scope and the population size).

---
