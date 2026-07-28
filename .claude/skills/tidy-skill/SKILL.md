---
name: tidy-skill
description: "Repo artifact governance & workspace hygiene. MANDATORY for: any file generation (Artifact Intent Check), task wrap-up (End-of-Task Checklist), auditing agent-created files, hygiene scoring. Full skill: /home/ubuntu/cluster/items/skills/devops/tidy-skill/SKILL.md"
---

# tidy-skill (洁癖.skill)

Full reference: `/home/ubuntu/cluster/items/skills/devops/tidy-skill/SKILL.md`
Scripts: `/home/ubuntu/cluster/items/skills/devops/tidy-skill/scripts/`

## When to Invoke (triggers)
整理项目 / clean up repo / 生成报告 / audit / 清理 Agent 文件 / hygiene score / 不要生成垃圾文档 / 任务结束前 / 任何要写文件之前

## Artifact Intent Check (MANDATORY before creating any file)
Fill this before writing ANY file. If any field is blank → don't create, answer in chat.
1. User requested a file? yes/no
2. Purpose:
3. Reader:
4. Expected lifetime: session / days / persistent / formal-doc
5. Destination path:
6. Why chat is not enough:
7. Class: A/B/C/D/E
8. Should be in .gitignore? yes/no

## 5 Classes
- A (Formal docs: README, docs/) → never delete
- B (User-requested deliverables) → .agent_reports/<task>_<YYYY-MM-DD>.md
- C (Temp working files: plan, todo) → .agent_tmp/ only
- D (Self-congratulatory: summary, final_report) → DO NOT CREATE
- E (Tool state: .claude/, .cursor/) → never touch

## Hard Rules
- Project root: FORBIDDEN for generic process Markdown
- Default: answer in chat, not files
- Forbidden filenames in root: todo.md, plan.md, summary.md, report.md, *_plan.md, *_report.md, *_summary.md

## End-of-Task Checklist
Before "done":
1. Did I create files? Each justified by Intent Check?
2. Any files in root that shouldn't be?
3. Any .agent_tmp/ safe to delete?
4. Did I avoid summary.md / final_report.md?
