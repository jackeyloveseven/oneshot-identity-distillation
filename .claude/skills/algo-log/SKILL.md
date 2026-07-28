---
name: algo-log
description: "ML micro-journal for experiments, decisions, pitfalls. Use when: logging exp results / decisions / pitfalls, generating weekly summaries, project closure reports, highlight reels. Log path: experiment-log/<YYYY-MM>/log.jsonl"
---

# algo-log — 算法工程师实时微日志

Full reference: `/home/ubuntu/cluster/items/skills/mlops/algo-log/SKILL.md`
Scripts: `/home/ubuntu/cluster/items/skills/mlops/algo-log/scripts/`
Templates: `/home/ubuntu/cluster/items/skills/mlops/algo-log/templates/`

## Triggers
log exp / log decision / log pitfall / 本周摘要 / 项目结项 <project> / 我的亮点 / 记一下 / 踩坑了

## Usage — Say it casually, I'll structure it
log exp：DPO阶段2完成，CLIP-I 0.78，编辑精度退化3%
log decision：放弃 FireRed 用 Qwen Edit，免适配
log pitfall：A100 DPO OOM，gradient checkpointing 没开

## 4 Types Schema (required fields)
- exp: ts, type, project, title, summary, metrics{value,baseline,delta}, artifacts
- decision: ts, type, project, title, summary, why, alternatives, impact
- pitfall: ts, type, project, title, summary, root_cause, solution, time_cost, prevention
- note: ts, type, project, title, summary, source

## Log Path
experiment-log/<YYYY-MM>/log.jsonl (relative to project root, append JSONL)

## Aggregation Commands
- 「本周摘要」→ STAR format by project, covers Mon–Sun
- 「项目结项 <project>」→ 6-module report → lark-doc (飞书文档, NOT local file)
- 「我的亮点 [时间范围]」→ decisions + pitfalls only, 3-section format

## File Rules (tidy-skill)
- log.jsonl writes → Class A, no Intent Check needed
- Weekly/highlights saved → .agent_reports/ only if user asks
- Project closure → lark-doc, NEVER local Markdown
- Root → FORBIDDEN
