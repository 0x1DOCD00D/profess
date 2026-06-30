---
name: profess-architecture
description: This skill should be used when designing PROFESS framework features, building a DSL on top of PROFESS, evaluating whether code belongs in the framework or in the DSL author's domain, or when working on files under runtime/, plugin/, or sentences/. Also use when the user asks "where does X belong", "should this be in the framework", or "how do I build a DSL with PROFESS".
version: 0.2.0
---

# PROFESS Architecture

PROFESS has a strict two-party split: the **framework** and the **DSL author**. Every design decision should be evaluated against this split.

## The Split

### Framework owns (lives in runtime/, plugin/)
- The compiler plugin (`ProfessPhase`, `FessCallCollector`, `FessIrLowering`, scaffolding)
- The flat IR node types (`IRNode` and subtypes in `Runtime.scala`)
- The **structuring layer** (`ClauseBuilder`) — the vocabulary-free grammar pass that turns a flat token stream into roled clauses (subject/head/args/modifiers)
- The handler abstractions (`ProfessInterpreter`, `HandlerRegistry`, `IRTraverser`, `RelationHandler`)
- The traversal engine (how IR nodes are walked and dispatched)
- The effect plumbing (`CatsInterpreter`, `runWithHandlers`, `executeWithHandlers`)
- `ProfessExpr` and its `Dynamic` chaining

### DSL author owns (lives in sentences/, examples/, or user code)
- The sentence strings (what goes inside `FESS("...")` or `.profess` files)
- A state type with optional fields (whatever the DSL needs to accumulate)
- A `Monoid` instance for that state type
- The handlers themselves — functions from IR nodes to state updates

### Structuring is framework; interpretation is the author
The framework owns **structuring** — the grammar that maps flat nodes into roled clauses (subject-of, arg-of, modified-by edges). The DSL author owns **interpretation** — what `sold` *means*. Structuring is grammar, which is framework; interpretation is domain, which is the author.

This consciously overturns the earlier "no normalizer layer" rule. The reason (Grechanik, 2026-06-06): *edges can't carry meaning without a structuring pass to produce them.* You cannot key a handler on "modified-by(at)" until something has produced that edge from the flat token stream — that is the `ClauseBuilder`'s job.

Structuring is **not** validation. The structuring pass rejects nothing — every sentence still yields a clause — so the "every sentence is valid" invariant holds. What is still forbidden is a *validation/conformity* pass that rejects or requires fields; that remains a DSL-author (handler) concern, not a framework layer. See [[feedback-avoid-early-validation]].

## The Execution Sequence

```
FessIrLowering  →  ClauseBuilder  →  Clause  →  RelationHandler dispatch  →  partial state  →  consumer
```

1. `ClauseBuilder` (framework) turns the flat `IRSequence` into a roled `Clause`: first `IRWord` is the head relation; each later `IRWord` opens a modifier phrase; fillers are the args
2. DSL author registers handlers — `RelationHandler` keyed by relation (head word), `ObjectHandler` keyed by kind, optional `ModifierHandler` keyed by preposition
3. A `RelationHandler` receives the `Clause` by role plus a recursive `interp: IRNode => F[A]` callback, so the author chooses what to recurse into (flat dispatch *or* bottom-up catamorphism)
4. Handlers accumulate into the DSL author's state type; missing handlers contribute the monoid empty (partial state, not error)
5. Consumer receives the final accumulated state

## Key Design Rules

**Rule 1: If it's domain-specific, it belongs to the DSL author.**
The framework has no opinion about what sentences mean. A handler that interprets `IRWord("Dog")` as an animal is DSL-author code, not framework code.

**Rule 2: The framework abstracts traversal, not interpretation.**
`IRTraverser` knows how to walk an IR tree. It does not know what to do with any particular node. That knowledge lives in handlers.

**Rule 3: New framework abstractions are extracted from real DSLs, not invented.**
Do not add framework machinery speculatively. See `profess-framework-evolution` skill.

**Rule 4: The `ProfessExpr` Dynamic chain is for sentence construction only.**
`ProfessExpr.selectDynamic` and `applyDynamic` build IR during sentence evaluation. They are not a general-purpose Scala DSL tool.

## File Map

| File | Owner | Purpose |
|------|-------|---------|
| `runtime/src/main/scala/profess/runtime/Runtime.scala` | Framework | IR nodes, ProfessExpr, traversal base |
| `runtime/src/main/scala/profess/runtime/effects/CatsInterpreter.scala` | Framework | Handler registry, traverser, effect plumbing |
| `plugin/src/main/scala/profess/plugin/` | Framework | Compiler plugin, scaffolding, FESS lowering |
| `sentences/src/main/profess/` | DSL author | Sentence source files (.profess) |
| `sentences/src/main/scala/profess/sentences/` | DSL author | Handlers, state types, Monoid instances |
| `examples/` | DSL author | Demonstration DSLs |

## Common Mistakes

**Putting domain logic in the IR** — IR nodes are structural (`IRWord`, `IRObject`, `IRSequence`). Adding an `IRAnimal` or `IRTemperature` node is a DSL-author concern expressed through handlers, not a new IR node type.

**Putting handler logic in the framework** — If a piece of code only makes sense for one specific DSL, it does not belong in `runtime/` or `plugin/`.

**Confusing structuring with validation** — The `ClauseBuilder` structuring pass (flat nodes → roled clause) IS a framework layer and IS allowed; it produces the edges handlers key on and rejects nothing. A *validation/normalization* pass (rejecting sentences, requiring fields, conformity checks) is NOT a framework layer — that stays a handler responsibility, defined after real DSLs surface the pattern.
