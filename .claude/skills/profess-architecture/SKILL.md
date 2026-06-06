---
name: profess-architecture
description: This skill should be used when designing PROFESS framework features, building a DSL on top of PROFESS, evaluating whether code belongs in the framework or in the DSL author's domain, or when working on files under runtime/, plugin/, or sentences/. Also use when the user asks "where does X belong", "should this be in the framework", or "how do I build a DSL with PROFESS".
version: 0.1.0
---

# PROFESS Architecture

PROFESS has a strict two-party split: the **framework** and the **DSL author**. Every design decision should be evaluated against this split.

## The Split

### Framework owns (lives in runtime/, plugin/)
- The compiler plugin (`ProfessPhase`, `FessCallCollector`, `FessIrLowering`, scaffolding)
- The flat IR node types (`IRNode` and subtypes in `Runtime.scala`)
- The handler abstractions (`ProfessInterpreter`, `HandlerRegistry`, `IRTraverser`)
- The traversal engine (how IR nodes are walked and dispatched)
- The effect plumbing (`CatsInterpreter`, `runWithHandlers`, `executeWithHandlers`)
- `ProfessExpr` and its `Dynamic` chaining

### DSL author owns (lives in sentences/, examples/, or user code)
- The sentence strings (what goes inside `FESS("...")` or `.profess` files)
- A state type with optional fields (whatever the DSL needs to accumulate)
- A `Monoid` instance for that state type
- The handlers themselves — functions from IR nodes to state updates

### No normalizer layer
There is no intermediate "normalizer" or "semantic enrichment" layer between the IR and the handlers. The IR is a faithful recording of sentence structure. Interpretation is entirely the DSL author's responsibility via handlers.

## The Execution Sequence

```
HandlerRegistry  →  IRTraverser  →  partial state  →  consumer
```

1. DSL author registers handlers with `HandlerRegistry`
2. `IRTraverser` walks flat IR nodes and dispatches to matching handlers
3. Handlers accumulate into the DSL author's state type (via `Monoid`)
4. Consumer receives the final accumulated state

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

**Adding a validation/normalization pass** — There is no pre-handler normalization layer. If sentences need validation, that is a handler responsibility.
