# PROFESS Examples Guide

**Comprehensive Tutorial for Cats Effect Integration and π-Calculus Translation**

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Cats Effect Examples](#cats-effect-examples)
    - [Trading Example with StateT](#trading-example-with-statet)
    - [Logging Effects Example](#logging-effects-example)
    - [Given-Based Handler Injection](#given-based-handler-injection)
    - [Composable Domain Handlers](#composable-domain-handlers)
    - [Code Generation Example](#code-generation-example)
4. [π-Calculus Translation Example](#π-calculus-translation-example)
    - [IR Design Decisions](#ir-design-decisions)
    - [Optimization Passes](#optimization-passes)
    - [Akka Code Generation](#akka-code-generation)
5. [Generalization Patterns](#generalization-patterns)

---

## Overview

This guide covers two advanced PROFESS example sets:

| Example Set | Purpose | Key Techniques |
|-------------|---------|----------------|
| **Cats Effect** | Effect-based interpretation of trading DSL | StateT, Handlers, Monoids, Given injection |
| **π-Calculus** | Process calculus to Akka actor translation | Custom IR, Algebraic optimizations, Code generation |

Both demonstrate how PROFESS's **deferred semantics** enable:
- Multiple interpretations of the same expression
- Algebraic transformations before execution
- Target-independent DSL design

---

## Design Philosophy

### Handler-Based Architecture

PROFESS uses a **handler pattern** for interpretation:

```
ProfessExpr → IR → Handlers → F[A]
```

**Design Decisions**:

1. **Handlers are values, not inheritance** — Handlers are first-class objects that can be composed, replaced, and injected via givens

2. **Effect polymorphism via F[_]** — Handlers produce effects in any monad (IO, StateT, Either, etc.)

3. **Monoid combination** — Results combine via `Monoid[A].combine`, enabling parallel and sequential composition

4. **Context threading** — `TransformContext` carries state without mutability

### Computation Model

```scala
// The core computation pattern
trait WordHandler[F[_], A]:
  def word: String
  def handle(args: List[IRNode], ctx: TransformContext): F[A]
```

**Why this design?**

| Decision | Rationale |
|----------|-----------|
| `F[_]` effect type | Supports pure (Id), IO, StateT, concurrent effects |
| `List[IRNode]` args | Words consume following IR nodes as arguments |
| `TransformContext` | Thread-safe state without mutable variables |
| Return `F[A]` | Composable with for-comprehensions, error handling |

---

## Cats Effect Examples

Located in: `examples/src/main/scala/examples/catseffect/CatsExamples.scala`

### Trading Example with StateT

This example demonstrates **stateful interpretation** using Cats StateT.

#### Domain Model

```scala
case class Trade(
  broker: String,
  action: String,
  quantity: Int,
  instrument: String,
  price: Option[Double] = None
)

case class TradingState(
  trades: List[Trade] = Nil,
  currentBroker: Option[String] = None,
  currentAction: Option[String] = None,
  currentQuantity: Option[Int] = None,
  currentInstrument: Option[String] = None,
  currentPrice: Option[Double] = None
)
```

**Design Decisions**:

| Component | Purpose |
|-----------|---------|
| `Trade` | Completed trade record (immutable) |
| `TradingState` | Accumulator for in-progress trade construction |
| `Option` fields | Represent partial information during parsing |
| `completeTrade` | Finalizes trade when all fields are present |

#### Effect Type

```scala
type TradingF[A] = StateT[IO, TradingState, A]
```

**Why StateT[IO, TradingState, A]?**

- **StateT** — Threads `TradingState` through computations without mutation
- **IO** — Base effect for potential async operations
- **TradingState** — The state being threaded
- **A** — Result type (Unit for handlers)

#### Object Handlers

```scala
val brokerHandler = ObjectHandler[TradingF, Unit]("broker") { (name, ctx) =>
  StateT.modify[IO, TradingState](_.withBroker(name))
}

val stockHandler = ObjectHandler[TradingF, Unit]("stock") { (name, ctx) =>
  StateT.modify[IO, TradingState](_.withInstrument(name))
}
```

**Analysis**:

| Part | Explanation |
|------|-------------|
| `ObjectHandler[TradingF, Unit]("broker")` | Creates handler for `(broker Name)` patterns |
| `(name, ctx) => ...` | Handler function receives entity name and context |
| `StateT.modify` | Pure state transformation, no side effects |
| `_.withBroker(name)` | State update using copy method |

#### Word Handlers

```scala
val soldHandler = WordHandler[TradingF, Unit]("sold") { (args, ctx) =>
  StateT.modify[IO, TradingState] { state =>
    val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
    state.withAction("sold").withQuantity(qty)
  }
}
```

**Analysis**:

| Part | Explanation |
|------|-------------|
| `WordHandler[TradingF, Unit]("sold")` | Handler for word "sold" |
| `args: List[IRNode]` | IR nodes following the word (e.g., quantity) |
| `args.collectFirst { case IRNumber(n) => n.toInt }` | Extract first number from args |
| `.withAction("sold").withQuantity(qty)` | Chain state updates |

#### Argument Extraction Pattern

```scala
val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
val price = args.collectFirst { case IRNumber(n) => n }.getOrElse(0.0)
```

**Pattern**: Use `collectFirst` with pattern matching to extract typed values from IR nodes.

**Why this approach?**
- Type-safe extraction from `List[IRNode]`
- Graceful handling of missing values with `getOrElse`
- Supports multiple value types in same expression

#### Default Handler

```scala
val defaultWord: (String, List[IRNode], TransformContext) => TradingF[Unit] = 
  (word, args, ctx) => StateT.modify[IO, TradingState](_.completeTrade)
```

**Purpose**: Handles unknown words by attempting to complete the current trade.

**Signature breakdown**:
- `String` — The unhandled word
- `List[IRNode]` — Following arguments
- `TransformContext` — Current context
- Returns `TradingF[Unit]` — State modification

#### Registry Assembly

```scala
val registry: HandlerRegistry[TradingF, Unit] =
  HandlerRegistry[TradingF, Unit](
    wordHandlers = List(soldHandler, boughtHandler, atHandler),
    objectHandlers = List(brokerHandler, stockHandler)
  ).withDefaultWordHandler(defaultWord)
```

**Design Pattern**: Builder pattern for registry construction.

**Composition**:
1. Create registry with explicit handler lists
2. Chain `.withDefaultWordHandler()` for fallback behavior
3. Registry is immutable — each `with*` returns new registry

#### Interpretation

```scala
def interpret(expr: ProfessExpr): IO[List[Trade]] =
  val traverser = IRTraverser[TradingF, Unit](registry)
  for
    (finalState, _) <- traverser.traverse(expr).run(TradingState())
  yield finalState.completeTrade.trades
```

**Execution Flow**:

```
ProfessExpr
    │
    ▼ toIR (implicit)
IRSequence([IRObject("broker","Mark"), IRWord("sold"), IRNumber(700), ...])
    │
    ▼ IRTraverser.traverse
StateT[IO, TradingState, Unit]
    │
    ▼ .run(TradingState())
IO[(TradingState, Unit)]
    │
    ▼ map to extract trades
IO[List[Trade]]
```

---

### Logging Effects Example

Demonstrates **effect-producing handlers** that generate executable operations.

#### Registry with HandlerDSL

```scala
val registry: HandlerRegistry[IO, Executable[Unit]] =
  HandlerDSL.handlers[IO, Executable[Unit]]
    .onObject("broker") { (name, ctx) =>
      IO.pure(Executable.Effect(IO.println(s"[BROKER] $name entered the trade")))
    }
    .onWord("sold") { (args, ctx) =>
      val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
      IO.pure(Executable.Effect(IO.println(s"[ACTION] Selling $qty units")))
    }
    .onAnyWord { (word, args, ctx) =>
      IO.pure(Executable.Effect(IO.println(s"[WORD] $word")))
    }
    .build
```

**Key Differences from StateT Example**:

| Aspect | StateT Example | Logging Example |
|--------|----------------|-----------------|
| Result type | `Unit` | `Executable[Unit]` |
| State | Threaded via StateT | None |
| Effect | State modification | IO effect wrapped in Executable |
| Combination | State accumulation | Monoid combination |

#### Executable ADT

```scala
sealed trait Executable[+A]

object Executable:
  case object NoOp extends Executable[Nothing]
  case class Pure[A](value: A) extends Executable[A]
  case class Effect[A](run: IO[A]) extends Executable[A]
  case class Sequence[A](steps: List[Executable[A]]) extends Executable[A]
```

**Purpose**: Represents deferred effects that can be combined before execution.

**Why wrap IO in Executable?**
- Enables Monoid combination of effects
- Separates effect description from execution
- Allows optimization before running

#### Execution

```scala
def run(expr: ProfessExpr): IO[Unit] =
  given Unit = ()
  EffectInterpreter[Unit].runIO(expr, registry)
```

**Note**: `given Unit = ()` provides default value for `NoOp` cases.

---

### Given-Based Handler Injection

Demonstrates **typeclass-style handler injection** using Scala 3 givens.

#### Domain Definition

```scala
object TradingDomain:
  case class TradeAction(description: String)
  
  given Monoid[TradeAction] with
    def empty: TradeAction = TradeAction("")
    def combine(x: TradeAction, y: TradeAction): TradeAction =
      TradeAction(s"${x.description}; ${y.description}".stripPrefix("; ").stripSuffix("; "))
```

**Design Pattern**: Domain-specific result type with custom combination.

**Monoid Laws Applied**:
- `empty` — Empty description
- `combine` — Semicolon-separated concatenation
- Trimming handles edge cases cleanly

#### Handler as Given Instance

```scala
given DomainHandlers[IO, TradeAction] with
  val registry: HandlerRegistry[IO, TradeAction] =
    HandlerDSL.handlers[IO, TradeAction]
      .onObject("broker") { (name, ctx) =>
        IO.pure(TradeAction(s"Broker $name"))
      }
      .onWord("sold") { (args, ctx) =>
        val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
        IO.pure(TradeAction(s"sells $qty"))
      }
      .build
```

**Injection Pattern**:
1. Define `given DomainHandlers[F, A]` in a scope
2. Import the given where needed
3. `runWithHandlers` summons the given automatically

#### Usage

```scala
def interpret(expr: ProfessExpr): IO[TradingDomain.TradeAction] =
  import TradingDomain.given
  runWithHandlers[IO, TradingDomain.TradeAction](expr)
```

**Benefits**:
- No explicit registry passing
- Easily swap implementations via imports
- Compile-time handler resolution

---

### Composable Domain Handlers

Demonstrates **modular handler composition** using traits.

#### Handler Traits

```scala
trait TradingWords[F[_]]:
  def soldHandler: WordHandler[F, Unit]
  def boughtHandler: WordHandler[F, Unit]

trait TradingObjects[F[_]]:
  def brokerHandler: ObjectHandler[F, Unit]
  def stockHandler: ObjectHandler[F, Unit]
```

**Design Pattern**: Separate handler categories into traits for:
- Independent testing
- Mix-and-match composition
- Effect polymorphism (same handlers, different effects)

#### Implementation

```scala
object IOTradingWords extends TradingWords[IO]:
  def soldHandler = WordHandler[IO, Unit]("sold") { (args, ctx) =>
    IO.println(s"SOLD ${args.collect { case IRNumber(n) => n.toInt }.mkString}")
  }
  def boughtHandler = WordHandler[IO, Unit]("bought") { (args, ctx) =>
    IO.println(s"BOUGHT ${args.collect { case IRNumber(n) => n.toInt }.mkString}")
  }
```

#### Composition

```scala
def makeRegistry(
  words: TradingWords[IO],
  objects: TradingObjects[IO]
): HandlerRegistry[IO, Unit] =
  HandlerRegistry[IO, Unit](
    wordHandlers = List(words.soldHandler, words.boughtHandler),
    objectHandlers = List(objects.brokerHandler, objects.stockHandler)
  )
```

**Generalization**: This pattern enables:
- Swapping handler implementations
- Testing with mock handlers
- Combining handlers from multiple sources

---

### Code Generation Example

Demonstrates **IR to code translation** for metaprogramming.

#### Result Type

```scala
case class CodeFragment(code: String)

given Monoid[CodeFragment] with
  def empty: CodeFragment = CodeFragment("")
  def combine(x: CodeFragment, y: CodeFragment): CodeFragment =
    CodeFragment(s"${x.code}\n${y.code}".trim)
```

**Pattern**: Newline-separated code accumulation.

#### Code-Generating Handlers

```scala
val registry: HandlerRegistry[IO, CodeFragment] =
  HandlerDSL.handlers[IO, CodeFragment]
    .onObject("broker") { (name, ctx) =>
      IO.pure(CodeFragment(s"""val broker = Broker("$name")"""))
    }
    .onWord("sold") { (args, ctx) =>
      val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
      IO.pure(CodeFragment(s"""broker.sell($qty, stock)"""))
    }
    .onWord("at") { (args, ctx) =>
      val price = args.collectFirst { case IRNumber(n) => n }.getOrElse(0.0)
      IO.pure(CodeFragment(s"""  .atPrice($price)"""))
    }
    .build
```

**Output for `(broker Mark) sold 700 (stock MSFT) at 150`**:

```scala
val broker = Broker("Mark")
val stock = Stock("MSFT")
broker.sell(700, stock)
  .atPrice(150.0)
```

---

## π-Calculus Translation Example

Located in: `examples/src/main/scala/examples/PiCalculusExample.scala`

This example demonstrates **domain-specific IR**, **algebraic optimizations**, and **code generation**.

### IR Design Decisions

#### Why Custom IR?

The π-calculus has specific semantics that require dedicated representation:

```scala
sealed trait PiIR extends IRNode

case object PiNil extends PiIR
case class PiInput(channel: String, binding: String, cont: PiIR) extends PiIR
case class PiOutput(channel: String, value: String, cont: PiIR) extends PiIR
case class PiParallel(left: PiIR, right: PiIR) extends PiIR
case class PiRestriction(channel: String, body: PiIR) extends PiIR
case class PiReplication(body: PiIR) extends PiIR
case class PiChoice(left: PiIR, right: PiIR) extends PiIR
```

#### Node Semantics

| Node | Notation | Meaning |
|------|----------|---------|
| `PiNil` | 0 | Terminated process |
| `PiInput(c, x, P)` | c(x).P | Receive on channel c, bind to x, continue as P |
| `PiOutput(c, v, P)` | c̄⟨v⟩.P | Send v on channel c, continue as P |
| `PiParallel(P, Q)` | P \| Q | Parallel composition |
| `PiRestriction(c, P)` | (νc)P | Create private channel c, run P |
| `PiReplication(P)` | !P | Infinite parallel copies of P |
| `PiChoice(P, Q)` | P + Q | Non-deterministic choice |

#### Design Rationale

1. **`cont: PiIR`** — Continuation-passing style enables sequential composition
2. **`binding: String`** — Named binding for received values
3. **Sealed trait** — Enables exhaustive pattern matching in optimizations
4. **Extends IRNode** — Integrates with PROFESS infrastructure (but note: this requires unsealing IRNode)

---

### Optimization Passes

The π-calculus has well-defined algebraic laws that justify optimizations:

#### Dead Process Elimination

```scala
def eliminateNil(ir: PiIR): PiIR = ir match
  case PiParallel(p, PiNil) => eliminateNil(p)
  case PiParallel(PiNil, q) => eliminateNil(q)
  case PiParallel(p, q) => PiParallel(eliminateNil(p), eliminateNil(q))
  case PiInput(ch, b, cont) => PiInput(ch, b, eliminateNil(cont))
  case PiOutput(ch, v, cont) => PiOutput(ch, v, eliminateNil(cont))
  case PiRestriction(ch, body) => PiRestriction(ch, eliminateNil(body))
  case PiReplication(body) => PiReplication(eliminateNil(body))
  case PiChoice(l, r) => PiChoice(eliminateNil(l), eliminateNil(r))
  case other => other
```

**Algebraic Justification**: P | 0 ≡ P (monoid identity)

**Pattern**: Recursive descent with pattern matching, applying transformation at each node.

#### Parallel Normalization

```scala
def normalizeParallel(ir: PiIR): PiIR = ir match
  case PiParallel(PiParallel(a, b), c) =>
    normalizeParallel(PiParallel(a, PiParallel(b, c)))
  case PiParallel(a, b) =>
    PiParallel(normalizeParallel(a), normalizeParallel(b))
  // ... other cases
```

**Algebraic Justification**: (P | Q) | R ≡ P | (Q | R) (associativity)

**Benefit**: Consistent right-associative structure enables predictable traversal.

#### Channel Scope Narrowing

```scala
def narrowScope(ir: PiIR): PiIR =
  def freeChannels(p: PiIR): Set[String] = p match
    case PiNil => Set.empty
    case PiInput(ch, _, cont) => Set(ch) ++ freeChannels(cont)
    case PiOutput(ch, _, cont) => Set(ch) ++ freeChannels(cont)
    case PiParallel(l, r) => freeChannels(l) ++ freeChannels(r)
    case PiRestriction(ch, body) => freeChannels(body) - ch
    case PiReplication(body) => freeChannels(body)
    case PiChoice(l, r) => freeChannels(l) ++ freeChannels(r)
  
  ir match
    case PiRestriction(ch, PiParallel(p, q)) =>
      val fpP = freeChannels(p)
      if !fpP.contains(ch) then
        PiParallel(narrowScope(p), PiRestriction(ch, narrowScope(q)))
      // ... symmetric case
```

**Algebraic Justification**: (νx)(P | Q) ≡ P | (νx)Q when x ∉ fv(P)

**Benefit**: Reduces actor lifetime, enables earlier garbage collection.

**Helper Function**: `freeChannels` computes free variables using standard set operations.

#### Synchronous Communication Fusion

```scala
def fuseSynchronous(ir: PiIR): PiIR =
  def substitute(p: PiIR, variable: String, value: String): PiIR = p match
    case PiOutput(ch, v, cont) =>
      val newV = if v == variable then value else v
      PiOutput(ch, newV, substitute(cont, variable, value))
    // ... other cases
  
  ir match
    case PiRestriction(ch, PiParallel(PiOutput(ch1, v, p), PiInput(ch2, y, q)))
        if ch == ch1 && ch == ch2 =>
      // Fuse! No actors needed for this communication
      PiParallel(fuseSynchronous(p), fuseSynchronous(substitute(q, y, v)))
```

**Algebraic Justification**: (νx)(x̄⟨v⟩.P | x(y).Q) → P | Q[v/y]

**Benefit**: Eliminates actor creation for internal communication.

**Substitution**: `substitute(q, y, v)` replaces all occurrences of `y` with `v` in `q`.

#### Optimization Pipeline

```scala
def optimize(ir: PiIR): PiIR =
  val pass1 = eliminateNil(ir)
  val pass2 = normalizeParallel(pass1)
  val pass3 = narrowScope(pass2)
  val pass4 = fuseSynchronous(pass3)
  pass4
```

**Pattern**: Linear pipeline of pure transformations.

**Compositionality**: Each pass returns `PiIR`, enabling arbitrary ordering.

---

### Akka Code Generation

#### Target Representation

```scala
sealed trait AkkaCode

object AkkaCode:
  case class ActorDef(name: String, behavior: AkkaBehavior) extends AkkaCode
  case class Spawn(name: String, behaviorExpr: String) extends AkkaCode
  case class Tell(target: String, message: String) extends AkkaCode
  case class Seq(steps: List[AkkaCode]) extends AkkaCode
  case object Stopped extends AkkaCode
```

**Design Decisions**:

| Type | Purpose |
|------|---------|
| `ActorDef` | Actor definition with behavior |
| `Spawn` | Actor creation |
| `Tell` | Message send |
| `Seq` | Sequential composition |
| `Stopped` | Terminated actor |

#### Translation Context

```scala
case class TransContext(
  channelActors: Map[String, String] = Map.empty,
  counter: Int = 0
):
  def fresh(prefix: String): (String, TransContext) =
    (s"${prefix}_$counter", copy(counter = counter + 1))
  
  def bindChannel(ch: String, actor: String): TransContext =
    copy(channelActors = channelActors + (ch -> actor))
  
  def lookupChannel(ch: String): String =
    channelActors.getOrElse(ch, s"${ch}Actor")
```

**Purpose**: Thread translation state without mutation.

**Fresh Names**: Counter-based unique name generation.

**Channel Mapping**: Associates π-calculus channels with Akka actor refs.

#### Translation Function

```scala
type TransM[A] = StateT[IO, TransContext, A]

def translate(ir: PiIR): TransM[AkkaCode] = ir match
  
  case PiNil =>
    StateT.pure(Stopped)
  
  case PiInput(channel, binding, cont) =>
    for
      ctx <- StateT.get[IO, TransContext]
      actorRef = ctx.lookupChannel(channel)
      contCode <- translate(cont)
    yield ActorDef(
      s"${channel}Receiver",
      ReceiveMsg(List(binding -> contCode))
    )
  
  case PiOutput(channel, value, cont) =>
    for
      ctx <- StateT.get[IO, TransContext]
      actorRef = ctx.lookupChannel(channel)
      contCode <- translate(cont)
    yield Seq(List(
      Tell(actorRef, value),
      contCode
    ))
  
  case PiParallel(left, right) =>
    for
      ctx <- StateT.get[IO, TransContext]
      (leftName, ctx1) = ctx.fresh("par_left")
      _ <- StateT.set[IO, TransContext](ctx1)
      (rightName, ctx2) = ctx1.fresh("par_right")
      _ <- StateT.set[IO, TransContext](ctx2)
      leftCode <- translate(left)
      rightCode <- translate(right)
    yield Seq(List(
      Spawn(leftName, codeToString(leftCode)),
      Spawn(rightName, codeToString(rightCode))
    ))
```

**Translation Rules**:

| π-Calculus | Akka |
|------------|------|
| 0 | `Behaviors.stopped` |
| c(x).P | Actor receiving on c, continuing as P |
| c̄⟨v⟩.P | Tell to c actor, then P |
| P \| Q | Spawn two actors |
| (νc)P | Spawn channel actor, run P |
| !P | Actor that spawns copies on trigger |

#### Code Generation

```scala
def generate(code: AkkaCode, indent: Int = 0): String =
  val pad = "  " * indent
  code match
    case Stopped => 
      s"${pad}Behaviors.stopped"
    
    case Tell(target, message) =>
      s"""${pad}$target ! "$message""""
    
    case Spawn(name, behavior) =>
      s"""${pad}val $name = context.spawn($behavior, "$name")"""
    
    case Seq(steps) =>
      steps.map(s => generate(s, indent)).mkString("\n")
    
    case ActorDef(name, behavior) =>
      s"""${pad}def ${name}Behavior: Behavior[String] =
${generateBehavior(behavior, indent + 1)}"""
```

**Pattern**: Recursive pretty-printing with indentation tracking.

---

## Generalization Patterns

### Pattern 1: Effect-Polymorphic Handlers

```scala
trait DomainHandlers[F[_]: Monad, A]:
  def wordHandlers: List[WordHandler[F, A]]
  def objectHandlers: List[ObjectHandler[F, A]]
  
  def registry: HandlerRegistry[F, A] =
    HandlerRegistry(wordHandlers, objectHandlers)
```

**Generalization**: Define handlers once, use with any effect type.

### Pattern 2: Modular Handler Composition

```scala
// Define handler modules
trait OrderHandlers[F[_]]:
  def buyHandler: WordHandler[F, Result]
  def sellHandler: WordHandler[F, Result]

trait InventoryHandlers[F[_]]:
  def addHandler: WordHandler[F, Result]
  def removeHandler: WordHandler[F, Result]

// Compose modules
class FullSystem[F[_]: Monad](
  orders: OrderHandlers[F],
  inventory: InventoryHandlers[F]
):
  val registry = HandlerRegistry[F, Result](
    wordHandlers = List(
      orders.buyHandler,
      orders.sellHandler,
      inventory.addHandler,
      inventory.removeHandler
    ),
    objectHandlers = Nil
  )
```

### Pattern 3: Domain-Specific IR with Optimization

```scala
// 1. Define domain IR
sealed trait MyDomainIR
case class MyNode1(...) extends MyDomainIR
case class MyNode2(...) extends MyDomainIR

// 2. Define algebraic laws as optimizations
object MyOptimizations:
  def rule1(ir: MyDomainIR): MyDomainIR = ???
  def rule2(ir: MyDomainIR): MyDomainIR = ???
  
  def optimize(ir: MyDomainIR): MyDomainIR =
    rule2(rule1(ir))

// 3. Define target representation
sealed trait TargetCode
case class TargetNode1(...) extends TargetCode

// 4. Define translation
def translate(ir: MyDomainIR): TargetCode = ???

// 5. Define code generation
def generate(code: TargetCode): String = ???
```

### Pattern 4: Context-Aware Translation

```scala
case class MyContext(
  bindings: Map[String, Value] = Map.empty,
  counter: Int = 0
):
  def fresh(prefix: String): (String, MyContext) =
    (s"${prefix}_$counter", copy(counter = counter + 1))
  
  def bind(name: String, value: Value): MyContext =
    copy(bindings = bindings + (name -> value))

type TranslateM[A] = StateT[IO, MyContext, A]

def translate(node: MyIR): TranslateM[TargetCode] =
  for
    ctx <- StateT.get[IO, MyContext]
    (freshName, ctx1) = ctx.fresh("var")
    _ <- StateT.set(ctx1)
    // ... translation logic
  yield result
```

### Pattern 5: Result Combination via Monoid

```scala
// Define result type with Monoid
case class MyResult(items: List[Item])

given Monoid[MyResult] with
  def empty: MyResult = MyResult(Nil)
  def combine(x: MyResult, y: MyResult): MyResult =
    MyResult(x.items ++ y.items)

// Use in handlers - results combine automatically
val handler = WordHandler[IO, MyResult]("process") { (args, ctx) =>
  IO.pure(MyResult(List(Item(...))))
}
```

---

## Summary

### Key Takeaways

1. **Handlers are composable values** — Build registries from modular pieces
2. **Effects are polymorphic** — Same handlers work with IO, StateT, etc.
3. **Results combine via Monoid** — Enables parallel and sequential composition
4. **Domain IR enables optimization** — Algebraic laws justify transformations
5. **Code generation is pure** — IR → String is a simple recursive function

### When to Use Each Pattern

| Pattern | Use Case |
|---------|----------|
| StateT handlers | Accumulating state during interpretation |
| Executable handlers | Deferred effects, effect combination |
| Given injection | Swappable handler implementations |
| Composable traits | Modular, testable handler sets |
| Custom IR | Domain-specific optimizations |
| Code generation | Transpilation, metaprogramming |
