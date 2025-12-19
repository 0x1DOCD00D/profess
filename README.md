# PROFESS

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Scala 3](https://img.shields.io/badge/Scala-3.6+-red.svg)](https://www.scala-lang.org/)

**P**rogramming **R**ule **O**riented **F**ormalized **E**nglish **S**entence **S**pecifications

> A Scala 3 framework for building domain-specific languages with English-like syntax and formal semantic foundations.

---

## Table of Contents

- [Features](#-features)
- [Objectives](#-objectives)
- [Quick Start](#-quick-start)
- [Syntax Specification](#-syntax-specification)
- [Design Overview](#-design-overview)
- [Formal Semantics](#-formal-semantics)
- [Implementation](#-implementation)
- [Cats Effect Integration](#-cats-effect-integration)
- [Code Examples](#-code-examples)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- 🗣️ **Natural Syntax** — Write specifications in English-like sentences
- 📦 **First-Class Values** — Expressions are assignable, passable, composable
- 🔌 **Pluggable Semantics** — Define your own interpreters and handlers
- 🐱 **Cats Effect Integration** — Type-safe functional effect management
- 📐 **Formal Foundations** — Operational and denotational semantics with proven properties
- ⚡ **Algebraic Optimizations** — Provably correct transformations from monoid laws
- 🔧 **Minimal Learning Curve** — Domain experts can write specifications without learning Scala

---

## 🎯 Objectives

PROFESS is designed to bridge the gap between natural language specifications and executable Scala code:

1. **Enable Domain Experts to Write Specifications** — Allow business analysts and domain experts to write specifications in English-like syntax that compiles directly to working Scala programs.

2. **First-Class Expression Values** — Every PROFESS expression evaluates to a first-class value that can be assigned, passed, returned, and composed.

3. **Domain-Agnostic Framework** — No predefined vocabulary. Semantics are defined externally through interpreter functions.

4. **Formal Foundation** — Rigorous operational and denotational semantics ensuring predictable behavior.

5. **Functional Effect Integration** — Full integration with Typelevel Cats and Cats Effect for type-safe effect management.

---

## 📋 Quick Start

### Installation

Add to your `build.sbt`:

```scala
libraryDependencies += "io.github.yourusername" %% "profess-runtime" % "0.1.0"

// For compiler plugin (enables natural syntax)
addCompilerPlugin("io.github.yourusername" %% "profess-plugin" % "0.1.0")
```

### Basic Usage

```scala
import profess.runtime._

// Define expressions with natural syntax
val trade = (broker Mark) sold 700 shares (stock MSFT) at 150 dollars

// Variable interpolation with !
val name = "Jane"
val qty = 500
val order = (broker !name) bought !qty (stock AAPL)

// Unit values with :
val precise = (broker Mark) sold 700:shares at 150.50:dollars

// First-class values - pass to functions
def analyze(expr: ProfessExpr): Report = ???
val report = analyze(trade)
```

### With Cats Effect

```scala
import cats.effect.IO
import profess.runtime.cats._

// Define handlers for your domain
given DomainHandlers[IO, Trade] with
  val registry = HandlerDSL.handlers[IO, Trade]
    .onObject("broker") { (name, _) => IO.pure(Trade(broker = name)) }
    .onWord("sold") { (args, _) => 
      val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
      IO.pure(Trade(action = "sell", quantity = qty)) 
    }
    .build

// Execute with automatic handler injection
val result: IO[Trade] = runWithHandlers[IO, Trade](trade)
```

---

## 📝 Syntax Specification

### Grammar (EBNF)

```ebnf
expression     ::= subject parts
subject        ::= PROFESS-object | word | variable-ref
PROFESS-object ::= '(' kindId (name | '!' variable) ')'
variable-ref   ::= '!' variable
parts          ::= (word | PROFESS-object | number | variable-ref | unit-value)*
unit-value     ::= number ':' word
conditional    ::= 'If' expression 'then' expression ('else' | 'otherwise') expression
```

### Syntax Elements

| Element | Syntax | Example |
|---------|--------|---------|
| PROFESS Object | `(kindId Name)` | `(broker Mark)` |
| Variable Interpolation | `!variable` | `(broker !name)` |
| Unit Value | `number:unit` | `700:shares` |
| Conditional | `If...then...else` | `If online then allow else deny` |
| Sequence | juxtaposition | `sold 700 shares` |

### Requirements

**Functional Requirements:**
- Expression syntax with PROFESS objects `(kindId Name)` where kindId is lowercase
- Variable interpolation with `!` prefix operator
- Unit values with `:` operator for attaching units (e.g., `700:shares`, `150:dollars`)
- Conditional expressions with `If...then...else/otherwise` constructs
- Scope-aware scaffolding that only scaffolds truly undeclared identifiers

**Non-Functional Requirements:**
- Minimal compilation overhead
- Scala 3.4+ compatibility
- Extensibility through user-defined interpreters and handlers

---

## 🏛️ Design Overview

### Architecture

PROFESS consists of three main components:

1. **Compiler Plugin** — Runs between parser and typer phases. Identifies patterns and generates scaffolding for undeclared identifiers.

2. **Runtime Library** — Provides `ProfessExpr`, IR types, scaffolding types, and the `Dynamic` trait implementation.

3. **Cats Integration** — Handler typeclasses, registry, traverser, and effect execution.

### Compilation Flow

| Stage | Description |
|-------|-------------|
| 1. Parser | Scala parser produces untyped AST |
| 2. PROFESS Plugin | Collect declarations, identify patterns, generate scaffolding |
| 3. Typer | Type checking with scaffolded identifiers |
| 4. Execution | Expressions evaluate to `ProfessExpr` values |
| 5. Interpretation | Handlers transform IR nodes into domain objects or effects |

### How It Works

```
Source Code                    After Plugin                    At Runtime
─────────────────────────────────────────────────────────────────────────────
(broker Mark) sold 700    →    broker.apply(Mark)         →    ProfessExpr(
                               .selectDynamic("sold")           IRSequence([
                               .applyDynamic(700)                 IRObject("broker","Mark"),
                                                                  IRWord("sold"),
                                                                  IRNumber(700)
                                                                ])
                                                               )
```

---

## 📐 Formal Semantics

PROFESS has rigorous formal foundations based on operational and denotational semantics.

### Syntactic Categories

```
e ∈ Expr       Expressions
v ∈ Val        Values
n ∈ Num        Numeric literals
w ∈ Word       Words (identifiers)
k ∈ Kind       Kind identifiers (lowercase)
x ∈ Var        Variables
ι ∈ IR         Intermediate Representation nodes
```

### Abstract Syntax

```
e ::= (k n)                   -- PROFESS object
    | (k !x)                  -- Object with variable interpolation
    | !x                      -- Variable interpolation
    | n | n:w | w             -- Literals and words
    | e e                     -- Sequence (juxtaposition)
    | If e then e else e      -- Conditional
```

### Big-Step Operational Semantics

We define the judgment `ρ ⊢ e ⇓ v` meaning "under environment ρ, expression e evaluates to value v":

**Rule [E-Object]: PROFESS Object Evaluation**
```
─────────────────────────────────────────────────
ρ ⊢ (k n) ⇓ ProfessExpr(IRObject(k, n))
```

**Rule [E-Var]: Variable Interpolation**
```
ρ(x) = v
──────────────
ρ ⊢ !x ⇓ v
```

**Rule [E-Seq]: Sequence Evaluation**
```
ρ ⊢ e₁ ⇓ ProfessExpr(ι₁)    ρ ⊢ e₂ ⇓ ProfessExpr(ι₂)
────────────────────────────────────────────────────────
ρ ⊢ e₁ e₂ ⇓ ProfessExpr(flatten(ι₁, ι₂))
```

**Rule [E-Cond]: Conditional Evaluation**
```
ρ ⊢ e₁ ⇓ ProfessExpr(ι₁)  ρ ⊢ e₂ ⇓ ProfessExpr(ι₂)  ρ ⊢ e₃ ⇓ ProfessExpr(ι₃)
─────────────────────────────────────────────────────────────────────────────────
ρ ⊢ If e₁ then e₂ else e₃ ⇓ ProfessExpr(IRConditional(ι₁, ι₂, ι₃))
```

### Denotational Semantics

The semantic function `⟦_⟧ : Expr → Env → Val` maps expressions to values:

```
⟦(k n)⟧ρ = ProfessExpr(IRObject(k, n))

⟦!x⟧ρ = ρ(x)

⟦e₁ e₂⟧ρ = let ProfessExpr(ι₁) = ⟦e₁⟧ρ in
           let ProfessExpr(ι₂) = ⟦e₂⟧ρ in
           ProfessExpr(flatten(ι₁, ι₂))

⟦If e₁ then e₂ else e₃⟧ρ = 
    ProfessExpr(IRConditional(⟦e₁⟧ρ.ir, ⟦e₂⟧ρ.ir, ⟦e₃⟧ρ.ir))
```

### Handler Algebra

Handlers form a monoid `(H, ⊕, ε)` under composition:

```
Identity:      R ⊕ ε = ε ⊕ R = R
Associativity: (R₁ ⊕ R₂) ⊕ R₃ = R₁ ⊕ (R₂ ⊕ R₃)
```

Handler lookup with precedence (later handlers override):

```
lookup(w, R₁ ⊕ R₂) = R₂(w) if w ∈ dom(R₂)
                   = R₁(w) if w ∈ dom(R₁) ∧ w ∉ dom(R₂)
                   = ⊥     otherwise
```

### Interpretation Semantics

Given registry `R` and result monoid `(A, ⊗, ε_A)`, interpretation is:

```
interp : IR × Registry × Context → F[A]

interp(IRObject(k, n), R, ctx) = 
    case R.objects(k) of
      Some(h) → h.handle(n, ctx)
      None    → pure(ε_A)

interp(IRSequence(ι :: rest), R, ctx) =
    interp(ι, R, ctx) ⊗_F interp(IRSequence(rest), R, ctx)
```

Where `⊗_F` lifts the monoid into the effect:

```
m₁ ⊗_F m₂ = for { a₁ ← m₁; a₂ ← m₂ } yield a₁ ⊗ a₂
```

### Soundness Properties

| Property | Statement |
|----------|-----------|
| **Determinism** | Evaluation produces unique results |
| **Type Soundness (Progress)** | If `∅ ⊢ e : τ` then either e is a value or `∃e'. e → e'` |
| **Type Soundness (Preservation)** | If `Γ ⊢ e : τ` and `e → e'` then `Γ ⊢ e' : τ` |
| **Denotational-Operational Correspondence** | `⟦e⟧ρ = v` iff `ρ ⊢ e ⇓ v` |
| **Handler Compositionality** | For any R₁, R₂: `interp(e, R₁ ⊕ R₂)` is well-defined |
| **Effect Safety** | For pure handlers, interpretation is referentially transparent |

### Enabled Optimizations

| Algebraic Law | Optimization |
|---------------|--------------|
| `R ⊕ ε = R` | Dead handler elimination |
| Associativity | Handler fusion (single-pass compilation) |
| Commutativity | Reordering for locality |

---

## 🔧 Implementation

### IR Types

The Intermediate Representation captures the structure of PROFESS expressions:

```scala
sealed trait IRNode

case class IRObject(kind: String, name: String) extends IRNode
case class IRWord(word: String) extends IRNode
case class IRNumber(value: Double) extends IRNode
case class IRString(value: String) extends IRNode
case class IRUnitValue(value: Double, unit: String) extends IRNode
case class IRSequence(nodes: List[IRNode]) extends IRNode
case class IRConditional(
  condition: IRNode,
  thenBranch: IRNode,
  elseBranch: Option[IRNode]
) extends IRNode
case class IRBinding(variable: String, value: IRNode) extends IRNode
case class IRReference(variable: String) extends IRNode
```

### Scaffolding Types

These types enable the natural syntax:

```scala
// Kind identifier - lowercase, used as method name
class ProfessKind(val kind: String) extends Dynamic:
  def apply(name: ProfessName): ProfessExpr = ...
  def apply(ref: ProfessRef): ProfessExpr = ...

// Name identifier - capitalized entity name  
class ProfessName(val name: String)

// Word - action or modifier in expression
class ProfessWord(val word: String)

// Object wrapper for building expressions
class ProfessObject(val kind: String, val name: String)
```

### ProfessExpr

The core expression type:

```scala
class ProfessExpr(nodes: List[IRNode]) extends Dynamic:
  
  def toIR: IRNode = nodes match
    case single :: Nil => single
    case multiple      => IRSequence(multiple)
  
  // Enable method chaining: expr.sold
  def selectDynamic(word: String): ProfessExpr =
    ProfessExpr(nodes :+ IRWord(word))
  
  // Enable method calls: expr.sold(700)
  def applyDynamic(word: String)(args: Any*): ProfessExpr =
    val wordNode = IRWord(word)
    val argNodes = args.map(convertToIR).toList
    ProfessExpr(nodes ++ (wordNode :: argNodes))
  
  // Combine expressions
  def combine(other: ProfessExpr): ProfessExpr =
    ProfessExpr(this.nodes ++ other.nodes)
```

---

## 🐱 Cats Effect Integration

### Handler Typeclasses

```scala
trait WordHandler[F[_], A]:
  def word: String
  def handle(args: List[IRNode], ctx: TransformContext): F[A]

trait ObjectHandler[F[_], A]:
  def kind: String
  def handle(name: String, ctx: TransformContext): F[A]
```

### Handler Registry

```scala
class HandlerRegistry[F[_], A](
  wordHandlers: Map[String, WordHandler[F, A]],
  objectHandlers: Map[String, ObjectHandler[F, A]]
):
  def findWordHandler(word: String): Option[WordHandler[F, A]]
  def findObjectHandler(kind: String): Option[ObjectHandler[F, A]]
  
  // Monoid composition
  def combine(other: HandlerRegistry[F, A]): HandlerRegistry[F, A]
```

### Handler DSL

Fluent builder for creating registries:

```scala
val registry = HandlerDSL.handlers[IO, Trade]
  .onWord("sold") { (args, ctx) => 
    val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
    IO.pure(Trade(action = "sell", quantity = qty))
  }
  .onWord("bought") { (args, ctx) =>
    val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
    IO.pure(Trade(action = "buy", quantity = qty))
  }
  .onObject("broker") { (name, ctx) => 
    IO.pure(Trade(broker = name))
  }
  .onObject("stock") { (name, ctx) =>
    IO.pure(Trade(symbol = name))
  }
  .build
```

### Given-Based Injection

```scala
// Define handlers as a given instance
given DomainHandlers[IO, Trade] with
  val registry = HandlerDSL.handlers[IO, Trade]
    .onWord("sold") { (args, _) => IO.pure(Trade(action = "sell")) }
    .onObject("broker") { (name, _) => IO.pure(Trade(broker = name)) }
    .build

// Handlers automatically injected via context
val trade = (broker Mark) sold 700 (stock MSFT)
val result: IO[Trade] = runWithHandlers[IO, Trade](trade)
```

### IR Traverser

```scala
class IRTraverser[F[_]: Monad, A: Monoid](registry: HandlerRegistry[F, A]):
  
  def traverse(expr: ProfessExpr): F[A] =
    traverseNode(expr.toIR)
  
  private def traverseNode(node: IRNode): F[A] = node match
    case IRObject(kind, name) =>
      registry.findObjectHandler(kind) match
        case Some(handler) => handler.handle(name, ctx)
        case None          => Monoid[A].empty.pure[F]
    
    case IRWord(word) =>
      registry.findWordHandler(word) match
        case Some(handler) => handler.handle(Nil, ctx)
        case None          => Monoid[A].empty.pure[F]
    
    case IRSequence(nodes) =>
      nodes.traverse(traverseNode).map(_.combineAll)
    
    case IRConditional(cond, thenB, elseB) =>
      for
        c <- traverseNode(cond)
        t <- traverseNode(thenB)
        e <- elseB.traverse(traverseNode).map(_.getOrElse(Monoid[A].empty))
      yield c |+| t |+| e
    
    case _ => Monoid[A].empty.pure[F]
```

---

## 💻 Code Examples

### Basic Trading

```scala
val trade1 = (broker Mark) sold 700 shares (stock MSFT)
val trade2 = (broker Jane) bought 500 (stock AAPL)
```

### Variable Interpolation

```scala
val name = "Mark"
val qty = 700
val ticker = "MSFT"
val trade = (broker !name) sold !qty (stock !ticker)
```

### Unit Values

```scala
val order = (broker Mark) sold 700:shares at 150.50:dollars
```

### Conditional Expressions

```scala
val review = If trade exceeds 1000 then flag for review otherwise auto approve

val access = If user is authenticated then grant access else deny
```

### Complete Trading Interpreter

```scala
import cats._
import cats.effect._
import profess.runtime._
import profess.runtime.cats._

// Domain model
case class Trade(
  broker: String = "",
  action: String = "",
  quantity: Int = 0,
  symbol: String = "",
  price: Double = 0.0
)

// Monoid for combining partial trades
given Monoid[Trade] with
  def empty = Trade()
  def combine(x: Trade, y: Trade) = Trade(
    broker   = if y.broker.nonEmpty then y.broker else x.broker,
    action   = if y.action.nonEmpty then y.action else x.action,
    quantity = if y.quantity > 0 then y.quantity else x.quantity,
    symbol   = if y.symbol.nonEmpty then y.symbol else x.symbol,
    price    = if y.price > 0 then y.price else x.price
  )

// Handler definitions
given DomainHandlers[IO, Trade] with
  val registry = HandlerDSL.handlers[IO, Trade]
    .onObject("broker") { (name, _) => 
      IO.pure(Trade(broker = name)) 
    }
    .onObject("stock") { (name, _) => 
      IO.pure(Trade(symbol = name)) 
    }
    .onWord("sold") { (args, _) =>
      val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
      IO.pure(Trade(action = "sell", quantity = qty))
    }
    .onWord("bought") { (args, _) =>
      val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
      IO.pure(Trade(action = "buy", quantity = qty))
    }
    .onWord("at") { (args, _) =>
      val price = args.collectFirst { case IRNumber(n) => n }.getOrElse(0.0)
      IO.pure(Trade(price = price))
    }
    .build

// Usage
val trade = (broker Mark) sold 700 (stock MSFT) at 150.50
val result: IO[Trade] = runWithHandlers[IO, Trade](trade)

// result.unsafeRunSync() == Trade("Mark", "sell", 700, "MSFT", 150.50)
```

### π-Calculus to Akka Translation

The formal theory directly supports process calculus translation:

```scala
// π-calculus process definition
val pingPong = 
  new (channel ping) in
  new (channel pong) in
  (
    (channel ping) receives x then (channel pong) sends x then stop
  ) parallel (
    (channel pong) receives y then (channel ping) sends y then stop
  ) parallel (
    (channel ping) sends "hello" then stop
  )

// Translates to Akka actors with semantic preservation
val akkaCode = PiTranslator.translate(pingPong)
```

---

## 🏗️ Project Structure

```
profess/
├── plugin/                           # Scala 3 compiler plugin
│   └── src/main/scala/profess/plugin/
│       └── ProfessPlugin.scala       # AST transformation
├── runtime/                          # Core runtime library
│   └── src/main/scala/profess/runtime/
│       ├── Runtime.scala             # IR types, ProfessExpr
│       └── cats/
│           └── CatsInterpreter.scala # Cats Effect integration
├── examples/                         # Usage examples
│   └── src/main/scala/examples/
│       ├── BasicExamples.scala
│       ├── CatsExamples.scala
│       └── PiCalculusExample.scala
├── docs/                             # Documentation
│   ├── DESIGN.md
│   ├── FORMAL-SEMANTICS.md
│   └── APPLIED-THEORY.md
├── build.sbt
├── LICENSE                           # Apache 2.0
└── README.md
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Design](docs/DESIGN.md) | Architecture and implementation details |
| [Formal Semantics](docs/FORMAL-SEMANTICS.md) | Operational/denotational semantics, type system |
| [Applied Theory](docs/APPLIED-THEORY.md) | Optimizations, π-calculus example |
| [GitHub Setup](GITHUB-SETUP-GUIDE.md) | How to set up this project |

---

## 🛠️ Development

### Prerequisites

- JDK 17 or 21 (LTS)
- sbt 1.10+
- Scala 3.6+

### Build Commands

```bash
sbt compile        # Compile all modules
sbt test           # Run tests
sbt fmt            # Format code (alias for scalafmtAll)
sbt fmtCheck       # Check formatting
sbt runExamples    # Run example code
sbt coverage test  # Run tests with coverage
```

### IDE Setup

1. Install IntelliJ IDEA with Scala plugin
2. Open project: **File → Open → select profess directory**
3. Import as sbt project when prompted
4. Wait for indexing to complete

### Testing

```bash
# All tests
sbt test

# Specific module
sbt "runtime/test"

# With coverage report
sbt coverage test coverageReport
```

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Typelevel](https://typelevel.org/) for Cats and Cats Effect
- [Scala](https://www.scala-lang.org/) team for Scala 3

---

<p align="center">
  <strong>PROFESS</strong> — Bridging Natural Language and Executable Code
  <br>
  Made with ❤️ for the Scala community
</p>