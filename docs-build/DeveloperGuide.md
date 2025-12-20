# PROFESS Developer Guide

**Programming Rule Oriented Formalized English Sentence Specifications**

A Scala 3 Framework for Domain-Specific Languages with Formal Semantics

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Module Architecture](#module-architecture)
3. [Build System Analysis](#build-system-analysis)
4. [Code Organization](#code-organization)
5. [Plugin Module Deep Dive](#plugin-module-deep-dive)
6. [Runtime Module Deep Dive](#runtime-module-deep-dive)
7. [Cats Integration Module](#cats-integration-module)
8. [Execution Flow](#execution-flow)
9. [Adding New Features](#adding-new-features)

---

## Project Overview

PROFESS enables writing domain-specific expressions in natural English-like syntax that compile to Scala code. The key innovation is that **every PROFESS expression evaluates to a first-class value** (`ProfessExpr`) that can be assigned, passed to functions, returned from functions, and composed.

### Core Design Principles

1. **First-Class Values**: Unlike traditional DSLs that accumulate side effects, PROFESS expressions are values
2. **Deferred Semantics**: Syntax is defined separately from interpretation
3. **Compile-Time Safety**: The compiler plugin validates expressions during compilation
4. **Scope Awareness**: Only scaffolds identifiers not already declared in scope
5. **Extensible Interpretation**: Multiple interpreters can process the same IR

### Example

```scala
// Natural English syntax
val trade = (broker Mark) sold 700 shares (stock MSFT) on (exchange NYSE)

// Results in first-class value containing IR:
// IRSequence([IRObject("broker", "Mark"), IRWord("sold"), IRNumber(700), ...])
```

---

## Module Architecture

```
profess/
├── plugin/          # Scala 3 compiler plugin
│   └── profess.plugin.ProfessPlugin
│
├── runtime/         # Core runtime library
│   └── profess.runtime.*
│   └── profess.runtime.effect.*   # Cats integration (optional)
│
├── examples/        # Usage examples
│   └── examples.*
│
└── docs-build/      # Documentation (mdoc)
```

### Why This Structure?

| Module | Purpose | Dependencies |
|--------|---------|--------------|
| `plugin` | Compile-time AST transformation | `scala3-compiler` |
| `runtime` | Runtime types and interpreter API | `cats-core`, `cats-effect` |
| `examples` | Demonstrates PROFESS features | `runtime` + plugin enabled |

**Key Design Decision**: The plugin and runtime are separate modules because:

1. **Plugin runs at compile-time** with access to compiler internals
2. **Runtime runs at execution-time** and must not depend on compiler APIs
3. **Clean separation** allows runtime to be published independently

### Package Structure

```
profess.plugin          # Compiler plugin (compile-time only)
├── ProfessPlugin       # Plugin entry point
├── ProfessPhase        # Compiler phase
├── DeclarationCollector
├── ExpressionCollector
├── ScaffoldGenerator
└── ASTInjector

profess.runtime         # Core runtime types
├── IRNode              # IR hierarchy (sealed trait)
├── ProfessExpr         # First-class expression value
├── ProfessKind         # Object constructor
├── ProfessName         # Entity name
├── ProfessWord         # Standalone word
├── ProfessInterpreter  # Interpreter trait
└── IRVisitor           # Visitor pattern

profess.runtime.effect  # Cats Effect integration
├── TransformContext
├── WordHandler
├── ObjectHandler
├── HandlerRegistry
├── IRTraverser
├── Executable
└── HandlerDSL
```

---

## Build System Analysis

### Version Definitions

```scala
val scala3Version = "3.7.4"
val catsVersion = "2.13.0"
val catsEffectVersion = "3.6.3"
```

Versions are defined as `val` at the top of `build.sbt` for centralized management.

### Global Settings

```scala
ThisBuild / organization := "com.profess"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := scala3Version
```

`ThisBuild` scope ensures these settings apply to all subprojects.

### Common Settings

```scala
lazy val commonSettings = Seq(
  scalacOptions ++= Seq(
    "-deprecation",
    "-feature",
    "-unchecked",
    "-language:dynamics",        // Required for Dynamic trait
    "-language:higherKinds",     // Required for F[_] type parameters
    "-language:implicitConversions"
  ),
  libraryDependencies ++= Seq(
    catsCore,
    catsEffect,
    scalatest,
    // ... other deps
  )
)
```

**Critical Compiler Flags**:

- `-language:dynamics`: Enables `scala.Dynamic` for PROFESS chaining syntax
- `-language:higherKinds`: Required for `F[_]` effect types in Cats integration

### Plugin Module

```scala
lazy val plugin = project
  .in(file("plugin"))
  .settings(commonSettings)
  .settings(
    name := "profess-plugin",
    libraryDependencies ++= Seq(
      "org.scala-lang" %% "scala3-compiler" % scalaVersion.value
    ),
    Compile / packageBin / packageOptions +=
      Package.ManifestAttributes("Scala-Compiler-Plugin" -> "true")
  )
```

**Key Points**:

1. `scala3-compiler` dependency gives access to `dotty.tools.dotc.*` APIs
2. `Scala-Compiler-Plugin` manifest attribute marks JAR as a compiler plugin
3. Plugin runs in compiler process, not application process

### Examples Module

```scala
lazy val examples = project
  .in(file("examples"))
  .dependsOn(runtime)
  .settings(commonSettings)
  .settings(
    scalacOptions ++= Seq(
      s"-Xplugin:${(plugin / Compile / packageBin).value.getAbsolutePath}"
    ),
    Compile / compile := ((Compile / compile) dependsOn (plugin / Compile / packageBin)).value
  )
```

**Critical Configuration**:

1. `-Xplugin:...` loads the PROFESS compiler plugin
2. `dependsOn` ensures plugin is compiled before examples
3. Examples depend on `runtime` for runtime types

### Useful Commands

```bash
sbt compile           # Compile all modules
sbt plugin/compile    # Compile plugin only
sbt runtime/compile   # Compile runtime only
sbt examples/compile  # Compile examples (triggers plugin)
sbt examples/run      # Run example main class
sbt clean             # Clean all build artifacts
```

**Command Aliases** (defined in build.sbt):

```scala
addCommandAlias("build", "compile; test")
addCommandAlias("runExamples", "examples/run")
```

---

## Code Organization

### Plugin Module: `profess.plugin`

| File | Purpose |
|------|---------|
| `ProfessPlugin.scala` | All plugin code in single file |

### Runtime Module: `profess.runtime`

| File | Purpose |
|------|---------|
| `Runtime.scala` | Core IR types, expressions, interpreters |

### Runtime Effect Module: `profess.runtime.effect`

| File | Purpose |
|------|---------|
| `CatsInterpreter.scala` | Cats Effect integration |

---

## Plugin Module Deep Dive

The plugin transforms source code **after parsing but before type checking**.

### `ProfessPlugin` (Entry Point)

```scala
class ProfessPlugin extends StandardPlugin:
  val name: String = "profess"
  
  @nowarn("msg=deprecated")
  override def init(options: List[String]): List[PluginPhase] =
    List(new ProfessPhase)
```

**Purpose**: Registers the plugin with the Scala compiler.

**Note**: `init` is deprecated in Scala 3.7+ but still functional.

### `ProfessPhase` (Compiler Phase)

```scala
class ProfessPhase extends PluginPhase:
  val phaseName: String = "profess"
  override val runsAfter: Set[String] = Set("parser")
  override val runsBefore: Set[String] = Set("typer")
  
  override def prepareForUnit(tree: tpd.Tree)(using Context): Context
```

**Purpose**: Defines when the plugin runs in the compilation pipeline.

**Phase Ordering**:
```
parser → [profess] → typer → ...
```

Running after parser means we work with **untyped AST** (`untpd.Tree`).
Running before typer means our generated code will be type-checked.

### `IdKind` (Identifier Classification)

```scala
enum IdKind:
  case Kind   // lowercase in (kindId name) position: broker, stock
  case Name   // name position: Mark, MSFT  
  case Word   // standalone word: sold, bought, transferred
```

**Purpose**: Classifies identifiers to generate appropriate scaffolding types.

### `DeclarationCollector` (Scope Analysis)

```scala
class DeclarationCollector(using Context) extends UntypedTreeTraverser:
  private val declared = mutable.Set[String]()
  
  override def traverse(tree: Tree)(using Context): Unit =
    tree match
      case ValDef(name, _, _) => declared += name.toString
      case DefDef(name, paramss, _, _) => // collect name and params
      case TypeDef(name, _) => // class/trait names
      case ModuleDef(name, _) => // object names
      case Import(_, selectors) => // imported names
      case Bind(name, _) => // pattern bindings
      case GenFrom(pat, _, _) => // for-comprehension bindings
      case _ => traverseChildren(tree)
```

**Purpose**: Collects all identifiers that are **already declared** in scope.

**Why This Matters**: PROFESS only scaffolds identifiers that aren't already defined. This prevents conflicts with user code.

**Collected Identifiers**:
- `val`/`var` names
- Method names and parameters
- Class/trait/object names
- Imported names (including renames)
- Pattern bindings in match expressions
- For-comprehension generators

### `ExpressionCollector` (Pattern Detection)

```scala
class ExpressionCollector(declared: Set[String])(using Context) extends UntypedTreeTraverser:
  private val candidates = mutable.Map[String, IdKind]()
  
  override def traverse(tree: Tree)(using Context): Unit =
    tree match
      // PROFESS object pattern: (kindId name)
      case Apply(Ident(kindId), List(Ident(name))) 
          if isKindCandidate(kindId.toString) =>
        // Record kindId as Kind, name as Name
      
      // Standalone identifier
      case Ident(name) =>
        // Record as Word if not declared
```

**Purpose**: Identifies PROFESS expression patterns in source code.

**Pattern Recognition**:

| Source Code | AST Pattern | Classification |
|-------------|-------------|----------------|
| `(broker Mark)` | `Apply(Ident("broker"), List(Ident("Mark")))` | broker=Kind, Mark=Name |
| `sold` | `Ident("sold")` | sold=Word |

**Filtering Logic**:
- Skip Scala keywords (`if`, `val`, etc.)
- Skip already-declared identifiers
- Skip common Scala identifiers (`map`, `println`, etc.)

### `ScaffoldGenerator` (Code Generation)

```scala
object ScaffoldGenerator:
  def generate(identifiers: Map[String, IdKind])(using Context): List[ValDef] =
    identifiers.map { (id, kind) =>
      val typeName = kind match
        case IdKind.Kind => "ProfessKind"
        case IdKind.Name => "ProfessName"
        case IdKind.Word => "ProfessWord"
      
      // Generate: val id = _root_.profess.runtime.TypeName("id")
      ValDef(id.toTermName, TypeTree(), 
        Apply(Select(..., typeName.toTermName), List(Literal(Constant(id)))))
    }
```

**Purpose**: Generates `val` definitions for scaffolding types.

**Generated Code Example**:

For source code:
```scala
val trade = (broker Mark) sold 700 (stock MSFT)
```

Generates:
```scala
val broker = _root_.profess.runtime.ProfessKind("broker")
val Mark = _root_.profess.runtime.ProfessName("Mark")
val sold = _root_.profess.runtime.ProfessWord("sold")
val stock = _root_.profess.runtime.ProfessKind("stock")
val MSFT = _root_.profess.runtime.ProfessName("MSFT")
```

### `ASTInjector` (AST Modification)

```scala
object ASTInjector:
  def inject(tree: Tree, scaffolding: List[ValDef], declared: Set[String]): Tree =
    object Injector extends UntypedTreeMap:
      override def transform(tree: Tree)(using Context): Tree =
        tree match
          case md @ ModuleDef(name, impl) =>
            // Inject into object body
          case td @ TypeDef(name, rhs: Template) =>
            // Inject into class/trait body
```

**Purpose**: Inserts generated scaffolding into the AST.

**Injection Strategy**:
1. Find object/class definitions that contain PROFESS expressions
2. Prepend scaffolding `val` definitions to the body
3. Only inject scaffolding that's actually used in that scope

---

## Runtime Module Deep Dive

### IR Node Hierarchy

```scala
sealed trait IRNode:
  def render: String

case class IRObject(kind: String, name: String) extends IRNode
case class IRWord(word: String) extends IRNode
case class IRNumber(value: Double) extends IRNode
case class IRString(value: String) extends IRNode
case class IRSequence(nodes: List[IRNode]) extends IRNode
case class IRConditional(condition: IRNode, consequent: IRNode, alternative: Option[IRNode]) extends IRNode
case class IRBinding(name: String, value: IRNode) extends IRNode
case class IRReference(name: String) extends IRNode
case class IRTuple(elements: List[IRNode]) extends IRNode
case class IRUnitValue(value: Double, unit: String) extends IRNode
case class IRBoolean(value: Boolean) extends IRNode
case class IRParamBlock(params: List[(Option[String], IRNode)]) extends IRNode
case class IRAttributes(target: IRNode, attrs: List[String]) extends IRNode
```

**Purpose**: Intermediate Representation for PROFESS expressions.

**Design Rationale**:
- `sealed trait` enables exhaustive pattern matching
- Pure data classes (case classes) enable structural equality
- `render` method provides human-readable output

### `ProfessExpr` (First-Class Expression Value)

```scala
class ProfessExpr(protected val nodes: List[IRNode]) extends Dynamic:
  def toIR: IRNode
  def getNodes: List[IRNode]
  
  // Dynamic dispatch for chaining
  def selectDynamic(word: String): ProfessExpr
  def applyDynamic(word: String)(args: Any*): ProfessExpr
  
  // Conditional support
  def then(consequent: ProfessExpr): ProfessConditional
  
  // Helpers
  protected def argToNodes(arg: Any): List[IRNode]
```

**Purpose**: The core value type that all PROFESS expressions evaluate to.

**Key Design Decisions**:

1. **Extends `Dynamic`**: Enables arbitrary method calls like `.sold`, `.bought`
2. **Protected `nodes`**: Encapsulates IR, exposes via `getNodes`
3. **Immutable chaining**: Each method returns a new `ProfessExpr`

**Dynamic Dispatch**:
```scala
expr.sold        // calls selectDynamic("sold")
expr.sold(700)   // calls applyDynamic("sold")(700)
```

### Scaffolding Types

```scala
class ProfessKind(val kindId: String) extends Dynamic:
  def apply(name: ProfessName): ProfessObject
  def apply(name: String): ProfessObject
  def selectDynamic(name: String): ProfessObject

class ProfessName(val value: String)

class ProfessWord(val word: String) extends ProfessExpr(List(IRWord(word)))

class ProfessObject(val kind: String, val name: String) 
    extends ProfessExpr(List(IRObject(kind, name)))
```

**Purpose**: Types that the plugin scaffolds for undeclared identifiers.

**How They Work Together**:

```scala
// After scaffolding:
val broker = ProfessKind("broker")
val Mark = ProfessName("Mark")

// Expression evaluation:
(broker Mark)
// 1. broker.apply(Mark) → ProfessKind.apply(ProfessName)
// 2. Returns ProfessObject("broker", "Mark")
// 3. ProfessObject extends ProfessExpr with IRObject node
```

### Variable Interpolation

```scala
extension (s: String)
  def unary_! : ProfessName = ProfessName(s)

extension (n: Int)
  def unary_! : ProfessNumber = ProfessNumber(n)
  def `:`(unit: ProfessWord): ProfessUnitValue

extension (n: Double)
  def unary_! : ProfessNumber = ProfessNumber(n)
  def `:`(unit: ProfessWord): ProfessUnitValue
```

**Purpose**: Enables interpolating Scala variables into PROFESS expressions.

**Usage**:
```scala
val name = "Mark"
val qty = 700
(broker !name) sold !qty shares  // !name interpolates the variable
```

**Note**: `Boolean.unary_!` conflicts with built-in negation, so we use `.p`:
```scala
val flag = true
(status Active) is flag.p  // .p converts Boolean to ProfessBoolean
```

### `ProfessInterpreter[T]` (Interpreter Trait)

```scala
trait ProfessInterpreter[T]:
  def interpret(expr: ProfessExpr): T = interpret(expr.toIR)
  
  def interpret(ir: IRNode): T = ir match
    case IRObject(kind, name) => interpretObject(kind, name)
    case IRWord(word) => interpretWord(word)
    // ... dispatch to abstract methods
  
  // Abstract methods for each node type
  def interpretObject(kind: String, name: String): T
  def interpretWord(word: String): T
  def interpretNumber(value: Double): T
  // ...
```

**Purpose**: Template for creating interpreters over PROFESS IR.

**Usage Pattern**:
```scala
object MyInterpreter extends ProfessInterpreter[MyResult]:
  def interpretObject(kind: String, name: String): MyResult = ???
  def interpretWord(word: String): MyResult = ???
  // ...
```

### Utility Functions

```scala
def sequence(exprs: ProfessExpr*): ProfessExpr
def combine(exprs: ProfessExpr*): ProfessExpr  // alias
def let(name: String, expr: ProfessExpr): ProfessExpr
def ref(name: String): ProfessExpr
def tuple(exprs: ProfessExpr*): ProfessExpr

def foldIR[T](node: IRNode)(z: T)(f: (T, IRNode) => T): T
def collectNodes[T](node: IRNode)(pf: PartialFunction[IRNode, T]): List[T]
def extractObjects(expr: ProfessExpr): List[(String, String)]
def extractWords(expr: ProfessExpr): List[String]
def extractNumbers(expr: ProfessExpr): List[Double]
```

**Purpose**: Composition and introspection utilities.

---

## Cats Integration Module

Located in `profess.runtime.effect`, provides functional effect-based interpretation.

### `TransformContext`

```scala
case class TransformContext(
  bindings: Map[String, Any] = Map.empty,
  metadata: Map[String, String] = Map.empty,
  depth: Int = 0
):
  def bind(name: String, value: Any): TransformContext
  def withMeta(key: String, value: String): TransformContext
  def descend: TransformContext
```

**Purpose**: Carries state through transformations.

### Handler Traits

```scala
trait WordHandler[F[_], A]:
  def word: String
  def handle(args: List[IRNode], ctx: TransformContext): F[A]

trait ObjectHandler[F[_], A]:
  def kind: String
  def handle(name: String, ctx: TransformContext): F[A]
```

**Purpose**: Define how to handle specific words and object kinds.

**Type Parameters**:
- `F[_]`: Effect type (e.g., `IO`, `StateT[IO, S, *]`)
- `A`: Result type

### `HandlerRegistry[F[_], A]`

```scala
class HandlerRegistry[F[_]: Monad, A](
  val wordHandlers: Map[String, WordHandler[F, A]],
  val objectHandlers: Map[String, ObjectHandler[F, A]],
  val defaultWordHandler: Option[(String, List[IRNode], TransformContext) => F[A]],
  val defaultObjectHandler: Option[(String, String, TransformContext) => F[A]]
):
  def withWordHandler(handler: WordHandler[F, A]): HandlerRegistry[F, A]
  def withObjectHandler(handler: ObjectHandler[F, A]): HandlerRegistry[F, A]
```

**Purpose**: Collects handlers for use during IR traversal.

### `IRTraverser[F[_], A]`

```scala
class IRTraverser[F[_]: Monad, A: Monoid](registry: HandlerRegistry[F, A]):
  def traverse(expr: ProfessExpr, ctx: TransformContext): F[A]
  def traverseNode(node: IRNode, ctx: TransformContext): F[A]
  def traverseSequence(nodes: List[IRNode], ctx: TransformContext): F[A]
```

**Purpose**: Walks IR and applies registered handlers.

**Traversal Logic**:
1. Match each node against handlers
2. For sequences, detect `word + args` patterns
3. Combine results using `Monoid[A].combine`

### `Executable[A]` (Code Representation)

```scala
sealed trait Executable[+A]

object Executable:
  case object NoOp extends Executable[Nothing]
  case class Pure[A](value: A) extends Executable[A]
  case class Effect[A](run: IO[A]) extends Executable[A]
  case class Sequence[A](steps: List[Executable[A]]) extends Executable[A]
  
  def execute[A](exec: Executable[A])(using default: A): IO[A]
```

**Purpose**: Represents executable code produced by transformation.

### `HandlerDSL` (Builder Pattern)

```scala
object HandlerDSL:
  def handlers[F[_]: Monad, A]: RegistryBuilder[F, A]
  
  class RegistryBuilder[F[_]: Monad, A](registry: HandlerRegistry[F, A]):
    def onWord(word: String)(f: (List[IRNode], TransformContext) => F[A]): RegistryBuilder[F, A]
    def onObject(kind: String)(f: (String, TransformContext) => F[A]): RegistryBuilder[F, A]
    def onAnyWord(f: (String, List[IRNode], TransformContext) => F[A]): RegistryBuilder[F, A]
    def build: HandlerRegistry[F, A]
```

**Usage**:
```scala
val registry = HandlerDSL.handlers[IO, Unit]
  .onObject("broker") { (name, ctx) => IO.println(s"Broker: $name") }
  .onWord("sold") { (args, ctx) => IO.println(s"Sold: $args") }
  .build
```

---

## Execution Flow

### Compile-Time (Plugin)

```
Source Code
    │
    ▼
┌─────────────────┐
│  Scala Parser   │
└─────────────────┘
    │
    ▼ (Untyped AST)
┌─────────────────┐
│  ProfessPhase   │
│                 │
│  1. DeclarationCollector  ──► Collect declared ids
│  2. ExpressionCollector   ──► Find PROFESS patterns
│  3. ScaffoldGenerator     ──► Generate val defs
│  4. ASTInjector           ──► Insert into AST
└─────────────────┘
    │
    ▼ (Modified Untyped AST)
┌─────────────────┐
│  Scala Typer    │
└─────────────────┘
    │
    ▼
  Typed AST
```

### Run-Time (Evaluation)

```
PROFESS Expression
    │
    ▼
┌─────────────────┐
│  ProfessKind    │ ◄── val broker = ProfessKind("broker")
└─────────────────┘
    │ (broker Mark)
    ▼
┌─────────────────┐
│  ProfessObject  │ ◄── broker.apply(Mark)
└─────────────────┘
    │ .sold
    ▼
┌─────────────────┐
│  ProfessExpr    │ ◄── selectDynamic("sold")
└─────────────────┘
    │ (700)
    ▼
┌─────────────────┐
│  ProfessExpr    │ ◄── apply(700)
└─────────────────┘
    │ .toIR
    ▼
┌─────────────────┐
│   IRSequence    │
│  [IRObject,     │
│   IRWord,       │
│   IRNumber]     │
└─────────────────┘
```

### Interpretation Flow

```
ProfessExpr
    │
    ▼ toIR
  IRNode
    │
    ▼
┌─────────────────────┐
│    IRTraverser      │
│                     │
│  ┌───────────────┐  │
│  │HandlerRegistry│  │
│  │  wordHandlers │  │
│  │  objectHandlers│ │
│  └───────────────┘  │
└─────────────────────┘
    │
    ▼ traverse
  F[A] (Effect)
    │
    ▼ unsafeRunSync (or similar)
  Result
```

---

## Adding New Features

### Adding a New IR Node Type

1. **Define the case class**:
```scala
case class IRNewNode(field1: Type1, field2: Type2) extends IRNode:
  def render: String = s"new($field1, $field2)"
```

2. **Add to interpreter dispatch**:
```scala
// In ProfessInterpreter
def interpret(ir: IRNode): T = ir match
  // ... existing cases
  case IRNewNode(f1, f2) => interpretNewNode(f1, f2)

def interpretNewNode(f1: Type1, f2: Type2): T
```

3. **Add to visitor**:
```scala
// In IRVisitor
def visit(node: IRNode): Unit = node match
  // ... existing cases
  case IRNewNode(f1, f2) => visitNewNode(f1, f2)

def visitNewNode(f1: Type1, f2: Type2): Unit = ()
```

4. **Add to foldIR if needed**:
```scala
def foldIR[T](node: IRNode)(z: T)(f: (T, IRNode) => T): T =
  // ... existing cases
  case IRNewNode(f1, f2) => // handle children if any
```

### Adding a New Scaffolding Type

1. **Define in Runtime.scala**:
```scala
class ProfessNewType(val field: String) extends ProfessExpr(List(IRNewNode(field))):
  // ...

object ProfessNewType:
  def apply(field: String): ProfessNewType = new ProfessNewType(field)
```

2. **Add to IdKind enum** (if new classification):
```scala
enum IdKind:
  case Kind
  case Name
  case Word
  case NewKind  // New classification
```

3. **Update ScaffoldGenerator**:
```scala
val typeName = kind match
  case IdKind.Kind => "ProfessKind"
  case IdKind.Name => "ProfessName"
  case IdKind.Word => "ProfessWord"
  case IdKind.NewKind => "ProfessNewType"
```

### Adding a New Handler Type

1. **Define trait**:
```scala
trait NewHandler[F[_], A]:
  def criterion: SomeCriterion
  def handle(data: SomeData, ctx: TransformContext): F[A]
```

2. **Add to registry**:
```scala
class HandlerRegistry[F[_]: Monad, A](
  // ... existing handlers
  val newHandlers: Map[SomeCriterion, NewHandler[F, A]]
)
```

3. **Add to traverser**:
```scala
// In IRTraverser.traverseNode
case IRSomeNode(data) =>
  registry.findNewHandler(criterion) match
    case Some(handler) => handler.handle(data, ctx)
    case None => Monoid[A].empty.pure[F]
```

---

## Summary

PROFESS is architected around these key concepts:

1. **Compile-Time Scaffolding**: Plugin generates runtime types for undeclared identifiers
2. **First-Class Values**: All expressions evaluate to `ProfessExpr` values
3. **Intermediate Representation**: IR enables deferred, pluggable semantics
4. **Functional Effects**: Cats integration enables pure, composable interpretation

The separation of compile-time (plugin) and run-time (runtime) concerns enables:
- Clean module boundaries
- Independent publishing
- Flexible interpretation strategies
- Type-safe expression construction
