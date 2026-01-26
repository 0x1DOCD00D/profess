# PROFESS Implementation Plan: Detailed Task Descriptions

## Part A: Scala 3 Compiler Plugin

**Goal:** Create a compiler plugin that intercepts compilation, detects natural language constructs that cannot type-check, and generates supporting Scala scaffolding to satisfy the type checker while producing IR.

**Duration:** 48-56 hours (2 developers × 3-4 weeks part-time)

---

### PROF-P01: Plugin Project Scaffolding and SBT Configuration
**Priority:** P0 | **Effort:** 4h | **Depends on:** None

**Description:**
Set up the complete build infrastructure for the compiler plugin subproject. Configure sbt for cross-compilation between the plugin (which must match compiler version) and runtime library. Establish the dependency relationship where the examples project depends on both.

**Acceptance Criteria:**
- [ ] Plugin subproject compiles against `scala3-compiler` library
- [ ] Runtime subproject compiles independently
- [ ] Examples subproject has plugin enabled via `addCompilerPlugin`
- [ ] `sbt plugin/package` produces a JAR with correct manifest
- [ ] Plugin JAR includes `plugin.properties` resource file
- [ ] `sbt examples/compile` invokes the plugin (verify via log output)

**Technical Notes:**
- Scala 3 plugins require matching the exact compiler version; use `scalaVersion := "3.6.2"`
- Plugin must be packaged before examples can use it; configure `dependsOn` carefully
- Use `scalacOptions += "-Xplugin:path/to/plugin.jar"` or sbt's `addCompilerPlugin`
- The plugin JAR must contain `META-INF/services/dotty.tools.dotc.plugins.Plugin` or use `plugin.properties`
- Consider using `sbt-assembly` if plugin has external dependencies

**Files:**
```
profess/
├── build.sbt                          # Root build with subprojects
├── project/
│   ├── build.properties               # sbt.version=1.10.x
│   └── plugins.sbt                    # Any sbt plugins needed
├── plugin/
│   ├── src/main/scala/profess/plugin/
│   │   └── ProfessPlugin.scala        # Plugin entry point (stub)
│   └── src/main/resources/
│       └── plugin.properties          # Plugin registration
├── runtime/
│   └── src/main/scala/profess/runtime/
│       └── package.scala              # Runtime exports (stub)
└── examples/
    └── src/main/scala/examples/
        └── BasicExample.scala         # Test file for plugin
```

**build.sbt Structure:**
```scala
val scala3Version = "3.6.2"

lazy val commonSettings = Seq(
  scalaVersion := scala3Version,
  organization := "io.github.0x1docd00d",
  version := "0.1.0-SNAPSHOT"
)

lazy val plugin = (project in file("plugin"))
  .settings(commonSettings)
  .settings(
    name := "profess-plugin",
    libraryDependencies ++= Seq(
      "org.scala-lang" %% "scala3-compiler" % scala3Version % "provided"
    ),
    Compile / resourceGenerators += Def.task {
      val file = (Compile / resourceManaged).value / "plugin.properties"
      IO.write(file, s"pluginClass=profess.plugin.ProfessPlugin\n")
      Seq(file)
    }.taskValue
  )

lazy val runtime = (project in file("runtime"))
  .settings(commonSettings)
  .settings(
    name := "profess-runtime",
    libraryDependencies ++= Seq(
      "org.typelevel" %% "cats-core" % "2.12.0",
      "org.typelevel" %% "cats-effect" % "3.5.4"
    )
  )

lazy val examples = (project in file("examples"))
  .dependsOn(runtime)
  .settings(commonSettings)
  .settings(
    name := "profess-examples",
    scalacOptions ++= {
      val jar = (plugin / Compile / packageBin).value
      Seq(s"-Xplugin:${jar.getAbsolutePath}", "-Xplugin-require:profess")
    }
  )

lazy val root = (project in file("."))
  .aggregate(plugin, runtime, examples)
  .settings(
    name := "profess",
    publish / skip := true
  )
```

---

### PROF-P02: Minimal Plugin Registration and Phase Ordering
**Priority:** P0 | **Effort:** 4h | **Depends on:** PROF-P01

**Description:**
Implement the minimal plugin class that registers with the Scala 3 compiler and inserts a custom phase. Verify the phase runs at the correct point in the compilation pipeline (after parsing, before typing). Add diagnostic logging to confirm execution.

**Acceptance Criteria:**
- [ ] Plugin class extends `StandardPlugin` and overrides required methods
- [ ] Plugin phase extends `PluginPhase` with correct `runsAfter`/`runsBefore`
- [ ] Compiling any Scala file with plugin enabled prints diagnostic message
- [ ] Phase runs exactly once per compilation unit
- [ ] Plugin accepts command-line options (e.g., `-P:profess:debug`)
- [ ] No compilation errors or warnings from plugin code itself

**Technical Notes:**
- Use `dotty.tools.dotc.plugins.{StandardPlugin, PluginPhase}`
- Phase ordering: `runsAfter = Set("parser")`, `runsBefore = Set("typer")`
- Access compilation context via `using Context` parameter
- Use `report.echo(msg)` for debug output, `report.warning(msg, pos)` for warnings
- Plugin options come as `List[String]` in `init` method

**Files:**
- `plugin/src/main/scala/profess/plugin/ProfessPlugin.scala`
- `plugin/src/main/scala/profess/plugin/ProfessPhase.scala`

```scala
// ProfessPlugin.scala
package profess.plugin

import dotty.tools.dotc.plugins.*

class ProfessPlugin extends StandardPlugin:
  val name: String = "profess"
  override val description: String = "PROFESS: Natural language DSL scaffolding generator"
  
  var debug: Boolean = false
  
  override def init(options: List[String]): List[PluginPhase] =
    debug = options.contains("debug")
    List(new ProfessPhase(debug))
  
  override val optionsHelp: Option[String] = Some(
    """  -P:profess:debug      Enable debug output
      |  -P:profess:dump-ast   Dump transformed AST to console""".stripMargin
  )
```

```scala
// ProfessPhase.scala
package profess.plugin

import dotty.tools.dotc.*
import dotty.tools.dotc.ast.Trees.*
import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.ast.untpd
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Phases.*
import dotty.tools.dotc.plugins.*
import dotty.tools.dotc.report

class ProfessPhase(debug: Boolean) extends PluginPhase:
  val phaseName: String = "professScaffold"
  
  override val runsAfter: Set[String] = Set("parser")
  override val runsBefore: Set[String] = Set("typer")
  
  override def transformUnit(tree: tpd.Tree)(using Context): tpd.Tree =
    if debug then
      report.echo(s"[PROFESS] Processing: ${ctx.compilationUnit.source.file.name}")
    
    // TODO: Actual transformation
    tree
```

---

### PROF-P03: AST Printer and Structure Analyzer
**Priority:** P0 | **Effort:** 6h | **Depends on:** PROF-P02

**Description:**
Implement a comprehensive AST printer that outputs the structure of parsed Scala code in a human-readable format. This is essential for understanding what patterns we need to detect and how to generate scaffolding. Support both untyped (pre-typer) and typed AST output.

**Acceptance Criteria:**
- [ ] `-P:profess:dump-ast` option enables AST dumping
- [ ] Output shows tree node types, names, and positions
- [ ] Nested structures are properly indented
- [ ] Can distinguish between `Ident`, `Apply`, `Select`, `Literal` nodes
- [ ] Output includes source position (line:column) for each node
- [ ] Both untyped (`untpd.Tree`) and transformed trees can be printed

**Technical Notes:**
- Work with `untpd.Tree` since we're before typer phase
- Use pattern matching with `case` classes from `dotty.tools.dotc.ast.untpd`
- Key node types: `Ident`, `Apply`, `Select`, `Literal`, `ValDef`, `DefDef`, `TypeDef`
- Position available via `tree.span` and `ctx.source.atSpan(span)`
- Consider outputting to a file for large ASTs

**Files:**
- `plugin/src/main/scala/profess/plugin/ASTPrinter.scala`
- `plugin/src/main/scala/profess/plugin/ProfessPhase.scala` (update)

```scala
// ASTPrinter.scala
package profess.plugin

import dotty.tools.dotc.ast.untpd.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.util.SourcePosition

object ASTPrinter:
  def print(tree: Tree, indent: Int = 0)(using Context): String =
    val prefix = "  " * indent
    val pos = positionString(tree)
    val nodeInfo = tree match
      case Ident(name) => 
        s"Ident(${name.toString})"
      case Select(qual, name) => 
        s"Select(_, ${name.toString})\n${print(qual, indent + 1)}"
      case Apply(fun, args) =>
        val argsStr = args.map(print(_, indent + 2)).mkString("\n")
        s"Apply\n${print(fun, indent + 1)}\n${prefix}  args:\n$argsStr"
      case Literal(const) =>
        s"Literal(${const.value})"
      case ValDef(name, tpt, rhs) =>
        s"ValDef(${name.toString})\n${prefix}  tpt: ${print(tpt, indent + 1)}\n${prefix}  rhs: ${print(rhs, indent + 1)}"
      case Block(stats, expr) =>
        val statsStr = stats.map(print(_, indent + 1)).mkString("\n")
        s"Block\n$statsStr\n${print(expr, indent + 1)}"
      case PackageDef(pid, stats) =>
        val statsStr = stats.map(print(_, indent + 1)).mkString("\n")
        s"PackageDef(${pid})\n$statsStr"
      case ModuleDef(name, impl) =>
        s"ModuleDef(${name.toString})\n${print(impl, indent + 1)}"
      case Template(constr, parents, self, body) =>
        val bodyStr = body.map(print(_, indent + 1)).mkString("\n")
        s"Template\n$bodyStr"
      case other =>
        s"${other.getClass.getSimpleName}"
    
    s"$prefix[$pos] $nodeInfo"
  
  private def positionString(tree: Tree)(using Context): String =
    if tree.span.exists then
      val pos = ctx.source.atSpan(tree.span)
      s"${pos.line + 1}:${pos.column + 1}"
    else "?"
```

**Example Output:**
```
[1:1] PackageDef(examples)
  [3:1] ModuleDef(BasicExample)
    [3:1] Template
      [4:3] ValDef(trade)
        tpt: [4:3] TypeTree
        rhs: [4:15] Apply
          [4:15] Select(_, sold)
            [4:15] Apply
              [4:15] Ident(broker)
              args:
                [4:23] Ident(Mark)
          args:
            [4:33] Literal(700)
```

---

### PROF-P04: NL Construct Pattern Detection
**Priority:** P0 | **Effort:** 8h | **Depends on:** PROF-P03

**Description:**
Implement pattern recognition to detect natural language constructs that will fail type checking. Identify: (1) PROFESS object patterns `(identifier Identifier)`, (2) undefined lowercase identifiers (potential kinds/words), (3) undefined capitalized identifiers (potential names), (4) method chains on untyped expressions.

**Acceptance Criteria:**
- [ ] Detects `Apply(Ident(lowercase), List(Ident(Capitalized)))` as PROFESS object
- [ ] Collects all `Ident` nodes that are not in scope
- [ ] Distinguishes kinds (lowercase) from names (Capitalized) from words (lowercase after chain)
- [ ] Handles chained calls: `(broker Mark) sold 700` → detects `sold` as word
- [ ] Ignores standard library identifiers (`println`, `List`, etc.)
- [ ] Returns structured collection of detected patterns with positions

**Technical Notes:**
- Use `TreeAccumulator` to collect patterns in single traversal
- Pattern for PROFESS object: `Apply(Ident(name), args)` where `name` starts lowercase, `args` contains `Ident` starting uppercase
- Track context: is this `Ident` a call target, argument, or standalone?
- Build a "suspicious identifier" list for identifiers that don't resolve
- Consider: nested PROFESS expressions `(broker Mark) sold 700 (stock MSFT)`

**Files:**
- `plugin/src/main/scala/profess/plugin/PatternDetector.scala`
- `plugin/src/main/scala/profess/plugin/NLConstruct.scala` (data types)

```scala
// NLConstruct.scala
package profess.plugin

import dotty.tools.dotc.util.Spans.Span

/** Represents detected NL constructs */
sealed trait NLConstruct:
  def span: Span

/** PROFESS object: (kind Name) or (kind name1 name2) */
case class ProfessObject(
  kind: String,
  names: List[String],
  span: Span
) extends NLConstruct

/** Potential word in a chain: .sold, .bought */
case class ProfessWord(
  word: String,
  span: Span
) extends NLConstruct

/** Standalone identifier that might need scaffolding */
case class UnresolvedIdent(
  name: String,
  isCapitalized: Boolean,
  context: IdentContext,
  span: Span
) extends NLConstruct

enum IdentContext:
  case KindPosition      // First identifier in Apply: broker(Mark)
  case NamePosition      // Argument to kind: broker(Mark)
  case ChainPosition     // After dot: expr.sold
  case ArgumentPosition  // Numeric/other argument: sold(700)
  case Standalone        // Not in any special context
```

```scala
// PatternDetector.scala
package profess.plugin

import dotty.tools.dotc.ast.untpd.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Names.*
import scala.collection.mutable

class PatternDetector:
  
  def detect(tree: Tree)(using Context): DetectionResult =
    val accumulator = new PatternAccumulator
    accumulator.traverse(tree)
    accumulator.result
  
  private class PatternAccumulator(using Context) extends TreeTraverser:
    val professObjects = mutable.ListBuffer[ProfessObject]()
    val professWords = mutable.ListBuffer[ProfessWord]()
    val unresolvedIdents = mutable.ListBuffer[UnresolvedIdent]()
    
    def result: DetectionResult = DetectionResult(
      professObjects.toList,
      professWords.toList,
      unresolvedIdents.toList
    )
    
    override def traverse(tree: Tree)(using Context): Unit = tree match
      // Pattern: (broker Mark) -> Apply(Ident(broker), List(Ident(Mark)))
      case Apply(Ident(kindName), args) 
        if isLowerCase(kindName) && args.nonEmpty && hasCapitalizedIdent(args) =>
          val names = args.collect { case Ident(n) => n.toString }
          professObjects += ProfessObject(kindName.toString, names, tree.span)
          // Don't traverse into args; they're part of this construct
      
      // Pattern: expr.word or expr.word(args)
      case Select(qual, name) if isLowerCase(name) =>
        professWords += ProfessWord(name.toString, tree.span)
        traverse(qual)
      
      // Standalone identifier
      case Ident(name) if !isBuiltin(name) =>
        val ctx = determineContext(tree)
        unresolvedIdents += UnresolvedIdent(
          name.toString,
          isCapitalized(name),
          ctx,
          tree.span
        )
      
      case _ => 
        traverseChildren(tree)
    
    private def isLowerCase(name: Name): Boolean =
      name.toString.headOption.exists(_.isLower)
    
    private def isCapitalized(name: Name): Boolean =
      name.toString.headOption.exists(_.isUpper)
    
    private def hasCapitalizedIdent(trees: List[Tree]): Boolean =
      trees.exists {
        case Ident(n) => isCapitalized(n)
        case _ => false
      }
    
    private def isBuiltin(name: Name): Boolean =
      val builtins = Set("println", "print", "List", "Map", "Set", 
                         "Some", "None", "true", "false", "null")
      builtins.contains(name.toString)
    
    private def determineContext(tree: Tree)(using Context): IdentContext =
      // TODO: Analyze parent nodes to determine context
      IdentContext.Standalone

case class DetectionResult(
  professObjects: List[ProfessObject],
  professWords: List[ProfessWord],
  unresolvedIdents: List[UnresolvedIdent]
)
```

---

### PROF-P05: Declaration Scope Analysis
**Priority:** P0 | **Effort:** 6h | **Depends on:** PROF-P04

**Description:**
Implement scope analysis to determine which identifiers are already declared and should NOT be scaffolded. Track val/var/def/class/object/import declarations at each scope level. Handle nested scopes, imports (including wildcards), and shadowing correctly.

**Acceptance Criteria:**
- [ ] Collects all `ValDef`, `DefDef`, `TypeDef`, `ClassDef`, `ModuleDef` names
- [ ] Processes `Import` statements (specific and wildcard)
- [ ] Handles nested scopes (blocks, methods, classes)
- [ ] Returns whether an identifier at a given position is already in scope
- [ ] Test: `val broker = 1; val x = broker(Mark)` → only `Mark` needs scaffolding
- [ ] Test: `import profess.runtime._` makes runtime types available

**Technical Notes:**
- Build scope stack during traversal; push on enter, pop on exit
- `Import` node contains `List[untpd.ImportSelector]` with names
- Wildcard import: `ImportSelector` with `name == nme.WILDCARD`
- Consider: companion objects, package objects, inherited members
- For MVP, focus on local declarations; imports are harder

**Files:**
- `plugin/src/main/scala/profess/plugin/ScopeAnalyzer.scala`

```scala
// ScopeAnalyzer.scala
package profess.plugin

import dotty.tools.dotc.ast.untpd.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Names.*
import dotty.tools.dotc.util.Spans.Span
import scala.collection.mutable

class ScopeAnalyzer:
  
  def analyze(tree: Tree)(using Context): ScopeInfo =
    val collector = new ScopeCollector
    collector.traverse(tree)
    collector.scopeInfo
  
  private class ScopeCollector(using Context) extends TreeTraverser:
    private val scopes = mutable.Stack[mutable.Set[String]]()
    private val globalDeclarations = mutable.Set[String]()
    private val declarationSpans = mutable.Map[String, Span]()
    
    // Track all declarations with their scope depth
    private val scopedDeclarations = mutable.ListBuffer[(String, Int, Span)]()
    
    scopes.push(mutable.Set.empty) // Global scope
    
    def scopeInfo: ScopeInfo = ScopeInfo(
      globalDeclarations.toSet,
      declarationSpans.toMap,
      scopedDeclarations.toList
    )
    
    override def traverse(tree: Tree)(using Context): Unit = tree match
      case ValDef(name, _, _) =>
        addDeclaration(name.toString, tree.span)
        traverseChildren(tree)
      
      case DefDef(name, _, _, _) =>
        addDeclaration(name.toString, tree.span)
        // New scope for method body
        withNewScope { traverseChildren(tree) }
      
      case TypeDef(name, _) =>
        addDeclaration(name.toString, tree.span)
        traverseChildren(tree)
      
      case ModuleDef(name, _) =>
        addDeclaration(name.toString, tree.span)
        withNewScope { traverseChildren(tree) }
      
      case ClassDef(name, _, _) =>
        addDeclaration(name.toString, tree.span)
        withNewScope { traverseChildren(tree) }
      
      case Import(expr, selectors) =>
        processImport(expr, selectors)
        traverseChildren(tree)
      
      case Block(stats, expr) =>
        withNewScope {
          stats.foreach(traverse)
          traverse(expr)
        }
      
      case _ =>
        traverseChildren(tree)
    
    private def addDeclaration(name: String, span: Span): Unit =
      scopes.top += name
      if scopes.size == 1 then
        globalDeclarations += name
      declarationSpans.getOrElseUpdate(name, span)
      scopedDeclarations += ((name, scopes.size - 1, span))
    
    private def withNewScope[T](body: => T): T =
      scopes.push(mutable.Set.empty)
      try body
      finally scopes.pop()
    
    private def processImport(expr: Tree, selectors: List[ImportSelector])(using Context): Unit =
      // For MVP: track imported names but don't resolve them
      selectors.foreach { sel =>
        val name = sel.name.toString
        if name != "_" then
          addDeclaration(name, sel.span)
      }
    
    def isDeclaredAt(name: String, atSpan: Span): Boolean =
      scopedDeclarations.exists { case (n, _, declSpan) =>
        n == name && declSpan.start < atSpan.start
      }

case class ScopeInfo(
  globalDeclarations: Set[String],
  declarationSpans: Map[String, Span],
  scopedDeclarations: List[(String, Int, Span)]
):
  def isDeclared(name: String): Boolean = 
    globalDeclarations.contains(name)
  
  def isDeclaredBefore(name: String, position: Span): Boolean =
    declarationSpans.get(name).exists(_.start < position.start)
```

---

### PROF-P06: Scaffolding AST Generator
**Priority:** P0 | **Effort:** 8h | **Depends on:** PROF-P04, PROF-P05

**Description:**
Generate the AST nodes for scaffolding code that will make PROFESS expressions type-check. For each detected pattern, generate appropriate `ValDef` nodes with the correct types (`ProfessKind`, `ProfessName`, `ProfessWord`). Also generate the necessary import statement.

**Acceptance Criteria:**
- [ ] Generates `val broker = new ProfessKind("broker")` for kind identifiers
- [ ] Generates `val Mark = new ProfessName("Mark")` for name identifiers
- [ ] Generates `val sold = new ProfessWord("sold")` for word identifiers
- [ ] Generates `import profess.runtime._` at the top of the file
- [ ] Generated AST is syntactically valid (no typer errors from scaffolding itself)
- [ ] Handles duplicates (same identifier used multiple times → one declaration)

**Technical Notes:**
- Use `untpd` node constructors: `ValDef`, `Ident`, `Apply`, `Select`, `New`
- Assign synthetic spans to generated code (use source file span or `Span.NoSpan`)
- Import generation: `Import(Ident(profess), List(ImportSelector(runtime, _)))`
- Consider using `lazy val` to avoid initialization order issues
- Must handle name collisions: if user declares `broker`, don't scaffold it

**Files:**
- `plugin/src/main/scala/profess/plugin/ScaffoldGenerator.scala`

```scala
// ScaffoldGenerator.scala
package profess.plugin

import dotty.tools.dotc.ast.untpd.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Names.*
import dotty.tools.dotc.core.StdNames.*
import dotty.tools.dotc.util.Spans.Span

class ScaffoldGenerator:
  
  def generate(
    detected: DetectionResult,
    declared: ScopeInfo
  )(using Context): GeneratedScaffolding =
    val needed = computeNeededScaffolding(detected, declared)
    
    val importStmt = generateImport()
    val valDefs = needed.flatMap(generateValDef)
    
    GeneratedScaffolding(importStmt, valDefs)
  
  private def computeNeededScaffolding(
    detected: DetectionResult,
    declared: ScopeInfo
  ): List[ScaffoldingItem] =
    val items = mutable.Set[ScaffoldingItem]()
    
    // From PROFESS objects: kind and names
    detected.professObjects.foreach { obj =>
      if !declared.isDeclared(obj.kind) then
        items += ScaffoldingItem(obj.kind, ScaffoldType.Kind)
      obj.names.foreach { name =>
        if !declared.isDeclared(name) then
          items += ScaffoldingItem(name, ScaffoldType.Name)
      }
    }
    
    // From words
    detected.professWords.foreach { word =>
      if !declared.isDeclared(word.word) then
        items += ScaffoldingItem(word.word, ScaffoldType.Word)
    }
    
    // From unresolved identifiers
    detected.unresolvedIdents.foreach { ident =>
      if !declared.isDeclared(ident.name) then
        val scaffoldType = ident.context match
          case IdentContext.KindPosition => ScaffoldType.Kind
          case IdentContext.NamePosition => ScaffoldType.Name
          case IdentContext.ChainPosition => ScaffoldType.Word
          case _ if ident.isCapitalized => ScaffoldType.Name
          case _ => ScaffoldType.Word
        items += ScaffoldingItem(ident.name, scaffoldType)
    }
    
    items.toList
  
  private def generateImport()(using Context): Import =
    // import profess.runtime._
    Import(
      Select(Ident(termName("profess")), termName("runtime")),
      List(ImportSelector(nme.WILDCARD))
    )
  
  private def generateValDef(item: ScaffoldingItem)(using Context): Option[ValDef] =
    val rhs = item.scaffoldType match
      case ScaffoldType.Kind =>
        // new ProfessKind("broker")
        Apply(
          Select(New(Ident(typeName("ProfessKind"))), nme.CONSTRUCTOR),
          List(Literal(Constant(item.name)))
        )
      case ScaffoldType.Name =>
        // new ProfessName("Mark")
        Apply(
          Select(New(Ident(typeName("ProfessName"))), nme.CONSTRUCTOR),
          List(Literal(Constant(item.name)))
        )
      case ScaffoldType.Word =>
        // new ProfessWord("sold")
        Apply(
          Select(New(Ident(typeName("ProfessWord"))), nme.CONSTRUCTOR),
          List(Literal(Constant(item.name)))
        )
    
    Some(ValDef(termName(item.name), TypeTree(), rhs))

enum ScaffoldType:
  case Kind, Name, Word

case class ScaffoldingItem(name: String, scaffoldType: ScaffoldType)

case class GeneratedScaffolding(
  importStmt: Import,
  valDefs: List[ValDef]
)
```

---

### PROF-P07: AST Transformation and Insertion
**Priority:** P0 | **Effort:** 6h | **Depends on:** PROF-P06

**Description:**
Combine all components into the main transformation phase. Insert generated scaffolding at the correct location in the AST (after imports, before other declarations). Handle various source file structures (objects, classes, packages).

**Acceptance Criteria:**
- [ ] Scaffolding inserted at top of module/class body
- [ ] Import inserted before scaffolding
- [ ] Original user code is unchanged
- [ ] Works with `object`, `class`, and package-level code
- [ ] Multiple PROFESS expressions in same file handled correctly
- [ ] Compilation succeeds for valid PROFESS expressions

**Technical Notes:**
- Use `TreeMap` to transform specific nodes while preserving others
- `Template` node contains the body of classes/objects; insert there
- `PackageDef` for package-level code
- Preserve source positions for error messages pointing to user code
- Consider: should scaffolding go in a synthetic companion object?

**Files:**
- `plugin/src/main/scala/profess/plugin/ProfessTransformer.scala`
- `plugin/src/main/scala/profess/plugin/ProfessPhase.scala` (update)

```scala
// ProfessTransformer.scala
package profess.plugin

import dotty.tools.dotc.ast.untpd.*
import dotty.tools.dotc.ast.Trees.*
import dotty.tools.dotc.core.Contexts.*

class ProfessTransformer(debug: Boolean):
  
  def transform(tree: Tree)(using Context): Tree =
    val detector = new PatternDetector
    val scopeAnalyzer = new ScopeAnalyzer
    val generator = new ScaffoldGenerator
    
    // Step 1: Detect NL constructs
    val detected = detector.detect(tree)
    if debug then
      report.echo(s"[PROFESS] Detected: ${detected.professObjects.size} objects, " +
                  s"${detected.professWords.size} words, " +
                  s"${detected.unresolvedIdents.size} unresolved")
    
    // Step 2: Analyze existing declarations
    val scopeInfo = scopeAnalyzer.analyze(tree)
    if debug then
      report.echo(s"[PROFESS] Declared: ${scopeInfo.globalDeclarations.mkString(", ")}")
    
    // Step 3: Generate scaffolding
    val scaffolding = generator.generate(detected, scopeInfo)
    if debug then
      report.echo(s"[PROFESS] Generated: ${scaffolding.valDefs.size} scaffolding declarations")
    
    // Step 4: Insert into AST
    if scaffolding.valDefs.isEmpty then tree
    else insertScaffolding(tree, scaffolding)
  
  private def insertScaffolding(tree: Tree, scaffolding: GeneratedScaffolding)(using Context): Tree =
    val mapper = new TreeMap:
      override def transform(tree: Tree)(using Context): Tree = tree match
        // For object/class: insert into Template body
        case Template(constr, parents, self, body) =>
          val newBody = scaffolding.importStmt :: scaffolding.valDefs ::: body
          Template(constr, parents, self, newBody)
        
        // For package: insert after existing imports
        case PackageDef(pid, stats) =>
          val (imports, rest) = stats.partition(_.isInstanceOf[Import])
          val newStats = imports ::: (scaffolding.importStmt :: scaffolding.valDefs) ::: rest
          PackageDef(pid, newStats)
        
        case other => super.transform(other)
    
    mapper.transform(tree)
```

```scala
// Updated ProfessPhase.scala
package profess.plugin

import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.plugins.*

class ProfessPhase(debug: Boolean) extends PluginPhase:
  val phaseName: String = "professScaffold"
  
  override val runsAfter: Set[String] = Set("parser")
  override val runsBefore: Set[String] = Set("typer")
  
  override def transformUnit(tree: tpd.Tree)(using Context): tpd.Tree =
    if debug then
      report.echo(s"[PROFESS] Processing: ${ctx.compilationUnit.source.file.name}")
    
    val transformer = new ProfessTransformer(debug)
    
    // Note: We're working with untyped trees despite the tpd.Tree signature
    // This is because we're before the typer phase
    val untypedTree = tree.asInstanceOf[dotty.tools.dotc.ast.untpd.Tree]
    val transformed = transformer.transform(untypedTree)
    transformed.asInstanceOf[tpd.Tree]
```

---

### PROF-P08: IR Construction from AST
**Priority:** P0 | **Effort:** 6h | **Depends on:** PROF-P06, Runtime IR types

**Description:**
Ensure that the scaffolding types in the runtime library correctly construct IR when PROFESS expressions are evaluated. The scaffolded code must produce `ProfessExpr` values containing the appropriate `IRNode` structures.

**Acceptance Criteria:**
- [ ] `ProfessKind("broker")(ProfessName("Mark"))` produces `ProfessExpr(IRObject("broker", "Mark"))`
- [ ] Method chaining produces `IRSequence` with correct nodes
- [ ] Numeric literals produce `IRNumber` nodes
- [ ] String interpolation produces correct IR
- [ ] Runtime compiles and works with scaffolding generated by plugin
- [ ] Integration test: plugin + runtime produces expected IR

**Technical Notes:**
- This is primarily runtime work, but must coordinate with plugin
- `ProfessKind` extends `Dynamic` for method call interception
- `ProfessExpr` extends `Dynamic` for chaining
- Implicit conversions handle `Int`/`Double` → `ProfessExpr`
- Test by compiling with plugin and inspecting produced IR at runtime

**Files:**
- `runtime/src/main/scala/profess/runtime/IR.scala`
- `runtime/src/main/scala/profess/runtime/ProfessExpr.scala`
- `runtime/src/main/scala/profess/runtime/Scaffolding.scala`
- `runtime/src/test/scala/profess/runtime/IRConstructionSpec.scala`

```scala
// Scaffolding.scala - Runtime scaffolding types
package profess.runtime

import scala.language.dynamics

class ProfessKind(val kind: String) extends Dynamic:
  /** (broker Mark) - Apply with ProfessName */
  def apply(name: ProfessName): ProfessExpr =
    ProfessExpr(List(IRObject(kind, name.name)))
  
  /** (broker !variable) - Apply with interpolated value */
  def apply(ref: ProfessRef): ProfessExpr =
    ProfessExpr(List(IRReference(ref.variable)))
  
  /** broker Mark - Select style (if plugin generates it) */
  def selectDynamic(name: String): ProfessExpr =
    ProfessExpr(List(IRObject(kind, name)))
  
  /** broker(Mark, Jane) - Multiple names */
  def applyDynamic(method: String)(args: Any*): ProfessExpr =
    val names = args.collect { case n: ProfessName => n.name }.toList
    if names.nonEmpty then
      ProfessExpr(names.map(n => IRObject(kind, n)))
    else
      ProfessExpr(List(IRObject(kind, method)))

class ProfessName(val name: String):
  override def toString: String = s"ProfessName($name)"

class ProfessWord(val word: String):
  def toIR: IRNode = IRWord(word)
  override def toString: String = s"ProfessWord($word)"

class ProfessRef(val variable: String):
  override def toString: String = s"ProfessRef($variable)"
```

---

### PROF-P09: Sample Project and End-to-End Test
**Priority:** P0 | **Effort:** 4h | **Depends on:** PROF-P07, PROF-P08

**Description:**
Create a complete sample Scala 3 project that uses the PROFESS plugin. Demonstrate the full workflow: write natural language expressions, compile with plugin, and verify correct IR is produced at runtime.

**Acceptance Criteria:**
- [ ] Sample project compiles without manual scaffolding
- [ ] Trading example: `(broker Mark) sold 700 (stock MSFT)` works
- [ ] Multiple expressions in same file work
- [ ] Variable interpolation works
- [ ] Runtime prints produced IR for verification
- [ ] Clear documentation of how to use the plugin

**Technical Notes:**
- Create as separate sbt project or subproject
- Include both simple and complex examples
- Add assertions that verify IR structure
- Document any limitations or known issues
- Include negative test cases (intentional errors)

**Files:**
```
examples/
├── src/main/scala/examples/
│   ├── BasicExample.scala       # Simple expressions
│   ├── TradingExample.scala     # Full trading DSL
│   ├── InterpolationExample.scala # Variable interpolation
│   └── ChainedExample.scala     # Complex chains
└── src/test/scala/examples/
    └── PluginIntegrationSpec.scala # Verify IR structure
```

```scala
// examples/src/main/scala/examples/TradingExample.scala
package examples

// No imports needed for broker, Mark, etc. - plugin scaffolds them!

object TradingExample:
  def main(args: Array[String]): Unit =
    // This should compile and produce ProfessExpr
    val trade1 = (broker Mark) sold 700 (stock MSFT)
    println(s"Trade 1 IR: ${trade1.toIR}")
    
    // Multiple trades
    val trade2 = (broker Jane) bought 500 (stock AAPL) at 150
    println(s"Trade 2 IR: ${trade2.toIR}")
    
    // With numeric units (if implemented)
    val trade3 = (broker Mark) sold 1000 shares (stock GOOG)
    println(s"Trade 3 IR: ${trade3.toIR}")
    
    // Verify structure
    assert(trade1.toIR match
      case IRSequence(nodes) => nodes.exists {
        case IRObject("broker", "Mark") => true
        case _ => false
      }
      case _ => false
    , "Expected broker Mark in IR")
```

---

## Part B: IR Interpreter and Handler System

**Goal:** Build a pluggable interpreter that traverses IR and generates executable code based on programmer-defined handlers, using Scala's given/using mechanism for dependency injection.

**Duration:** 40-48 hours (2 developers × 3-4 weeks part-time)

---

### PROF-I01: Handler Typeclass Definitions
**Priority:** P0 | **Effort:** 4h | **Depends on:** Runtime IR types

**Description:**
Define the core handler typeclasses that programmers implement to give meaning to PROFESS constructs. Create `WordHandler` for action words, `ObjectHandler` for PROFESS objects, `NumberHandler` for numeric values, and `UnitValueHandler` for unit-annotated values.

**Acceptance Criteria:**
- [ ] `WordHandler[F[_], A]` trait with `handle(word: String, args: List[IRNode], ctx: Context): F[A]`
- [ ] `ObjectHandler[F[_], A]` trait with `handle(kind: String, name: String, ctx: Context): F[A]`
- [ ] `NumberHandler[F[_], A]` trait with `handle(value: Double, ctx: Context): F[A]`
- [ ] `UnitValueHandler[F[_], A]` trait with `handle(value: Double, unit: String, ctx: Context): F[A]`
- [ ] Context provides access to parent node, siblings, position in sequence
- [ ] Handlers are contravariant in input, covariant in output where appropriate

**Technical Notes:**
- Use higher-kinded type `F[_]` for effect abstraction (IO, Future, Id)
- Context should be immutable; create new context for nested traversal
- Consider: should handlers return `Option[A]` or use fallback mechanism?
- Provide `Functor` instance for handlers to transform output type

**Files:**
- `runtime/src/main/scala/profess/runtime/cats/Handler.scala`
- `runtime/src/main/scala/profess/runtime/cats/Context.scala`

```scala
// Handler.scala
package profess.runtime.cats

import cats.Functor

/** Handler for words like "sold", "bought", "at" */
trait WordHandler[F[_], A]:
  def word: String
  def handle(args: List[IRNode], ctx: InterpreterContext): F[A]

object WordHandler:
  def apply[F[_], A](w: String)(f: (List[IRNode], InterpreterContext) => F[A]): WordHandler[F, A] =
    new WordHandler[F, A]:
      val word = w
      def handle(args: List[IRNode], ctx: InterpreterContext) = f(args, ctx)

/** Handler for PROFESS objects like (broker Mark) */
trait ObjectHandler[F[_], A]:
  def kind: String
  def handle(name: String, ctx: InterpreterContext): F[A]

object ObjectHandler:
  def apply[F[_], A](k: String)(f: (String, InterpreterContext) => F[A]): ObjectHandler[F, A] =
    new ObjectHandler[F, A]:
      val kind = k
      def handle(name: String, ctx: InterpreterContext) = f(name, ctx)

/** Handler for numeric literals */
trait NumberHandler[F[_], A]:
  def handle(value: Double, ctx: InterpreterContext): F[A]

/** Handler for unit values like 700:shares */
trait UnitValueHandler[F[_], A]:
  def handle(value: Double, unit: String, ctx: InterpreterContext): F[A]

/** Handler for conditionals */
trait ConditionalHandler[F[_], A]:
  def handle(
    condition: A,
    thenBranch: A,
    elseBranch: Option[A],
    ctx: InterpreterContext
  ): F[A]
```

```scala
// Context.scala
package profess.runtime.cats

/** Context passed to handlers during interpretation */
case class InterpreterContext(
  /** The parent IR node containing this node */
  parent: Option[IRNode],
  
  /** Sibling nodes at the same level */
  siblings: List[IRNode],
  
  /** Position of current node among siblings (0-indexed) */
  position: Int,
  
  /** The complete IR tree being interpreted */
  root: IRNode,
  
  /** Custom metadata that handlers can use */
  metadata: Map[String, Any]
):
  def withParent(node: IRNode): InterpreterContext =
    copy(parent = Some(node))
  
  def withSiblings(nodes: List[IRNode], pos: Int): InterpreterContext =
    copy(siblings = nodes, position = pos)
  
  def withMetadata(key: String, value: Any): InterpreterContext =
    copy(metadata = metadata + (key -> value))
  
  def getMetadata[T](key: String): Option[T] =
    metadata.get(key).map(_.asInstanceOf[T])

object InterpreterContext:
  def root(tree: IRNode): InterpreterContext =
    InterpreterContext(
      parent = None,
      siblings = Nil,
      position = 0,
      root = tree,
      metadata = Map.empty
    )
```

---

### PROF-I02: Handler Registry with Monoid Composition
**Priority:** P0 | **Effort:** 6h | **Depends on:** PROF-I01

**Description:**
Implement `HandlerRegistry` that stores handlers by their key (word name, object kind) and supports monoid composition. Later registries override earlier ones when combined. Provide empty registry as identity element.

**Acceptance Criteria:**
- [ ] Registry stores word handlers in `Map[String, WordHandler[F, A]]`
- [ ] Registry stores object handlers in `Map[String, ObjectHandler[F, A]]`
- [ ] `combine` merges registries with right-precedence
- [ ] `Monoid[HandlerRegistry[F, A]]` instance provided
- [ ] Monoid laws verified: identity (`empty |+| r == r`) and associativity
- [ ] Lookup methods return `Option[Handler]`

**Technical Notes:**
- Use `cats.Monoid` for composition
- `combine` should use `++` on maps (right map wins on conflicts)
- Consider adding `NumberHandler` and `UnitValueHandler` to registry
- Provide introspection: `registeredWords`, `registeredKinds`

**Files:**
- `runtime/src/main/scala/profess/runtime/cats/HandlerRegistry.scala`
- `runtime/src/test/scala/profess/runtime/cats/HandlerRegistrySpec.scala`

```scala
// HandlerRegistry.scala
package profess.runtime.cats

import cats.Monoid
import cats.syntax.all.*

final class HandlerRegistry[F[_], A] private (
  private val wordHandlers: Map[String, WordHandler[F, A]],
  private val objectHandlers: Map[String, ObjectHandler[F, A]],
  private val numberHandler: Option[NumberHandler[F, A]],
  private val unitValueHandler: Option[UnitValueHandler[F, A]],
  private val conditionalHandler: Option[ConditionalHandler[F, A]]
):
  def findWordHandler(word: String): Option[WordHandler[F, A]] =
    wordHandlers.get(word)
  
  def findObjectHandler(kind: String): Option[ObjectHandler[F, A]] =
    objectHandlers.get(kind)
  
  def getNumberHandler: Option[NumberHandler[F, A]] = numberHandler
  def getUnitValueHandler: Option[UnitValueHandler[F, A]] = unitValueHandler
  def getConditionalHandler: Option[ConditionalHandler[F, A]] = conditionalHandler
  
  def registeredWords: Set[String] = wordHandlers.keySet
  def registeredKinds: Set[String] = objectHandlers.keySet
  
  def combine(other: HandlerRegistry[F, A]): HandlerRegistry[F, A] =
    new HandlerRegistry(
      this.wordHandlers ++ other.wordHandlers,
      this.objectHandlers ++ other.objectHandlers,
      other.numberHandler.orElse(this.numberHandler),
      other.unitValueHandler.orElse(this.unitValueHandler),
      other.conditionalHandler.orElse(this.conditionalHandler)
    )
  
  def withWordHandler(handler: WordHandler[F, A]): HandlerRegistry[F, A] =
    new HandlerRegistry(
      wordHandlers + (handler.word -> handler),
      objectHandlers, numberHandler, unitValueHandler, conditionalHandler
    )
  
  def withObjectHandler(handler: ObjectHandler[F, A]): HandlerRegistry[F, A] =
    new HandlerRegistry(
      wordHandlers,
      objectHandlers + (handler.kind -> handler),
      numberHandler, unitValueHandler, conditionalHandler
    )

object HandlerRegistry:
  def empty[F[_], A]: HandlerRegistry[F, A] =
    new HandlerRegistry(Map.empty, Map.empty, None, None, None)
  
  given [F[_], A]: Monoid[HandlerRegistry[F, A]] with
    def empty = HandlerRegistry.empty[F, A]
    def combine(x: HandlerRegistry[F, A], y: HandlerRegistry[F, A]) = x.combine(y)
```

---

### PROF-I03: Fluent Handler DSL Builder
**Priority:** P0 | **Effort:** 6h | **Depends on:** PROF-I02

**Description:**
Create a fluent builder API for constructing handler registries. Support method chaining with `.onWord`, `.onObject`, `.onNumber`, etc. Ensure type inference works without explicit annotations in handler lambdas.

**Acceptance Criteria:**
- [ ] `HandlerDSL.handlers[IO, Trade]` starts builder with types fixed
- [ ] `.onWord("sold") { (args, ctx) => ... }` adds word handler
- [ ] `.onObject("broker") { (name, ctx) => ... }` adds object handler
- [ ] `.onNumber { (value, ctx) => ... }` sets number handler
- [ ] `.build` returns `HandlerRegistry[F, A]`
- [ ] Type inference works: no need to annotate lambda parameter types

**Technical Notes:**
- Use builder pattern with immutable state
- Consider phantom types to track what's been configured
- Lambda type: `(List[IRNode], InterpreterContext) => F[A]`
- Add convenience methods for common patterns (extracting numbers from args)

**Files:**
- `runtime/src/main/scala/profess/runtime/cats/HandlerDSL.scala`
- `runtime/src/test/scala/profess/runtime/cats/HandlerDSLSpec.scala`

```scala
// HandlerDSL.scala
package profess.runtime.cats

import cats.Applicative
import cats.syntax.all.*

final class HandlerDSL[F[_], A] private (
  private val wordHandlers: List[WordHandler[F, A]],
  private val objectHandlers: List[ObjectHandler[F, A]],
  private val numberHandler: Option[NumberHandler[F, A]],
  private val unitValueHandler: Option[UnitValueHandler[F, A]],
  private val conditionalHandler: Option[ConditionalHandler[F, A]]
):
  /** Add a word handler */
  def onWord(word: String)(handler: (List[IRNode], InterpreterContext) => F[A]): HandlerDSL[F, A] =
    new HandlerDSL(
      WordHandler(word)(handler) :: wordHandlers,
      objectHandlers, numberHandler, unitValueHandler, conditionalHandler
    )
  
  /** Add an object handler */
  def onObject(kind: String)(handler: (String, InterpreterContext) => F[A]): HandlerDSL[F, A] =
    new HandlerDSL(
      wordHandlers,
      ObjectHandler(kind)(handler) :: objectHandlers,
      numberHandler, unitValueHandler, conditionalHandler
    )
  
  /** Set the number handler */
  def onNumber(handler: (Double, InterpreterContext) => F[A]): HandlerDSL[F, A] =
    new HandlerDSL(
      wordHandlers, objectHandlers,
      Some(new NumberHandler[F, A] { def handle(v: Double, ctx: InterpreterContext) = handler(v, ctx) }),
      unitValueHandler, conditionalHandler
    )
  
  /** Set the unit value handler */
  def onUnitValue(handler: (Double, String, InterpreterContext) => F[A]): HandlerDSL[F, A] =
    new HandlerDSL(
      wordHandlers, objectHandlers, numberHandler,
      Some(new UnitValueHandler[F, A] { def handle(v: Double, u: String, ctx: InterpreterContext) = handler(v, u, ctx) }),
      conditionalHandler
    )
  
  /** Build the registry */
  def build: HandlerRegistry[F, A] =
    wordHandlers.foldLeft(HandlerRegistry.empty[F, A])((reg, h) => reg.withWordHandler(h))
      .pipe(reg => objectHandlers.foldLeft(reg)((r, h) => r.withObjectHandler(h)))
      // Add number/unit/conditional handlers...

object HandlerDSL:
  def handlers[F[_], A]: HandlerDSL[F, A] =
    new HandlerDSL(Nil, Nil, None, None, None)
  
  /** Convenience: extract first number from args */
  def extractNumber(args: List[IRNode]): Option[Double] =
    args.collectFirst { case IRNumber(n) => n }
  
  /** Convenience: extract first string from args */
  def extractString(args: List[IRNode]): Option[String] =
    args.collectFirst { case IRString(s) => s }
  
  /** Convenience: extract unit value from args */
  def extractUnitValue(args: List[IRNode]): Option[(Double, String)] =
    args.collectFirst { case IRUnitValue(v, u) => (v, u) }
```

---

### PROF-I04: IR Traverser with Effect Composition
**Priority:** P0 | **Effort:** 8h | **Depends on:** PROF-I02

**Description:**
Implement the core `IRTraverser` that walks the IR tree, invokes appropriate handlers, and combines results using `Monoid`. Handle all IR node types. Use `F[_]: Monad` for sequencing effects and `A: Monoid` for combining results.

**Acceptance Criteria:**
- [ ] Traverses `IRSequence` by visiting children and combining results
- [ ] Invokes `ObjectHandler` for `IRObject` nodes
- [ ] Invokes `WordHandler` for `IRWord` nodes with collected arguments
- [ ] Invokes `NumberHandler` or returns `empty` for `IRNumber`
- [ ] Handles `IRConditional` with `ConditionalHandler` or default behavior
- [ ] Unhandled nodes contribute `Monoid[A].empty`
- [ ] Context correctly reflects parent/sibling relationships

**Technical Notes:**
- Arguments to a word = subsequent nodes until next word or object
- Example: `sold 700 shares` → word "sold" gets args [IRNumber(700), IRWord("shares")]
- Use `Monad.flatMap` for sequencing, `Monoid.combine` for results
- Consider trampolining or `Eval` for stack safety on deep IR
- `IRReference` requires environment; defer to advanced implementation

**Files:**
- `runtime/src/main/scala/profess/runtime/cats/IRTraverser.scala`
- `runtime/src/test/scala/profess/runtime/cats/IRTraverserSpec.scala`

```scala
// IRTraverser.scala
package profess.runtime.cats

import cats.{Monad, Monoid}
import cats.syntax.all.*

class IRTraverser[F[_]: Monad, A: Monoid](registry: HandlerRegistry[F, A]):
  
  def traverse(expr: ProfessExpr): F[A] =
    traverseNode(expr.toIR, InterpreterContext.root(expr.toIR))
  
  def traverseNode(node: IRNode, ctx: InterpreterContext): F[A] = node match
    case IRObject(kind, name) =>
      registry.findObjectHandler(kind) match
        case Some(handler) => handler.handle(name, ctx)
        case None => 
          logUnhandled(s"object kind '$kind'")
          Monoid[A].empty.pure[F]
    
    case IRWord(word) =>
      registry.findWordHandler(word) match
        case Some(handler) =>
          val args = collectArguments(ctx)
          handler.handle(args, ctx)
        case None =>
          logUnhandled(s"word '$word'")
          Monoid[A].empty.pure[F]
    
    case IRNumber(value) =>
      registry.getNumberHandler match
        case Some(handler) => handler.handle(value, ctx)
        case None => Monoid[A].empty.pure[F]
    
    case IRString(value) =>
      // Strings typically handled as arguments to words
      Monoid[A].empty.pure[F]
    
    case IRUnitValue(value, unit) =>
      registry.getUnitValueHandler match
        case Some(handler) => handler.handle(value, unit, ctx)
        case None => Monoid[A].empty.pure[F]
    
    case IRSequence(nodes) =>
      traverseSequence(nodes, ctx)
    
    case IRConditional(condition, thenBranch, elseBranch) =>
      traverseConditional(condition, thenBranch, elseBranch, ctx)
    
    case IRReference(variable) =>
      // TODO: Requires runtime environment
      Monoid[A].empty.pure[F]
    
    case IRBinding(variable, value) =>
      // TODO: Update environment
      traverseNode(value, ctx)
  
  private def traverseSequence(nodes: List[IRNode], ctx: InterpreterContext): F[A] =
    nodes.zipWithIndex.traverse { case (node, idx) =>
      val nodeCtx = ctx
        .withParent(ctx.parent.getOrElse(IRSequence(nodes)))
        .withSiblings(nodes, idx)
      traverseNode(node, nodeCtx)
    }.map(_.combineAll)
  
  private def traverseConditional(
    condition: IRNode,
    thenBranch: IRNode,
    elseBranch: Option[IRNode],
    ctx: InterpreterContext
  ): F[A] =
    for
      condResult <- traverseNode(condition, ctx.withParent(IRConditional(condition, thenBranch, elseBranch)))
      thenResult <- traverseNode(thenBranch, ctx)
      elseResult <- elseBranch.traverse(traverseNode(_, ctx)).map(_.getOrElse(Monoid[A].empty))
    yield registry.getConditionalHandler match
      case Some(handler) => ??? // Need to handle differently
      case None => condResult |+| thenResult |+| elseResult
  
  /** Collect arguments for a word: nodes after current until next word/object */
  private def collectArguments(ctx: InterpreterContext): List[IRNode] =
    val remaining = ctx.siblings.drop(ctx.position + 1)
    remaining.takeWhile {
      case _: IRWord | _: IRObject => false
      case _ => true
    }
  
  private def logUnhandled(what: String): Unit =
    // TODO: Configurable logging
    ()

object IRTraverser:
  def apply[F[_]: Monad, A: Monoid](registry: HandlerRegistry[F, A]): IRTraverser[F, A] =
    new IRTraverser(registry)
```

---

### PROF-I05: Given/Using Handler Injection
**Priority:** P0 | **Effort:** 6h | **Depends on:** PROF-I04

**Description:**
Design the typeclass for domain handlers that can be injected via Scala's `given/using` mechanism. Create `DomainHandlers` typeclass and `runWithHandlers` function that automatically picks up handlers from scope.

**Acceptance Criteria:**
- [ ] `DomainHandlers[F, A]` typeclass wraps `HandlerRegistry[F, A]`
- [ ] `runWithHandlers[F, A](expr)(using handlers)` interprets expression
- [ ] `given DomainHandlers[IO, Trade]` in scope enables `runWithHandlers`
- [ ] Missing handlers produces compile error (no implicit found)
- [ ] Extension method `expr.interpret[A]` as alternative API
- [ ] Works with Cats Effect `IO` and pure `Id` effect

**Technical Notes:**
- Typeclass should require `Monad[F]` and `Monoid[A]` evidence
- Consider `MonadError[F, Throwable]` for error handling
- Provide `DomainHandlers.fromRegistry` constructor
- Add `run` extension on `ProfessExpr` for convenience

**Files:**
- `runtime/src/main/scala/profess/runtime/cats/DomainHandlers.scala`
- `runtime/src/main/scala/profess/runtime/cats/Interpreter.scala`
- `runtime/src/test/scala/profess/runtime/cats/InterpreterSpec.scala`

```scala
// DomainHandlers.scala
package profess.runtime.cats

import cats.{Monad, Monoid}

/** Typeclass providing handlers for a domain */
trait DomainHandlers[F[_], A]:
  def registry: HandlerRegistry[F, A]
  
  given Monad[F] = monad
  given Monoid[A] = monoid
  
  def monad: Monad[F]
  def monoid: Monoid[A]

object DomainHandlers:
  def apply[F[_], A](using dh: DomainHandlers[F, A]): DomainHandlers[F, A] = dh
  
  /** Create from registry with explicit typeclass instances */
  def fromRegistry[F[_]: Monad, A: Monoid](reg: HandlerRegistry[F, A]): DomainHandlers[F, A] =
    new DomainHandlers[F, A]:
      val registry = reg
      val monad = Monad[F]
      val monoid = Monoid[A]
  
  /** Create using HandlerDSL */
  def build[F[_]: Monad, A: Monoid](f: HandlerDSL[F, A] => HandlerDSL[F, A]): DomainHandlers[F, A] =
    fromRegistry(f(HandlerDSL.handlers[F, A]).build)
```

```scala
// Interpreter.scala
package profess.runtime.cats

import cats.{Monad, Monoid}
import cats.effect.IO
import cats.syntax.all.*

/** Main entry point for interpreting PROFESS expressions */
object Interpreter:
  
  /** Interpret using handlers from given scope */
  def runWithHandlers[F[_], A](expr: ProfessExpr)(using handlers: DomainHandlers[F, A]): F[A] =
    given Monad[F] = handlers.monad
    given Monoid[A] = handlers.monoid
    IRTraverser(handlers.registry).traverse(expr)
  
  /** Interpret with explicit registry */
  def runWithRegistry[F[_]: Monad, A: Monoid](
    expr: ProfessExpr,
    registry: HandlerRegistry[F, A]
  ): F[A] =
    IRTraverser(registry).traverse(expr)
  
  /** Interpret purely (no effects) */
  def runPure[A: Monoid](expr: ProfessExpr, registry: HandlerRegistry[cats.Id, A]): A =
    IRTraverser[cats.Id, A](registry).traverse(expr)

/** Extension methods for ProfessExpr */
extension (expr: ProfessExpr)
  def interpret[F[_], A](using handlers: DomainHandlers[F, A]): F[A] =
    Interpreter.runWithHandlers(expr)
  
  def interpretWith[F[_]: Monad, A: Monoid](registry: HandlerRegistry[F, A]): F[A] =
    Interpreter.runWithRegistry(expr, registry)
  
  def interpretPure[A: Monoid](registry: HandlerRegistry[cats.Id, A]): A =
    Interpreter.runPure(expr, registry)
```

---

### PROF-I06: Trading Domain Example
**Priority:** P0 | **Effort:** 6h | **Depends on:** PROF-I05, PROF-P09

**Description:**
Create a complete trading domain example demonstrating the full PROFESS workflow: natural language expressions compiled with the plugin, interpreted with custom handlers, producing domain objects. Include domain model, handlers, and execution.

**Acceptance Criteria:**
- [ ] `Trade` case class with broker, action, quantity, symbol, price fields
- [ ] `Monoid[Trade]` combines partial trades intelligently
- [ ] Handlers for "sold", "bought", "at", "broker", "stock"
- [ ] `(broker Mark) sold 700 (stock MSFT) at 150` produces correct `Trade`
- [ ] Multiple trades in sequence work
- [ ] Prints results showing complete trade objects

**Technical Notes:**
- Use `IO` as effect type for real-world example
- Monoid: non-empty fields in right operand override left
- Handler for "sold"/"bought" extracts quantity from args
- Handler for "at" extracts price from args
- Consider: partial application (trade without price is still valid)

**Files:**
- `examples/src/main/scala/examples/trading/TradeDomain.scala`
- `examples/src/main/scala/examples/trading/TradeHandlers.scala`
- `examples/src/main/scala/examples/trading/TradingApp.scala`

```scala
// TradeDomain.scala
package examples.trading

import cats.Monoid

/** Domain model for a trade */
case class Trade(
  broker: String = "",
  action: String = "",
  quantity: Int = 0,
  symbol: String = "",
  price: Double = 0.0
):
  def isComplete: Boolean =
    broker.nonEmpty && action.nonEmpty && quantity > 0 && symbol.nonEmpty

object Trade:
  val empty: Trade = Trade()
  
  /** Monoid: later values override earlier (non-empty wins) */
  given Monoid[Trade] with
    def empty = Trade.empty
    def combine(x: Trade, y: Trade) = Trade(
      broker   = if y.broker.nonEmpty then y.broker else x.broker,
      action   = if y.action.nonEmpty then y.action else x.action,
      quantity = if y.quantity > 0 then y.quantity else x.quantity,
      symbol   = if y.symbol.nonEmpty then y.symbol else x.symbol,
      price    = if y.price > 0 then y.price else x.price
    )
```

```scala
// TradeHandlers.scala
package examples.trading

import cats.effect.IO
import cats.Monoid
import profess.runtime.*
import profess.runtime.cats.*

object TradeHandlers:
  
  given DomainHandlers[IO, Trade] = DomainHandlers.build[IO, Trade] { dsl =>
    dsl
      .onObject("broker") { (name, ctx) =>
        IO.pure(Trade(broker = name))
      }
      .onObject("stock") { (name, ctx) =>
        IO.pure(Trade(symbol = name))
      }
      .onWord("sold") { (args, ctx) =>
        val qty = HandlerDSL.extractNumber(args).map(_.toInt).getOrElse(0)
        IO.pure(Trade(action = "sell", quantity = qty))
      }
      .onWord("bought") { (args, ctx) =>
        val qty = HandlerDSL.extractNumber(args).map(_.toInt).getOrElse(0)
        IO.pure(Trade(action = "buy", quantity = qty))
      }
      .onWord("at") { (args, ctx) =>
        val price = HandlerDSL.extractNumber(args).getOrElse(0.0)
        IO.pure(Trade(price = price))
      }
      .onWord("shares") { (_, _) =>
        IO.pure(Trade.empty) // "shares" is decorative
      }
  }
```

```scala
// TradingApp.scala
package examples.trading

import cats.effect.{IO, IOApp}
import profess.runtime.*
import profess.runtime.cats.*

object TradingApp extends IOApp.Simple:
  import TradeHandlers.given
  
  def run: IO[Unit] =
    for
      _ <- IO.println("=== PROFESS Trading Example ===\n")
      
      // Trade 1: Mark sells MSFT
      trade1 <- (broker Mark) sold 700 (stock MSFT) at 150.50
      _ <- IO.println(s"Trade 1: $trade1")
      _ <- IO.println(s"  Complete: ${trade1.isComplete}")
      
      // Trade 2: Jane buys AAPL
      trade2 <- (broker Jane) bought 500 shares (stock AAPL) at 175.25
      _ <- IO.println(s"\nTrade 2: $trade2")
      
      // Trade 3: Without price
      trade3 <- (broker Mark) sold 1000 (stock GOOG)
      _ <- IO.println(s"\nTrade 3 (no price): $trade3")
      _ <- IO.println(s"  Complete: ${trade3.isComplete}")
      
      _ <- IO.println("\n=== Done ===")
    yield ()
```

---

### PROF-I07: Process Algebra Example (π-Calculus Style)
**Priority:** P1 | **Effort:** 8h | **Depends on:** PROF-I05

**Description:**
Create a second example demonstrating PROFESS with a different domain: a simplified process algebra (π-calculus inspired). This shows that PROFESS is domain-agnostic and can model concurrent/distributed computations.

**Acceptance Criteria:**
- [ ] `Process` ADT with Send, Receive, Parallel, Sequential, Stop
- [ ] NL syntax: `(channel ping) sends "hello" then stop`
- [ ] NL syntax: `(channel pong) receives x then (channel ping) sends x`
- [ ] Parallel composition: `process1 parallel process2`
- [ ] Handlers produce `Process` IR that could be translated to Akka/FS2
- [ ] Example shows composition and execution

**Technical Notes:**
- This is a code generation example, not execution
- Handler produces `Process` ADT, not side effects
- Use `Id` or `Writer` effect to collect process structure
- Consider: generate Akka actor code as string output
- Simpler than full π-calculus: no restriction, no replication

**Files:**
- `examples/src/main/scala/examples/process/ProcessDomain.scala`
- `examples/src/main/scala/examples/process/ProcessHandlers.scala`
- `examples/src/main/scala/examples/process/ProcessExample.scala`

```scala
// ProcessDomain.scala
package examples.process

import cats.Monoid

/** Simple process algebra */
sealed trait Process

object Process:
  case object Stop extends Process
  case class Send(channel: String, message: String) extends Process
  case class Receive(channel: String, variable: String) extends Process
  case class Sequential(first: Process, second: Process) extends Process
  case class Parallel(left: Process, right: Process) extends Process
  case class Channel(name: String) extends Process
  
  val empty: Process = Stop
  
  /** Monoid: Sequential composition */
  given Monoid[Process] with
    def empty = Stop
    def combine(x: Process, y: Process) = (x, y) match
      case (Stop, p) => p
      case (p, Stop) => p
      case _ => Sequential(x, y)
  
  /** Pretty print process */
  def show(p: Process, indent: Int = 0): String =
    val pad = "  " * indent
    p match
      case Stop => s"${pad}STOP"
      case Send(ch, msg) => s"${pad}$ch ! $msg"
      case Receive(ch, v) => s"${pad}$ch ? $v"
      case Sequential(a, b) => s"${show(a, indent)}\n${pad}.\n${show(b, indent)}"
      case Parallel(a, b) => s"${show(a, indent)}\n${pad}|\n${show(b, indent)}"
      case Channel(n) => s"${pad}(channel $n)"
```

```scala
// ProcessHandlers.scala
package examples.process

import cats.Id
import profess.runtime.*
import profess.runtime.cats.*

object ProcessHandlers:
  import Process.*
  
  given DomainHandlers[Id, Process] = DomainHandlers.build[Id, Process] { dsl =>
    dsl
      .onObject("channel") { (name, ctx) =>
        Channel(name)
      }
      .onWord("sends") { (args, ctx) =>
        // Extract channel from context, message from args
        val channel = ctx.siblings.take(ctx.position).collectFirst {
          case IRObject("channel", name) => name
        }.getOrElse("unknown")
        val message = args.collectFirst {
          case IRString(s) => s
          case IRNumber(n) => n.toString
        }.getOrElse("")
        Send(channel, message)
      }
      .onWord("receives") { (args, ctx) =>
        val channel = ctx.siblings.take(ctx.position).collectFirst {
          case IRObject("channel", name) => name
        }.getOrElse("unknown")
        val variable = args.collectFirst {
          case IRWord(w) => w
        }.getOrElse("x")
        Receive(channel, variable)
      }
      .onWord("then") { (_, _) =>
        Stop // Continuation handled by monoid
      }
      .onWord("parallel") { (_, _) =>
        Stop // Would need special handling
      }
      .onWord("stop") { (_, _) =>
        Stop
      }
  }
```

```scala
// ProcessExample.scala
package examples.process

import profess.runtime.*
import profess.runtime.cats.*

object ProcessExample extends App:
  import ProcessHandlers.given
  import Process.*
  
  println("=== PROFESS Process Algebra Example ===\n")
  
  // Simple send
  val p1 = (channel ping) sends "hello" then stop
  println(s"Process 1:\n${show(p1)}\n")
  
  // Receive and forward
  val p2 = (channel pong) receives x then (channel ping) sends x then stop
  println(s"Process 2:\n${show(p2)}\n")
  
  // Note: Full parallel composition would need more sophisticated handling
  // This example shows the basic structure
  
  println("=== Potential Akka Translation ===")
  println(generateAkkaCode(p1))

  def generateAkkaCode(p: Process): String =
    p match
      case Send(ch, msg) => s"""$ch ! "$msg""""
      case Receive(ch, v) => s"""case $v => // from $ch"""
      case Sequential(a, b) => s"${generateAkkaCode(a)}\n${generateAkkaCode(b)}"
      case Stop => "context.stop(self)"
      case _ => "// ..."
```

---

## Summary: Combined Implementation Plan

| Task ID | Title | Priority | Effort | Dependencies |
|---------|-------|----------|--------|--------------|
| **Part A: Compiler Plugin** |
| PROF-P01 | Plugin Project Scaffolding | P0 | 4h | None |
| PROF-P02 | Plugin Registration & Phase Ordering | P0 | 4h | P01 |
| PROF-P03 | AST Printer and Analyzer | P0 | 6h | P02 |
| PROF-P04 | NL Construct Pattern Detection | P0 | 8h | P03 |
| PROF-P05 | Declaration Scope Analysis | P0 | 6h | P04 |
| PROF-P06 | Scaffolding AST Generator | P0 | 8h | P04, P05 |
| PROF-P07 | AST Transformation & Insertion | P0 | 6h | P06 |
| PROF-P08 | IR Construction from AST | P0 | 6h | P06, Runtime |
| PROF-P09 | Sample Project & E2E Test | P0 | 4h | P07, P08 |
| **Subtotal** | | | **52h** | |
| **Part B: Interpreter** |
| PROF-I01 | Handler Typeclass Definitions | P0 | 4h | Runtime IR |
| PROF-I02 | Handler Registry with Monoid | P0 | 6h | I01 |
| PROF-I03 | Fluent Handler DSL Builder | P0 | 6h | I02 |
| PROF-I04 | IR Traverser with Effects | P0 | 8h | I02 |
| PROF-I05 | Given/Using Handler Injection | P0 | 6h | I04 |
| PROF-I06 | Trading Domain Example | P0 | 6h | I05, P09 |
| PROF-I07 | Process Algebra Example | P1 | 8h | I05 |
| **Subtotal** | | | **44h** | |
| **Total** | | | **96h** | |

---

## Development Timeline

### Week 1-2: Foundation (Both Developers)
- **Developer A:** PROF-P01, PROF-P02, PROF-P03 (Plugin infrastructure)
- **Developer B:** Runtime IR types, PROF-I01, PROF-I02 (Handler foundation)

### Week 3-4: Core Implementation
- **Developer A:** PROF-P04, PROF-P05 (Pattern detection, scope analysis)
- **Developer B:** PROF-I03, PROF-I04 (DSL builder, traverser)

### Week 5-6: Integration
- **Developer A:** PROF-P06, PROF-P07 (Code generation, transformation)
- **Developer B:** PROF-I05, PROF-P08 (Handler injection, IR construction)

### Week 7-8: Examples and Testing
- **Developer A:** PROF-P09 (Plugin E2E testing)
- **Developer B:** PROF-I06, PROF-I07 (Domain examples)

---

## Critical Path

```
Runtime IR Types (prerequisite)
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
   PROF-P01 ──► P02 ──► P03 ──► P04 ──► P05   PROF-I01 ──► I02 ──► I03
                                  │                         │
                                  ▼                         ▼
                            PROF-P06 ◄─────────────────► PROF-I04
                                  │                         │
                                  ▼                         ▼
                            PROF-P07                   PROF-I05
                                  │                         │
                                  └──────────┬──────────────┘
                                             ▼
                                        PROF-P08
                                             │
                                             ▼
                                        PROF-P09
                                             │
                                             ▼
                                  PROF-I06 ──► PROF-I07
```
