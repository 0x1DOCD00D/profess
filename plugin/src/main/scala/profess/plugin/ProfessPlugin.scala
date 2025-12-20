/*
 * Copyright (c) 2025 Mark Grechanik and Lone Star Consulting, Inc. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.
 */

package profess.plugin

import dotty.tools.dotc.ast.{tpd, untpd}
import dotty.tools.dotc.ast.untpd.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Decorators.*
import dotty.tools.dotc.core.Names.*
import dotty.tools.dotc.core.Flags.*
import dotty.tools.dotc.core.Constants.Constant
import dotty.tools.dotc.plugins.*
import dotty.tools.dotc.report

import scala.annotation.nowarn
import scala.collection.mutable

/**
 * PROFESS COMPILER PLUGIN
 *
 * Programming Rule Oriented Formalized English Sentence Specifications
 *
 * KEY FEATURES:
 *   - Expressions evaluate to first-class values (ProfessExpr)
 *   - Scope-aware: only scaffolds identifiers NOT declared in scope
 *   - Conservative: when in doubt, leaves to Scala compiler
 *
 * SCOPE CHECKING:
 *   - Collects all declared identifiers (val, var, def, class, object, params, imports)
 *   - Only generates scaffolding for identifiers that are:
 *     1. Not Scala keywords
 *     2. Not declared anywhere in the current scope
 *     3. Appear in PROFESS expression patterns
 */

class ProfessPlugin extends StandardPlugin:
  val name: String = "profess"
  override val description: String = "PROFESS - Programming Rule Oriented Formalized English Sentence Specifications"

  @nowarn("msg=deprecated")
  override def init(options: List[String]): List[PluginPhase] =
    List(new ProfessPhase)


/**
 * PROFESS Transform Phase
 *
 * Runs after parser, before typer.
 */
class ProfessPhase extends PluginPhase:

  val phaseName: String = "profess"

  override val runsAfter: Set[String] = Set("parser")
  override val runsBefore: Set[String] = Set("typer")

  override def prepareForUnit(tree: tpd.Tree)(using Context): Context =
    val unit = ctx.compilationUnit
    val untypedTree = unit.untpdTree

    if untypedTree != null then
      // 1. Collect ALL declared identifiers in the compilation unit
      val declCollector = DeclarationCollector()
      declCollector.traverse(untypedTree)
      val declared = declCollector.result

      // 2. Collect candidate identifiers from PROFESS expressions
      val exprCollector = ExpressionCollector(declared)
      exprCollector.traverse(untypedTree)
      val candidates = exprCollector.result

      if candidates.nonEmpty then
        // 3. Filter to only truly undeclared identifiers
        val toScaffold = candidates.filter { case (id, _) =>
          !declared.contains(id) && !isScalaKeyword(id)
        }

        if toScaffold.nonEmpty then
          val kinds = toScaffold.filter(_._2 == IdKind.Kind).keys.toList.sorted
          val names = toScaffold.filter(_._2 == IdKind.Name).keys.toList.sorted
          val words = toScaffold.filter(_._2 == IdKind.Word).keys.toList.sorted

          report.inform(s"[PROFESS] Processing: ${ctx.source.name}")
          if kinds.nonEmpty then
            report.inform(s"[PROFESS]   Kinds: ${kinds.mkString(", ")}")
          if names.nonEmpty then
            report.inform(s"[PROFESS]   Names: ${names.mkString(", ")}")
          if words.nonEmpty then
            report.inform(s"[PROFESS]   Words: ${words.mkString(", ")}")

          // 4. Generate scaffolding
          val scaffolding = ScaffoldGenerator.generate(toScaffold)

          // 5. Inject into AST
          val transformed = ASTInjector.inject(untypedTree, scaffolding, declared)

          unit.untpdTree = transformed

    ctx

  private def isScalaKeyword(name: String): Boolean =
    scalaKeywords.contains(name)

  private val scalaKeywords = Set(
    "abstract", "case", "catch", "class", "def", "do", "else", "extends",
    "false", "final", "finally", "for", "forSome", "if", "implicit",
    "import", "lazy", "match", "new", "null", "object", "override",
    "package", "private", "protected", "return", "sealed", "super",
    "this", "throw", "trait", "true", "try", "type", "val", "var",
    "while", "with", "yield", "given", "using", "then", "enum", "export",
    "end", "infix", "inline", "opaque", "open", "transparent", "derives"
  )


/** Identifier classification */
enum IdKind:
  case Kind   // lowercase in (kindId name) position
  case Name   // in name position of (kindId name)
  case Word   // standalone word


/**
 * Collects ALL declared identifiers in the compilation unit.
 *
 * This is used to determine what should NOT be scaffolded.
 */
class DeclarationCollector(using Context) extends UntypedTreeTraverser:

  private val declared = mutable.Set[String]()

  def result: Set[String] = declared.toSet

  override def traverse(tree: Tree)(using Context): Unit =
    tree match
      // val/var definitions - the defined name
      case ValDef(name, _, _) =>
        declared += name.toString
        traverseChildren(tree)

      // def definitions - method name and parameters
      case DefDef(name, paramss, _, _) =>
        declared += name.toString
        paramss.foreach { params =>
          params.foreach {
            case vd: ValDef => declared += vd.name.toString
            case td: TypeDef => declared += td.name.toString
          }
        }
        traverseChildren(tree)

      // class/trait definitions
      case TypeDef(name, _) =>
        declared += name.toString
        traverseChildren(tree)

      // object definitions
      case ModuleDef(name, _) =>
        declared += name.toString
        traverseChildren(tree)

      // imports - both direct and renamed
      case Import(_, selectors) =>
        selectors.foreach { sel =>
          val imported = sel.name.toString
          declared += imported
          // If there's a rename, add that too
          sel match
            case untpd.ImportSelector(_, renamed, _) if renamed != EmptyTree =>
              renamed match
                case Ident(rname) => declared += rname.toString
                case _ => ()
            case _ => ()
        }
        traverseChildren(tree)

      // Pattern bindings (in match, for comprehensions)
      case Bind(name, _) =>
        declared += name.toString
        traverseChildren(tree)

      // For comprehension generators (Scala 3.6+ has 3 params)
      case GenFrom(pat, _, _) =>
        collectPatternNames(pat)
        traverseChildren(tree)

      case _ =>
        traverseChildren(tree)

  private def collectPatternNames(tree: Tree): Unit = tree match
    case Bind(name, body) =>
      declared += name.toString
      collectPatternNames(body)
    case Apply(_, args) =>
      args.foreach(collectPatternNames)
    case Tuple(elems) =>
      elems.foreach(collectPatternNames)
    case _ => ()


/**
 * Collects candidate identifiers from PROFESS expressions.
 *
 * Only collects identifiers that might need scaffolding.
 */
class ExpressionCollector(declared: Set[String])(using Context) extends UntypedTreeTraverser:

  private val candidates = mutable.Map[String, IdKind]()

  def result: Map[String, IdKind] = candidates.toMap

  override def traverse(tree: Tree)(using Context): Unit =
    tree match
      // Skip inside val/def names (but traverse RHS)
      case vd @ ValDef(_, tpt, _) =>
        traverse(tpt)
        traverse(vd.rhs)

      case dd @ DefDef(_, paramss, tpt, _) =>
        paramss.foreach { params =>
          params.foreach {
            case vd: ValDef => traverse(vd.tpt)
            case td: TypeDef => traverse(td.rhs)
          }
        }
        traverse(tpt)
        traverse(dd.rhs)

      // PROFESS object pattern: (kindId name)
      // Parses as: Apply(Ident(kindId), List(Ident(name)))
      case Apply(Ident(kindId), List(Ident(name)))
        if isKindCandidate(kindId.toString) =>
        val kindStr = kindId.toString
        val nameStr = name.toString

        // Only add if not declared
        if !declared.contains(kindStr) && !isScalaKeyword(kindStr) then
          candidates.getOrElseUpdate(kindStr, IdKind.Kind)
        if !declared.contains(nameStr) && !isScalaKeyword(nameStr) then
          candidates.getOrElseUpdate(nameStr, IdKind.Name)

        traverseChildren(tree)

      // Standalone identifier - potential PROFESS word
      case Ident(name) =>
        val nameStr = name.toString
        if !declared.contains(nameStr) &&
          !isScalaKeyword(nameStr) &&
          !candidates.contains(nameStr) &&
          isWordCandidate(nameStr) then
          candidates(nameStr) = IdKind.Word
      // Don't traverse children of Ident

      // Method call on identifier - could be PROFESS chaining
      case Apply(Select(qual, _), args) =>
        traverse(qual)
        args.foreach(traverse)

      case Select(qual, _) =>
        traverse(qual)

      case _ =>
        traverseChildren(tree)

  /** Check if this could be a kind identifier (lowercase, not keyword) */
  private def isKindCandidate(name: String): Boolean =
    name.nonEmpty &&
      name.head.isLower &&
      !isScalaKeyword(name)

  /** Check if this could be a PROFESS word */
  private def isWordCandidate(name: String): Boolean =
    name.nonEmpty &&
      !commonScalaIds.contains(name) &&
      !name.startsWith("_")

  private val commonScalaIds = Set(
    // Common methods
    "println", "print", "printf", "main", "args", "apply", "unapply", "update",
    "toString", "hashCode", "equals", "getClass", "wait", "notify", "notifyAll",
    "synchronized", "clone", "finalize",
    // Collection methods
    "map", "flatMap", "filter", "foreach", "fold", "foldLeft", "foldRight",
    "reduce", "reduceLeft", "reduceRight", "collect", "collectFirst",
    "head", "tail", "last", "init", "isEmpty", "nonEmpty", "size", "length",
    "take", "drop", "slice", "splitAt", "takeWhile", "dropWhile", "span",
    "find", "exists", "forall", "contains", "indexOf", "lastIndexOf",
    "zip", "zipWithIndex", "unzip", "flatten", "distinct", "sorted", "reverse",
    "mkString", "toList", "toSeq", "toSet", "toMap", "toArray", "toVector",
    "groupBy", "partition", "count", "sum", "product", "min", "max",
    // Common types (as identifiers)
    "Some", "None", "Left", "Right", "Nil", "Seq", "List", "Set", "Map",
    "Vector", "Array", "Range", "Option", "Either", "Try", "Future",
    "Int", "Long", "Double", "Float", "String", "Boolean", "Char", "Byte", "Short",
    "Unit", "Any", "AnyRef", "AnyVal", "Nothing", "Null",
    // Boolean literals (already keywords but be safe)
    "true", "false", "null",
    // Common implicits
    "implicitly", "summon", "the",
    // Predef
    "require", "assert", "assume", "identity", "locally"
  )

  private def isScalaKeyword(name: String): Boolean =
    scalaKeywords.contains(name)

  private val scalaKeywords = Set(
    "abstract", "case", "catch", "class", "def", "do", "else", "extends",
    "false", "final", "finally", "for", "forSome", "if", "implicit",
    "import", "lazy", "match", "new", "null", "object", "override",
    "package", "private", "protected", "return", "sealed", "super",
    "this", "throw", "trait", "true", "try", "type", "val", "var",
    "while", "with", "yield", "given", "using", "then", "enum", "export",
    "end", "infix", "inline", "opaque", "open", "transparent", "derives"
  )


/**
 * Generates scaffolding ValDefs for PROFESS identifiers.
 */
object ScaffoldGenerator:

  def generate(identifiers: Map[String, IdKind])(using Context): List[ValDef] =
    identifiers.toList.sortBy(_._1).map { (id, kind) =>
      generateValDef(id, kind)
    }

  private def generateValDef(id: String, kind: IdKind)(using Context): ValDef =
    val typeName = kind match
      case IdKind.Kind => "ProfessKind"
      case IdKind.Name => "ProfessName"
      case IdKind.Word => "ProfessWord"

    // val id = _root_.profess.runtime.TypeName("id")
    ValDef(
      name = id.toTermName,
      tpt = TypeTree(),
      rhs = Apply(
        Select(
          Select(
            Select(
              Ident("_root_".toTermName),
              "profess".toTermName
            ),
            "runtime".toTermName
          ),
          typeName.toTermName
        ),
        List(Literal(Constant(id)))
      )
    ).withFlags(Synthetic)


/**
 * Injects scaffolding into the AST.
 */
object ASTInjector:

  def inject(tree: Tree, scaffolding: List[ValDef], declared: Set[String])(using Context): Tree =
    if scaffolding.isEmpty then return tree

    object Injector extends UntypedTreeMap:
      override def transform(tree: Tree)(using Context): Tree =
        tree match
          case md @ ModuleDef(name, impl) =>
            val newImpl = injectIntoTemplate(impl, scaffolding, declared)
            ModuleDef(name, newImpl).withSpan(md.span)

          case td @ TypeDef(name, rhs: Template) =>
            val newRhs = injectIntoTemplate(rhs, scaffolding, declared)
            cpy.TypeDef(tree)(name, newRhs)

          case _ =>
            super.transform(tree)

    Injector.transform(tree)

  private def injectIntoTemplate(
                                  template: Template,
                                  scaffolding: List[ValDef],
                                  declared: Set[String]
                                )(using Context): Template =
    // Check if this template uses any PROFESS patterns
    if !containsProfessExpressions(template.body) then return template

    // Get identifiers actually used in this template
    val usedIds = collectUsedIdentifiers(template.body)

    // Filter scaffolding to only what's needed and not declared in this scope
    val localDeclared = collectLocalDeclarations(template.body)
    val relevantScaffolding = scaffolding.filter { vd =>
      val name = vd.name.toString
      usedIds.contains(name) &&
        !localDeclared.contains(name) &&
        !declared.contains(name)
    }

    if relevantScaffolding.isEmpty then return template

    cpy.Template(template)(
      constr = template.constr,
      parents = template.parents,
      derived = template.derived,
      self = template.self,
      body = relevantScaffolding ++ template.body
    )

  private def containsProfessExpressions(stats: List[Tree])(using Context): Boolean =
    var found = false
    object Checker extends UntypedTreeTraverser:
      override def traverse(tree: Tree)(using Context): Unit =
        if !found then
          tree match
            case Apply(Ident(name), List(Ident(_))) if name.toString.head.isLower =>
              found = true
            case _ =>
              traverseChildren(tree)
    stats.foreach(Checker.traverse)
    found

  private def collectUsedIdentifiers(stats: List[Tree])(using Context): Set[String] =
    val ids = mutable.Set[String]()
    object Collector extends UntypedTreeTraverser:
      override def traverse(tree: Tree)(using Context): Unit =
        tree match
          case Ident(name) =>
            ids += name.toString
          case _ =>
            traverseChildren(tree)
    stats.foreach(Collector.traverse)
    ids.toSet

  private def collectLocalDeclarations(stats: List[Tree])(using Context): Set[String] =
    val decls = mutable.Set[String]()
    stats.foreach {
      case ValDef(name, _, _) => decls += name.toString
      case DefDef(name, _, _, _) => decls += name.toString
      case _ => ()
    }
    decls.toSet
