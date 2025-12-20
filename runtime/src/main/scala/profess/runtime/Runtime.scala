/*
 * Copyright (c) 2025 Mark Grechanik and Lone Star Consulting, Inc. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.
 */

package profess.runtime

import scala.language.dynamics
import scala.collection.mutable

/**
 * PROFESS RUNTIME LIBRARY
 *
 * Programming Rule Oriented Formalized English Sentence Specifications
 *
 * KEY DIFFERENCE FROM FESS:
 *   - Every PROFESS expression evaluates to a FIRST-CLASS VALUE (ProfessExpr)
 *   - Values can be assigned, passed to functions, returned from functions
 *   - IR is encapsulated in the expression, not accumulated globally
 */

// ═══════════════════════════════════════════════════════════════════════════
// IR NODE TYPES - Pure data representation
// ═══════════════════════════════════════════════════════════════════════════

sealed trait IRNode:
  def render: String

case class IRObject(kind: String, name: String) extends IRNode:
  def render: String = s"($kind $name)"
  override def toString: String = render

case class IRWord(word: String) extends IRNode:
  def render: String = word
  override def toString: String = render

case class IRNumber(value: Double) extends IRNode:
  def render: String =
    if value == value.toLong then value.toLong.toString else value.toString
  override def toString: String = render

case class IRString(value: String) extends IRNode:
  def render: String = s""""$value""""
  override def toString: String = render

case class IRSequence(nodes: List[IRNode]) extends IRNode:
  def render: String = nodes.map(_.render).mkString(" ")
  override def toString: String = render

case class IRConditional(
                          condition: IRNode,
                          consequent: IRNode,
                          alternative: Option[IRNode]
                        ) extends IRNode:
  def render: String =
    val elseStr = alternative.map(a => s" else ${a.render}").getOrElse("")
    s"if (${condition.render}) then ${consequent.render}$elseStr"
  override def toString: String = render

case class IRBinding(name: String, value: IRNode) extends IRNode:
  def render: String = s"$name <- ${value.render}"
  override def toString: String = render

case class IRReference(name: String) extends IRNode:
  def render: String = s"@$name"
  override def toString: String = render

case class IRTuple(elements: List[IRNode]) extends IRNode:
  def render: String = s"(${elements.map(_.render).mkString(", ")})"
  override def toString: String = render

// Note: IRUnitValue, IRBoolean, IRParamBlock, IRAttributes are defined
// in the Variable Interpolation section above

// ═══════════════════════════════════════════════════════════════════════════
// VARIABLE INTERPOLATION - The ! operator
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Variable interpolation using ! prefix.
 *
 * Usage:
 *   val name = "Mark"
 *   val qty = 700
 *   (broker !name) sold !qty (stock !ticker)
 */

extension (s: String)
  /** Interpolate a String variable as a ProfessName */
  def unary_! : ProfessName = ProfessName(s)

extension (n: Int)
  /** Interpolate an Int variable as a ProfessNumber */
  def unary_! : ProfessNumber = ProfessNumber(n)
  /** Attach unit: 700:shares */
  def `:`(unit: ProfessWord): ProfessUnitValue = ProfessUnitValue(n.toDouble, unit.word)

extension (n: Long)
  /** Interpolate a Long variable as a ProfessNumber */
  def unary_! : ProfessNumber = ProfessNumber(n)
  /** Attach unit: 1000000:dollars */
  def `:`(unit: ProfessWord): ProfessUnitValue = ProfessUnitValue(n.toDouble, unit.word)

extension (n: Double)
  /** Interpolate a Double variable as a ProfessNumber */
  def unary_! : ProfessNumber = ProfessNumber(n)
  /** Attach unit: 150.50:dollars */
  def `:`(unit: ProfessWord): ProfessUnitValue = ProfessUnitValue(n, unit.word)

extension (b: Boolean)
  /** Interpolate a Boolean variable - use .p since ! is built-in negation */
  def p: ProfessBoolean = ProfessBoolean(b)

// ═══════════════════════════════════════════════════════════════════════════
// UNIT VALUES - The : operator for value:unit
// ═══════════════════════════════════════════════════════════════════════════

/**
 * A numeric value with an attached unit.
 *
 * Created by: 700:shares, 150:dollars
 */
class ProfessUnitValue(val value: Double, val unit: String)
  extends ProfessExpr(List(IRUnitValue(value, unit))):
  override def toString: String = s"$value:$unit"

object ProfessUnitValue:
  def apply(value: Double, unit: String): ProfessUnitValue =
    new ProfessUnitValue(value, unit)

/**
 * IR node for unit values
 */
case class IRUnitValue(value: Double, unit: String) extends IRNode:
  def render: String =
    val v = if value == value.toLong then value.toLong.toString else value.toString
    s"$v:$unit"
  override def toString: String = render

/**
 * Boolean wrapper for ! interpolation
 */
class ProfessBoolean(val value: Boolean) extends ProfessExpr(List(IRBoolean(value))):
  override def toString: String = value.toString

object ProfessBoolean:
  def apply(value: Boolean): ProfessBoolean = new ProfessBoolean(value)

case class IRBoolean(value: Boolean) extends IRNode:
  def render: String = value.toString
  override def toString: String = render

/**
 * Number wrapper for ! interpolation
 */
class ProfessNumber(val value: Double) extends ProfessExpr(List(IRNumber(value))):
  override def toString: String =
    if value == value.toLong then value.toLong.toString else value.toString

  /** Attach unit: !qty:shares */
  def `:`(unit: ProfessWord): ProfessUnitValue = ProfessUnitValue(value, unit.word)

object ProfessNumber:
  def apply(n: Int): ProfessNumber = new ProfessNumber(n.toDouble)
  def apply(n: Long): ProfessNumber = new ProfessNumber(n.toDouble)
  def apply(n: Double): ProfessNumber = new ProfessNumber(n)

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETER BLOCKS - The : operator for verb: params
// ═══════════════════════════════════════════════════════════════════════════

/**
 * A block of parameters attached to an expression.
 *
 * Created by: (order O123) execute: qty 700, price 150, venue NYSE
 */
case class IRParamBlock(params: List[(Option[String], IRNode)]) extends IRNode:
  def render: String =
    params.map {
      case (Some(name), value) => s"$name ${value.render}"
      case (None, value) => value.render
    }.mkString(", ")
  override def toString: String = render

/**
 * Attributes attached to an expression.
 *
 * Created by: (order O123): urgent, limit, day-only
 */
case class IRAttributes(target: IRNode, attrs: List[String]) extends IRNode:
  def render: String = s"${target.render}: ${attrs.mkString(", ")}"
  override def toString: String = render

// ═══════════════════════════════════════════════════════════════════════════
// PROFESS EXPRESSION - First-class value
// ═══════════════════════════════════════════════════════════════════════════

/**
 * The result of evaluating any PROFESS expression.
 *
 * This is a FIRST-CLASS VALUE that can be:
 *   - Assigned to variables: val x = expr
 *   - Passed to functions: process(expr)
 *   - Returned from functions: def f(): ProfessExpr = expr
 *   - Composed with other expressions: combine(expr1, expr2)
 */
class ProfessExpr(protected val nodes: List[IRNode]) extends Dynamic:

  /** Get the IR representation */
  def toIR: IRNode =
    if nodes.isEmpty then IRSequence(Nil)
    else if nodes.size == 1 then nodes.head
    else IRSequence(nodes)

  /** Get raw nodes list */
  def getNodes: List[IRNode] = nodes

  /** Check if empty */
  def isEmpty: Boolean = nodes.isEmpty

  /** Number of nodes */
  def size: Int = nodes.size

  // ─── Chaining via Dynamic ───

  /** Chaining: expr.word */
  def selectDynamic(word: String): ProfessExpr =
    ProfessExpr(nodes :+ IRWord(word))

  /** Chaining: expr.word(args) */
  def applyDynamic(word: String)(args: Any*): ProfessExpr =
    val wordNode = IRWord(word)
    val argNodes = args.flatMap(argToNodes).toList
    ProfessExpr((nodes :+ wordNode) ++ argNodes)

  /** Chaining: expr(args) */
  def apply(args: Any*): ProfessExpr =
    val argNodes = args.flatMap(argToNodes).toList
    ProfessExpr(nodes ++ argNodes)

  // ─── Conditional support ───

  /** Start conditional consequent: condition then result */
  def Then(consequent: ProfessExpr): ProfessConditional =
  ProfessConditional(this, consequent, None)

  // ─── Tuple construction ───

  /** Create tuple from this expression and others */
  def tuple(others: ProfessExpr*): ProfessExpr =
    val allNodes = this +: others
    ProfessExpr(IRTuple(allNodes.map(_.toIR).toList))

  // ─── Helper methods ───

  protected def argToNodes(arg: Any): List[IRNode] = arg match
    // Specific subtypes first (before ProfessExpr)
    case obj: ProfessObject => List(IRObject(obj.kind, obj.name))
    case word: ProfessWord => List(IRWord(word.word))
    case name: ProfessName => List(IRWord(name.value))
    // General ProfessExpr after subtypes
    case expr: ProfessExpr =>
      // Embed the entire expression's nodes
      if expr.getNodes.size == 1 then expr.getNodes
      else List(expr.toIR)
    case n: Int => List(IRNumber(n.toDouble))
    case n: Long => List(IRNumber(n.toDouble))
    case n: Double => List(IRNumber(n))
    case n: Float => List(IRNumber(n.toDouble))
    case s: String => List(IRString(s))
    case (a, b) =>
      List(IRTuple(List(anyToNode(a), anyToNode(b))))
    case (a, b, c) =>
      List(IRTuple(List(anyToNode(a), anyToNode(b), anyToNode(c))))
    case _ => Nil

  protected def anyToNode(a: Any): IRNode = a match
    // Specific subtypes first (before ProfessExpr)
    case obj: ProfessObject => IRObject(obj.kind, obj.name)
    case word: ProfessWord => IRWord(word.word)
    case name: ProfessName => IRWord(name.value)
    // General ProfessExpr after subtypes
    case expr: ProfessExpr => expr.toIR
    case n: Number => IRNumber(n.doubleValue)
    case s: String => IRString(s)
    case other => IRString(other.toString)

  override def toString: String = toIR.render

  override def equals(other: Any): Boolean = other match
    case that: ProfessExpr => this.nodes == that.nodes
    case _ => false

  override def hashCode: Int = nodes.hashCode

object ProfessExpr:
  def apply(nodes: List[IRNode]): ProfessExpr = new ProfessExpr(nodes)
  def apply(node: IRNode): ProfessExpr = new ProfessExpr(List(node))
  def empty: ProfessExpr = new ProfessExpr(Nil)

  /** Create from multiple expressions */
  def of(exprs: ProfessExpr*): ProfessExpr =
    ProfessExpr(exprs.flatMap(_.nodes).toList)

// ═══════════════════════════════════════════════════════════════════════════
// CONDITIONAL EXPRESSION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Represents a conditional PROFESS expression.
 *
 * Created by: If condition then consequent
 * Completed by: .otherwise(alternative) or .else_(alternative)
 */
class ProfessConditional(
                          val condition: ProfessExpr,
                          val consequent: ProfessExpr,
                          val alternative: Option[ProfessExpr]
                        ) extends ProfessExpr(Nil):

  override def toIR: IRNode =
    IRConditional(condition.toIR, consequent.toIR, alternative.map(_.toIR))

  override def getNodes: List[IRNode] = List(toIR)

  /** Add else branch: cond then result else_ alt */
  def else_(alt: ProfessExpr): ProfessExpr =
    ProfessExpr(IRConditional(condition.toIR, consequent.toIR, Some(alt.toIR)))

  /** Alternative name for else (since else is reserved) */
  def otherwise(alt: ProfessExpr): ProfessExpr = else_(alt)

  /** Chain more words after conditional (treated as alternative) */
  override def selectDynamic(word: String): ProfessExpr =
    // If no alternative yet, this starts the alternative
    if alternative.isEmpty then
      // Continue building what might become the alternative
      ProfessConditional(condition, consequent, Some(ProfessExpr(List(IRWord(word)))))
    else
      // Append to alternative
      val newAlt = alternative.map(a => ProfessExpr(a.getNodes :+ IRWord(word)))
      ProfessConditional(condition, consequent, newAlt)

  override def applyDynamic(word: String)(args: Any*): ProfessExpr =
    val wordNode = IRWord(word)
    val argNodes = args.flatMap(argToNodes).toList
    val newNodes = wordNode :: argNodes

    if alternative.isEmpty then
      ProfessConditional(condition, consequent, Some(ProfessExpr(newNodes)))
    else
      val newAlt = alternative.map(a => ProfessExpr(a.getNodes ++ newNodes))
      ProfessConditional(condition, consequent, newAlt)

// ═══════════════════════════════════════════════════════════════════════════
// SCAFFOLDING TYPES
// ═══════════════════════════════════════════════════════════════════════════

/**
 * PROFESS Object: represents (kindId name) patterns.
 *
 * Created when executing (kindId name) in source code.
 */
class ProfessObject(val kind: String, val name: String)
  extends ProfessExpr(List(IRObject(kind, name))):

  override def toString: String = s"($kind $name)"

object ProfessObject:
  def apply(kind: String, name: String): ProfessObject =
    new ProfessObject(kind, name)

/**
 * PROFESS Kind: creates objects via (kindId name) syntax.
 *
 * Generated by plugin for lowercase identifiers in (kindId name) patterns.
 */
class ProfessKind(val kindId: String) extends Dynamic:

  /** kindId(name) where name is ProfessName */
  def apply(name: ProfessName): ProfessObject =
    ProfessObject(kindId, name.value)

  /** kindId(name) where name is String */
  def apply(name: String): ProfessObject =
    ProfessObject(kindId, name)

  /** kindId.Name via Dynamic */
  def selectDynamic(name: String): ProfessObject =
    ProfessObject(kindId, name)

  override def toString: String = s"ProfessKind($kindId)"

object ProfessKind:
  def apply(kindId: String): ProfessKind = new ProfessKind(kindId)

/**
 * PROFESS Name: identifier for entities.
 *
 * Generated by plugin for the name part of (kindId name) patterns.
 */
class ProfessName(val value: String):
  override def toString: String = value

object ProfessName:
  def apply(value: String): ProfessName = new ProfessName(value)

/**
 * PROFESS Word: represents a word that starts or continues an expression.
 *
 * Generated by plugin for standalone words in PROFESS expressions.
 */
class ProfessWord(val word: String) extends ProfessExpr(List(IRWord(word))):
  override def toString: String = word

object ProfessWord:
  def apply(word: String): ProfessWord = new ProfessWord(word)

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/** Combine multiple expressions into a sequence */
def sequence(exprs: ProfessExpr*): ProfessExpr =
  ProfessExpr(IRSequence(exprs.map(_.toIR).toList))

/** Alias for sequence */
def combine(exprs: ProfessExpr*): ProfessExpr = sequence(exprs*)

/** Create a binding (name <- value) */
def let(name: String, expr: ProfessExpr): ProfessExpr =
  ProfessExpr(IRBinding(name, expr.toIR))

/** Reference a bound name */
def ref(name: String): ProfessExpr =
  ProfessExpr(IRReference(name))

/** Create a tuple */
def tuple(exprs: ProfessExpr*): ProfessExpr =
  ProfessExpr(IRTuple(exprs.map(_.toIR).toList))

// ═══════════════════════════════════════════════════════════════════════════
// INTERPRETER API
// ═══════════════════════════════════════════════════════════════════════════

/** Base trait for PROFESS interpreters */
trait ProfessInterpreter[T]:

  def interpret(expr: ProfessExpr): T = interpret(expr.toIR)

  def interpret(ir: IRNode): T = ir match
    case IRObject(kind, name) => interpretObject(kind, name)
    case IRWord(word) => interpretWord(word)
    case IRNumber(value) => interpretNumber(value)
    case IRString(value) => interpretString(value)
    case IRSequence(nodes) => interpretSequence(nodes)
    case IRConditional(cond, then_, else_) => interpretConditional(cond, then_, else_)
    case IRBinding(name, value) => interpretBinding(name, value)
    case IRReference(name) => interpretReference(name)
    case IRTuple(elements) => interpretTuple(elements)
    case IRUnitValue(value, unit) => interpretUnitValue(value, unit)
    case IRBoolean(value) => interpretBoolean(value)
    case IRParamBlock(params) => interpretParamBlock(params)
    case IRAttributes(target, attrs) => interpretAttributes(target, attrs)

  def interpretObject(kind: String, name: String): T
  def interpretWord(word: String): T
  def interpretNumber(value: Double): T
  def interpretString(value: String): T
  def interpretSequence(nodes: List[IRNode]): T
  def interpretConditional(cond: IRNode, then_ : IRNode, else_ : Option[IRNode]): T
  def interpretBinding(name: String, value: IRNode): T
  def interpretReference(name: String): T
  def interpretTuple(elements: List[IRNode]): T
  def interpretUnitValue(value: Double, unit: String): T
  def interpretBoolean(value: Boolean): T
  def interpretParamBlock(params: List[(Option[String], IRNode)]): T
  def interpretAttributes(target: IRNode, attrs: List[String]): T

/** Visitor pattern for IR traversal */
trait IRVisitor:

  def visit(node: IRNode): Unit = node match
    case IRObject(kind, name) => visitObject(kind, name)
    case IRWord(word) => visitWord(word)
    case IRNumber(value) => visitNumber(value)
    case IRString(value) => visitString(value)
    case IRSequence(nodes) => visitSequence(nodes)
    case IRConditional(cond, then_, else_) => visitConditional(cond, then_, else_)
    case IRBinding(name, value) => visitBinding(name, value)
    case IRReference(name) => visitReference(name)
    case IRTuple(elements) => visitTuple(elements)
    case IRUnitValue(value, unit) => visitUnitValue(value, unit)
    case IRBoolean(value) => visitBoolean(value)
    case IRParamBlock(params) => visitParamBlock(params)
    case IRAttributes(target, attrs) => visitAttributes(target, attrs)

  def visitObject(kind: String, name: String): Unit = ()
  def visitWord(word: String): Unit = ()
  def visitNumber(value: Double): Unit = ()
  def visitString(value: String): Unit = ()

  def visitSequence(nodes: List[IRNode]): Unit =
    nodes.foreach(visit)

  def visitConditional(cond: IRNode, then_ : IRNode, else_ : Option[IRNode]): Unit =
    visit(cond)
    visit(then_)
    else_.foreach(visit)

  def visitBinding(name: String, value: IRNode): Unit =
    visit(value)

  def visitReference(name: String): Unit = ()

  def visitTuple(elements: List[IRNode]): Unit =
    elements.foreach(visit)

  def visitUnitValue(value: Double, unit: String): Unit = ()

  def visitBoolean(value: Boolean): Unit = ()

  def visitParamBlock(params: List[(Option[String], IRNode)]): Unit =
    params.foreach { case (_, node) => visit(node) }

  def visitAttributes(target: IRNode, attrs: List[String]): Unit =
    visit(target)

/** Fold over IR nodes */
def foldIR[T](node: IRNode)(z: T)(f: (T, IRNode) => T): T =
  val acc = f(z, node)
  node match
    case IRSequence(nodes) => nodes.foldLeft(acc)((a, n) => foldIR(n)(a)(f))
    case IRConditional(cond, then_, else_) =>
      val a1 = foldIR(cond)(acc)(f)
      val a2 = foldIR(then_)(a1)(f)
      else_.fold(a2)(e => foldIR(e)(a2)(f))
    case IRBinding(_, value) => foldIR(value)(acc)(f)
    case IRTuple(elements) => elements.foldLeft(acc)((a, n) => foldIR(n)(a)(f))
    case IRParamBlock(params) => params.foldLeft(acc)((a, p) => foldIR(p._2)(a)(f))
    case IRAttributes(target, _) => foldIR(target)(acc)(f)
    case _ => acc

/** Collect all nodes of a certain type */
def collectNodes[T](node: IRNode)(pf: PartialFunction[IRNode, T]): List[T] =
  val results = mutable.ListBuffer[T]()
  foldIR(node)(()) { (_, n) =>
    if pf.isDefinedAt(n) then results += pf(n)
    ()
  }
  results.toList

/** Get all objects from an expression */
def extractObjects(expr: ProfessExpr): List[(String, String)] =
  collectNodes(expr.toIR) { case IRObject(k, n) => (k, n) }

/** Get all words from an expression */
def extractWords(expr: ProfessExpr): List[String] =
  collectNodes(expr.toIR) { case IRWord(w) => w }

/** Get all numbers from an expression */
def extractNumbers(expr: ProfessExpr): List[Double] =
  collectNodes(expr.toIR) { case IRNumber(n) => n }

// ═══════════════════════════════════════════════════════════════════════════
// PRETTY PRINTER
// ═══════════════════════════════════════════════════════════════════════════

object IRPrettyPrinter extends ProfessInterpreter[String]:
  def interpretObject(kind: String, name: String): String = s"[$kind: $name]"
  def interpretWord(word: String): String = word
  def interpretNumber(value: Double): String =
    if value == value.toLong then value.toLong.toString else value.toString
  def interpretString(value: String): String = s"\"$value\""
  def interpretSequence(nodes: List[IRNode]): String =
    nodes.map(interpret).mkString(" ")
  def interpretConditional(cond: IRNode, then_ : IRNode, else_ : Option[IRNode]): String =
    val elseStr = else_.map(e => s" ELSE ${interpret(e)}").getOrElse("")
    s"IF ${interpret(cond)} THEN ${interpret(then_)}$elseStr"
  def interpretBinding(name: String, value: IRNode): String =
    s"LET $name = ${interpret(value)}"
  def interpretReference(name: String): String = s"@$name"
  def interpretTuple(elements: List[IRNode]): String =
    s"(${elements.map(interpret).mkString(", ")})"
  def interpretUnitValue(value: Double, unit: String): String =
    val v = if value == value.toLong then value.toLong.toString else value.toString
    s"$v:$unit"
  def interpretBoolean(value: Boolean): String = value.toString
  def interpretParamBlock(params: List[(Option[String], IRNode)]): String =
    params.map {
      case (Some(name), value) => s"$name ${interpret(value)}"
      case (None, value) => interpret(value)
    }.mkString(", ")
  def interpretAttributes(target: IRNode, attrs: List[String]): String =
    s"${interpret(target)}: ${attrs.mkString(", ")}"
