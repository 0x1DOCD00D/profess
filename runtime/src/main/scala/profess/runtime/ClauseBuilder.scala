/*
 * Copyright (c) 2025 Mark Grechanik and Lone Star Consulting, Inc. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.
 */

package profess.runtime

import cats.{Applicative, Monoid}
import cats.syntax.all.*

/**
 * CLAUSE BUILDER — Structuring pass between FessIrLowering output and handler dispatch.
 *
 * Converts a flat IR token stream into a roled Clause using only token TYPE to find
 * phrase boundaries — no vocabulary, no domain knowledge:
 *
 *   Clause   ::= Subject Head Modifier*
 *   Subject  ::= Filler*          -- fillers before the first IRWord
 *   Head     ::= Word Filler*     -- first IRWord + its direct filler args
 *   Modifier ::= Word Filler*     -- each subsequent IRWord + its trailing fillers
 *   Filler   ::= IRObject | IRNumber | IRUnitValue | IRString
 *
 * The framework owns this pass. All domain meaning lives in the handlers.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Structured clause types — professor's exact types
// ─────────────────────────────────────────────────────────────────────────────

/** A modifier phrase: one relation word and the fillers that follow it. */
case class Phrase(relation: String, fillers: List[IRNode]):
  def render: String =
    val fs = if fillers.isEmpty then "" else " " + fillers.map(_.render).mkString(" ")
    s"$relation$fs"

/**
 * A structured view of one sentence after the builder pass.
 *
 * Example: "(broker mark) sold 700 (stock MSFT) at 150:dollars"
 *   subject   = [IRObject(broker, mark)]
 *   head      = "sold"
 *   args      = [IRNumber(700), IRObject(stock, MSFT)]
 *   modifiers = [Phrase("at", [IRUnitValue(150, dollars)])]
 */
case class Clause(
  subject: List[IRNode],
  head: String,
  args: List[IRNode],
  modifiers: List[Phrase]
):
  /** True when the sentence contained no IRWord (degenerate — subject only). */
  def isEmpty: Boolean = head.isEmpty

  def render: String =
    val subj    = subject.map(_.render).mkString(" ")
    val headPart = (head :: args.map(_.render)).mkString(" ")
    val mods    = modifiers.map(_.render).mkString(" ")
    List(subj, headPart, mods).filter(_.nonEmpty).mkString(" ")

// ─────────────────────────────────────────────────────────────────────────────
// TransformContext — carries the original flat sentence into every handler
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The original flat node list. Handlers that need to inspect sentence shape
 * beyond their structured clause view can read `sentence` directly.
 */
final case class TransformContext(sentence: List[IRNode])

object TransformContext:
  def of(ir: IRNode): TransformContext = ir match
    case IRSequence(ns) => TransformContext(ns)
    case single         => TransformContext(List(single))

// ─────────────────────────────────────────────────────────────────────────────
// RelationHandler — professor's exact trait
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A handler for the primary relation of a Clause (its head word).
 *
 * `interp` is a closure back into the registry. The handler calls it on
 * whichever fillers it wants to recurse into — enabling both flat independent
 * dispatch AND bottom-up compositional interpretation for free. The handler
 * decides what to recurse into; the framework does not force an evaluation order.
 *
 * Registration: registry.onRelation("sold") { (clause, interp, ctx) => ... }
 */
trait RelationHandler[F[_], A]:
  def relation: String
  def handle(clause: Clause, interp: IRNode => F[A], ctx: TransformContext): F[A]

// ─────────────────────────────────────────────────────────────────────────────
// ClauseBuilder — flat IR → Clause
// ─────────────────────────────────────────────────────────────────────────────

object ClauseBuilder:

  /**
   * Build a Clause from a flat IR node by splitting at IRWord boundaries.
   * Pure structural pass — O(n) in the number of nodes, no domain knowledge.
   */
  def build(ir: IRNode): Clause =
    val nodes = ir match
      case IRSequence(ns) => ns
      case single         => List(single)

    val firstWord = nodes.indexWhere(_.isInstanceOf[IRWord])
    if firstWord < 0 then
      return Clause(nodes, "", Nil, Nil)

    val subject = nodes.take(firstWord)
    val groups  = splitOnWords(nodes.drop(firstWord))

    groups match
      case Nil =>
        Clause(subject, "", Nil, Nil)
      case (headWord, headArgs) :: modGroups =>
        val modifiers = modGroups.map { case (w, fs) => Phrase(w, fs) }
        Clause(subject, headWord, headArgs, modifiers)

  /** Convenience: build directly from a ProfessExpr. */
  def build(expr: ProfessExpr): Clause = build(expr.toIR)

  /** Split a node list (beginning with an IRWord) into (word, fillers) groups. */
  private def splitOnWords(nodes: List[IRNode]): List[(String, List[IRNode])] =
    nodes match
      case Nil => Nil
      case IRWord(w) :: rest =>
        val (fillers, remaining) = rest.span(!_.isInstanceOf[IRWord])
        (w, fillers) :: splitOnWords(remaining)
      case _ :: rest => splitOnWords(rest)

// ─────────────────────────────────────────────────────────────────────────────
// ClauseRegistry — immutable registry of relation + object handlers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Holds relation handlers (keyed by head word) and object handlers (keyed by
 * kind, called via the `interp` closure from within relation handlers).
 *
 * Missing relation → `Monoid.empty`, recorded as a miss (the head word).
 * Missing object kind → `Monoid.empty` silently via `interp`.
 */
final class ClauseRegistry[F[_], A] private (
  private val relations: Map[String, RelationHandler[F, A]],
  private val kinds: Map[String, IRNode => F[A]]
)(using Applicative[F], Monoid[A]):

  /** Register a relation handler with a lambda. */
  def onRelation(word: String)(f: (Clause, IRNode => F[A], TransformContext) => F[A]): ClauseRegistry[F, A] =
    val h = new RelationHandler[F, A]:
      val relation                                                                = word
      def handle(clause: Clause, interp: IRNode => F[A], ctx: TransformContext): F[A] = f(clause, interp, ctx)
    new ClauseRegistry(relations.updated(word, h), kinds)

  /** Register an object handler — called via `interp` when a relation handler recurses. */
  def onKind(kind: String)(f: IRNode => F[A]): ClauseRegistry[F, A] =
    new ClauseRegistry(relations, kinds.updated(kind, f))

  /**
   * Dispatch a clause.
   * Returns `(result, None)` on a hit, `(pure(empty), Some(headWord))` on a miss.
   */
  def run(clause: Clause, ctx: TransformContext): (F[A], Option[String]) =
    val interp: IRNode => F[A] =
      case obj @ IRObject(kind, _) => kinds.get(kind).map(_(obj)).getOrElse(Monoid[A].empty.pure[F])
      case _                       => Monoid[A].empty.pure[F]
    relations.get(clause.head) match
      case Some(h) => (h.handle(clause, interp, ctx), None)
      case None    => (Monoid[A].empty.pure[F], Some(clause.head))

  /** Convenience: build the clause from an IR node, then dispatch it. */
  def run(ir: IRNode): (F[A], Option[String]) =
    run(ClauseBuilder.build(ir), TransformContext.of(ir))

  /** Relation words that currently have a registered handler. */
  def registeredRelations: Set[String] = relations.keySet

object ClauseRegistry:
  def empty[F[_] : Applicative, A : Monoid]: ClauseRegistry[F, A] =
    new ClauseRegistry(Map.empty, Map.empty)
