/*
 * Copyright (c) 2025 Mark Grechanik and Lone Star Consulting, Inc. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.
 */

package profess.runtime.effects

import cats.*
import cats.data.*
import cats.effect.*
import cats.effect.syntax.all.*
import cats.syntax.all.*
import profess.runtime.*

/**
 * PROFESS CATS INTEGRATION
 *
 * Programming Rule Oriented Formalized English Sentence Specifications
 *
 * This module provides:
 *   - Typeclasses for IR node transformation
 *   - Word-specific handlers injected via given instances
 *   - Cats Effect integration for effectful execution
 *   - Traversal system that applies programmer-defined transformations
 */

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFORMATION CONTEXT
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Context passed through transformations.
 * Carries state and configuration for the transformation pipeline.
 */
case class TransformContext(
                             bindings: Map[String, Any] = Map.empty,
                             metadata: Map[String, String] = Map.empty,
                             depth: Int = 0
                           ):
  def bind(name: String, value: Any): TransformContext =
    copy(bindings = bindings + (name -> value))

  def withMeta(key: String, value: String): TransformContext =
    copy(metadata = metadata + (key -> value))

  def descend: TransformContext = copy(depth = depth + 1)

object TransformContext:
  val empty: TransformContext = TransformContext()

// ═══════════════════════════════════════════════════════════════════════════
// NODE TRANSFORMER TYPECLASS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Typeclass for transforming IR nodes into executable code.
 *
 * F[_] - Effect type (IO, etc.)
 * A    - Result type of transformation
 *
 * Programmers provide instances for their domain.
 */
trait NodeTransformer[F[_], A]:
  def transform(node: IRNode, ctx: TransformContext): F[A]

object NodeTransformer:
  def apply[F[_], A](using nt: NodeTransformer[F, A]): NodeTransformer[F, A] = nt

  /** Create a NodeTransformer from a function */
  def instance[F[_], A](f: (IRNode, TransformContext) => F[A]): NodeTransformer[F, A] =
    new NodeTransformer[F, A]:
      def transform(node: IRNode, ctx: TransformContext): F[A] = f(node, ctx)

// ═══════════════════════════════════════════════════════════════════════════
// WORD HANDLER TYPECLASS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Handler for a specific word in PROFESS expressions.
 *
 * Programmers define handlers for words like "sold", "bought", "transferred", etc.
 * The handler receives the word, following arguments, and context.
 */
trait WordHandler[F[_], A]:
  /** The word this handler matches */
  def word: String

  /** Transform the word and its arguments */
  def handle(args: List[IRNode], ctx: TransformContext): F[A]

  /** Check if this handler matches a word */
  def matches(w: String): Boolean = w == word

object WordHandler:
  /** Create a WordHandler from word and function */
  def apply[F[_], A](w: String)(f: (List[IRNode], TransformContext) => F[A]): WordHandler[F, A] =
    new WordHandler[F, A]:
      val word: String = w
      def handle(args: List[IRNode], ctx: TransformContext): F[A] = f(args, ctx)

// ═══════════════════════════════════════════════════════════════════════════
// OBJECT HANDLER TYPECLASS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Handler for PROFESS objects of a specific kind.
 *
 * Programmers define handlers for kinds like "broker", "stock", "account", etc.
 */
trait ObjectHandler[F[_], A]:
  /** The kind this handler matches */
  def kind: String

  /** Transform the object */
  def handle(name: String, ctx: TransformContext): F[A]

  /** Check if this handler matches a kind */
  def matches(k: String): Boolean = k == kind

object ObjectHandler:
  /** Create an ObjectHandler from kind and function */
  def apply[F[_], A](k: String)(f: (String, TransformContext) => F[A]): ObjectHandler[F, A] =
    new ObjectHandler[F, A]:
      val kind: String = k
      def handle(name: String, ctx: TransformContext): F[A] = f(name, ctx)

// ═══════════════════════════════════════════════════════════════════════════
// HANDLER REGISTRY
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Registry of handlers for transforming IR nodes.
 *
 * Collects word handlers and object handlers, then uses them
 * during IR traversal.
 */
class HandlerRegistry[F[_]: Monad, A](
                                       val wordHandlers: Map[String, WordHandler[F, A]],
                                       val objectHandlers: Map[String, ObjectHandler[F, A]],
                                       val defaultWordHandler: Option[(String, List[IRNode], TransformContext) => F[A]] = None,
                                       val defaultObjectHandler: Option[(String, String, TransformContext) => F[A]] = None
                                     ):

  def findWordHandler(word: String): Option[WordHandler[F, A]] =
    wordHandlers.get(word)

  def findObjectHandler(kind: String): Option[ObjectHandler[F, A]] =
    objectHandlers.get(kind)

  def withWordHandler(handler: WordHandler[F, A]): HandlerRegistry[F, A] =
    new HandlerRegistry(
      wordHandlers + (handler.word -> handler),
      objectHandlers,
      defaultWordHandler,
      defaultObjectHandler
    )

  def withObjectHandler(handler: ObjectHandler[F, A]): HandlerRegistry[F, A] =
    new HandlerRegistry(
      wordHandlers,
      objectHandlers + (handler.kind -> handler),
      defaultWordHandler,
      defaultObjectHandler
    )

  def withDefaultWordHandler(h: (String, List[IRNode], TransformContext) => F[A]): HandlerRegistry[F, A] =
    new HandlerRegistry(wordHandlers, objectHandlers, Some(h), defaultObjectHandler)

  def withDefaultObjectHandler(h: (String, String, TransformContext) => F[A]): HandlerRegistry[F, A] =
    new HandlerRegistry(wordHandlers, objectHandlers, defaultWordHandler, Some(h))

object HandlerRegistry:
  def empty[F[_]: Monad, A]: HandlerRegistry[F, A] =
    new HandlerRegistry(Map.empty, Map.empty, None, None)

  /** Build a registry from a list of handlers */
  def apply[F[_]: Monad, A](
                             wordHandlers: List[WordHandler[F, A]] = Nil,
                             objectHandlers: List[ObjectHandler[F, A]] = Nil
                           ): HandlerRegistry[F, A] =
    new HandlerRegistry(
      wordHandlers.map(h => h.word -> h).toMap,
      objectHandlers.map(h => h.kind -> h).toMap,
      None,
      None
    )

// ═══════════════════════════════════════════════════════════════════════════
// IR TRAVERSER WITH CATS EFFECT
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Traverses IR nodes and applies handlers to produce effects.
 *
 * The traverser:
 *   1. Walks the IR structure
 *   2. Matches nodes against registered handlers
 *   3. Applies handlers to produce F[A] effects
 *   4. Combines results using Monad/Applicative operations
 */
class IRTraverser[F[_]: Monad, A: Monoid](registry: HandlerRegistry[F, A]):

  import Monoid.combineAll

  /** Traverse a PROFESS expression */
  def traverse(expr: ProfessExpr, ctx: TransformContext = TransformContext.empty): F[A] =
    traverseNode(expr.toIR, ctx)

  /** Traverse an IR node */
  def traverseNode(node: IRNode, ctx: TransformContext): F[A] =
    node match
      case IRObject(kind, name) =>
        registry.findObjectHandler(kind) match
          case Some(handler) => handler.handle(name, ctx)
          case None => registry.defaultObjectHandler match
            case Some(h) => h(kind, name, ctx)
            case None => Monoid[A].empty.pure[F]

      case IRWord(word) =>
        registry.findWordHandler(word) match
          case Some(handler) => handler.handle(Nil, ctx)
          case None => registry.defaultWordHandler match
            case Some(h) => h(word, Nil, ctx)
            case None => Monoid[A].empty.pure[F]

      case IRNumber(value) =>
        Monoid[A].empty.pure[F]

      case IRString(value) =>
        Monoid[A].empty.pure[F]

      case IRSequence(nodes) =>
        traverseSequence(nodes, ctx)

      case IRConditional(cond, then_, else_) =>
        for
          c <- traverseNode(cond, ctx)
          t <- traverseNode(then_, ctx)
          e <- else_.traverse(n => traverseNode(n, ctx))
        yield combineAll(List(c, t) ++ e.toList)

      case IRTuple(elements) =>
        elements.traverse(n => traverseNode(n, ctx.descend)).map(combineAll)

      case IRUnitValue(value, unit) =>
        Monoid[A].empty.pure[F]

      case IRBoolean(value) =>
        Monoid[A].empty.pure[F]

      case IRParamBlock(params) =>
        params.traverse { case (_, node) => traverseNode(node, ctx.descend) }.map(combineAll)

      case IRAttributes(target, attrs) =>
        traverseNode(target, ctx)

      case IRBinding(name, value) =>
        traverseNode(value, ctx.bind(name, value))

      case IRReference(name) =>
        Monoid[A].empty.pure[F]

  /** Traverse a sequence, detecting word+args patterns */
  def traverseSequence(nodes: List[IRNode], ctx: TransformContext): F[A] =
    nodes match
      case Nil => Monoid[A].empty.pure[F]

      case IRWord(word) :: rest =>
        registry.findWordHandler(word) match
          case Some(handler) =>
            // Collect arguments until next word
            val (args, remaining) = rest.span(!_.isInstanceOf[IRWord])
            for
              result <- handler.handle(args, ctx)
              restResult <- traverseSequence(remaining, ctx)
            yield Monoid[A].combine(result, restResult)

          case None =>
            registry.defaultWordHandler match
              case Some(h) =>
                val (args, remaining) = rest.span(!_.isInstanceOf[IRWord])
                for
                  result <- h(word, args, ctx)
                  restResult <- traverseSequence(remaining, ctx)
                yield Monoid[A].combine(result, restResult)
              case None =>
                traverseSequence(rest, ctx)

      case head :: rest =>
        for
          headResult <- traverseNode(head, ctx)
          restResult <- traverseSequence(rest, ctx)
        yield Monoid[A].combine(headResult, restResult)

object IRTraverser:
  def apply[F[_]: Monad, A: Monoid](registry: HandlerRegistry[F, A]): IRTraverser[F, A] =
    new IRTraverser(registry)

// ═══════════════════════════════════════════════════════════════════════════
// EXECUTABLE CODE REPRESENTATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Represents executable code produced by transformation.
 *
 * This is a simple representation - programmers can define their own.
 */
sealed trait Executable[+A]

object Executable:
  case object NoOp extends Executable[Nothing]
  case class Pure[A](value: A) extends Executable[A]
  case class Effect[A](run: IO[A]) extends Executable[A]
  case class Sequence[A](steps: List[Executable[A]]) extends Executable[A]
  case class Conditional[A](cond: Executable[Boolean], then_ : Executable[A], else_ : Executable[A]) extends Executable[A]

  given executableMonoid[A]: Monoid[Executable[A]] with
    def empty: Executable[A] = NoOp
    def combine(x: Executable[A], y: Executable[A]): Executable[A] =
      (x, y) match
        case (NoOp, b) => b
        case (a, NoOp) => a
        case (Sequence(xs), Sequence(ys)) => Sequence(xs ++ ys)
        case (Sequence(xs), b) => Sequence(xs :+ b)
        case (a, Sequence(ys)) => Sequence(a :: ys)
        case (a, b) => Sequence(List(a, b))

  /** Execute an Executable, producing an IO */
  def execute[A](exec: Executable[A])(using default: A): IO[A] =
    exec match
      case NoOp => IO.pure(default)
      case Pure(value) => IO.pure(value)
      case Effect(run) => run
      case Sequence(steps) =>
        steps.foldLeftM(default)((_, step) => execute(step)(using default))
      case Conditional(cond, then_, else_) =>
        for
          c <- execute(cond)(using false)
          result <- if c then execute(then_)(using default) else execute(else_)(using default)
        yield result

// ═══════════════════════════════════════════════════════════════════════════
// EFFECT-PRODUCING INTERPRETER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Full interpreter that transforms IR and executes effects.
 *
 * Usage:
 *   given registry: HandlerRegistry[IO, Executable[Unit]] = ...
 *   val interpreter = EffectInterpreter[Unit]
 *   interpreter.run(expression)
 */
class EffectInterpreter[A](using default: A):

  /** Run transformation and execute the result */
  def run[F[_]: Monad](
                        expr: ProfessExpr,
                        registry: HandlerRegistry[F, Executable[A]],
                        ctx: TransformContext = TransformContext.empty
                      )(using ev: F[Executable[A]] =:= IO[Executable[A]]): IO[A] =
    val traverser = IRTraverser[F, Executable[A]](registry)
    for
      executable <- ev(traverser.traverse(expr, ctx))
      result <- Executable.execute(executable)
    yield result

  /** Run with IO directly */
  def runIO(
             expr: ProfessExpr,
             registry: HandlerRegistry[IO, Executable[A]],
             ctx: TransformContext = TransformContext.empty
           ): IO[A] =
    val traverser = IRTraverser[IO, Executable[A]](registry)
    for
      executable <- traverser.traverse(expr, ctx)
      result <- Executable.execute(executable)
    yield result

object EffectInterpreter:
  def apply[A](using default: A): EffectInterpreter[A] = new EffectInterpreter[A]

// ═══════════════════════════════════════════════════════════════════════════
// DSL FOR BUILDING HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * DSL for building handler registries in a fluent way.
 */
object HandlerDSL:

  /** Start building a registry */
  def handlers[F[_]: Monad, A]: RegistryBuilder[F, A] =
    RegistryBuilder(HandlerRegistry.empty[F, A])

  class RegistryBuilder[F[_]: Monad, A](registry: HandlerRegistry[F, A]):

    /** Add a word handler */
    def onWord(word: String)(f: (List[IRNode], TransformContext) => F[A]): RegistryBuilder[F, A] =
      RegistryBuilder(registry.withWordHandler(WordHandler(word)(f)))

    /** Add an object handler */
    def onObject(kind: String)(f: (String, TransformContext) => F[A]): RegistryBuilder[F, A] =
      RegistryBuilder(registry.withObjectHandler(ObjectHandler(kind)(f)))

    /** Set default word handler */
    def onAnyWord(f: (String, List[IRNode], TransformContext) => F[A]): RegistryBuilder[F, A] =
      RegistryBuilder(registry.withDefaultWordHandler(f))

    /** Set default object handler */
    def onAnyObject(f: (String, String, TransformContext) => F[A]): RegistryBuilder[F, A] =
      RegistryBuilder(registry.withDefaultObjectHandler(f))

    /** Build the registry */
    def build: HandlerRegistry[F, A] = registry

// ═══════════════════════════════════════════════════════════════════════════
// GIVEN-BASED HANDLER INJECTION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Marker trait for domain-specific handlers.
 *
 * Programmers extend this to create injectable handlers.
 */
trait DomainHandlers[F[_], A]:
  def registry: HandlerRegistry[F, A]

/**
 * Run expressions using given handlers.
 */
def runWithHandlers[F[_]: Monad, A: Monoid](
                                             expr: ProfessExpr,
                                             ctx: TransformContext = TransformContext.empty
                                           )(using handlers: DomainHandlers[F, A]): F[A] =
  IRTraverser[F, A](handlers.registry).traverse(expr, ctx)

/**
 * Run and execute with given handlers.
 */
def executeWithHandlers[A](
                            expr: ProfessExpr,
                            ctx: TransformContext = TransformContext.empty
                          )(using handlers: DomainHandlers[IO, Executable[A]], default: A): IO[A] =
  for
    executable <- runWithHandlers[IO, Executable[A]](expr, ctx)
    result <- Executable.execute(executable)
  yield result