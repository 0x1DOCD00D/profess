/*
 * Copyright (c) 2025 Mark Grechanik and Lone Star Consulting, Inc. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.
 */

package examples

import cats.*
import cats.data.*
import cats.effect.*
import cats.syntax.all.*

import profess.runtime.*
import profess.runtime.effects.*

/**
 * π-CALCULUS TO AKKA TRANSLATION
 *
 * Demonstrates how PROFESS formal semantics enable:
 *   - Process calculus DSL definition
 *   - Algebraic optimizations (from monoid laws)
 *   - Translation to Akka actor code
 *   - Semantic preservation guarantees
 */

// ═══════════════════════════════════════════════════════════════════════════
// π-CALCULUS IR
// ═══════════════════════════════════════════════════════════════════════════

sealed trait PiIR
case object PiNil extends PiIR
case class PiInput(channel: String, binding: String, cont: PiIR) extends PiIR
case class PiOutput(channel: String, value: String, cont: PiIR) extends PiIR
case class PiParallel(left: PiIR, right: PiIR) extends PiIR
case class PiRestriction(channel: String, body: PiIR) extends PiIR
case class PiReplication(body: PiIR) extends PiIR
case class PiChoice(left: PiIR, right: PiIR) extends PiIR

// ═══════════════════════════════════════════════════════════════════════════
// AKKA TARGET REPRESENTATION
// ═══════════════════════════════════════════════════════════════════════════

sealed trait AkkaCode

object AkkaCode:
  case class ActorDef(name: String, behavior: AkkaBehavior) extends AkkaCode
  case class Spawn(name: String, behaviorExpr: String) extends AkkaCode
  case class Tell(target: String, message: String) extends AkkaCode
  case class Seq(steps: List[AkkaCode]) extends AkkaCode
  case object Stopped extends AkkaCode

  given Monoid[AkkaCode] with
    def empty: AkkaCode = Seq(Nil)
    def combine(a: AkkaCode, b: AkkaCode): AkkaCode = (a, b) match
      case (Seq(Nil), x) => x
      case (x, Seq(Nil)) => x
      case (Seq(xs), Seq(ys)) => Seq(xs ++ ys)
      case (Seq(xs), y) => Seq(xs :+ y)
      case (x, Seq(ys)) => Seq(x :: ys)
      case (x, y) => Seq(List(x, y))

sealed trait AkkaBehavior
case class ReceiveMsg(handlers: List[(String, AkkaCode)]) extends AkkaBehavior
case class SetupBehavior(init: AkkaCode, next: AkkaBehavior) extends AkkaBehavior
case object EmptyBehavior extends AkkaBehavior

// ═══════════════════════════════════════════════════════════════════════════
// OPTIMIZATION PASSES (Justified by Formal Theory)
// ═══════════════════════════════════════════════════════════════════════════

object PiOptimizations:

  /**
   * Dead Process Elimination
   *
   * Algebraic justification: P | 0 ≡ P (monoid identity)
   */
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

  /**
   * Parallel Normalization (Right-Associate)
   *
   * Algebraic justification: (P | Q) | R ≡ P | (Q | R) (associativity)
   * Enables consistent traversal order
   */
  def normalizeParallel(ir: PiIR): PiIR = ir match
    case PiParallel(PiParallel(a, b), c) =>
      normalizeParallel(PiParallel(a, PiParallel(b, c)))
    case PiParallel(a, b) =>
      PiParallel(normalizeParallel(a), normalizeParallel(b))
    case PiInput(ch, b, cont) => PiInput(ch, b, normalizeParallel(cont))
    case PiOutput(ch, v, cont) => PiOutput(ch, v, normalizeParallel(cont))
    case PiRestriction(ch, body) => PiRestriction(ch, normalizeParallel(body))
    case PiReplication(body) => PiReplication(normalizeParallel(body))
    case other => other

  /**
   * Channel Scope Narrowing
   *
   * Algebraic justification: (νx)(P | Q) ≡ P | (νx)Q when x ∉ fv(P)
   * Reduces actor lifetime, enables GC
   */
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
        val fpQ = freeChannels(q)
        if !fpP.contains(ch) then
          PiParallel(narrowScope(p), PiRestriction(ch, narrowScope(q)))
        else if !fpQ.contains(ch) then
          PiParallel(PiRestriction(ch, narrowScope(p)), narrowScope(q))
        else
          PiRestriction(ch, PiParallel(narrowScope(p), narrowScope(q)))
      case PiParallel(l, r) => PiParallel(narrowScope(l), narrowScope(r))
      case PiRestriction(ch, body) => PiRestriction(ch, narrowScope(body))
      case other => other

  /**
   * Synchronous Communication Fusion
   *
   * When send and receive on same channel are in same restriction:
   * (νx)(x̄⟨v⟩.P | x(y).Q) → P | Q[v/y]
   *
   * Eliminates actor creation for internal communication
   */
  def fuseSynchronous(ir: PiIR): PiIR =
    def substitute(p: PiIR, variable: String, value: String): PiIR = p match
      case PiOutput(ch, v, cont) =>
        val newV = if v == variable then value else v
        PiOutput(ch, newV, substitute(cont, variable, value))
      case PiInput(ch, b, cont) if b != variable =>
        PiInput(ch, b, substitute(cont, variable, value))
      case PiParallel(l, r) =>
        PiParallel(substitute(l, variable, value), substitute(r, variable, value))
      case other => other

    ir match
      case PiRestriction(ch, PiParallel(PiOutput(ch1, v, p), PiInput(ch2, y, q)))
        if ch == ch1 && ch == ch2 =>
        // Fuse! No actors needed for this communication
        PiParallel(fuseSynchronous(p), fuseSynchronous(substitute(q, y, v)))

      case PiRestriction(ch, PiParallel(PiInput(ch1, y, q), PiOutput(ch2, v, p)))
        if ch == ch1 && ch == ch2 =>
        // Symmetric case
        PiParallel(fuseSynchronous(p), fuseSynchronous(substitute(q, y, v)))

      case PiParallel(l, r) => PiParallel(fuseSynchronous(l), fuseSynchronous(r))
      case PiRestriction(ch, body) => PiRestriction(ch, fuseSynchronous(body))
      case other => other

  /** Full optimization pipeline */
  def optimize(ir: PiIR): PiIR =
    val pass1 = eliminateNil(ir)
    val pass2 = normalizeParallel(pass1)
    val pass3 = narrowScope(pass2)
    val pass4 = fuseSynchronous(pass3)
    pass4

// ═══════════════════════════════════════════════════════════════════════════
// TRANSLATION CONTEXT
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// TRANSLATION (IR → Akka Code)
// ═══════════════════════════════════════════════════════════════════════════

object PiTranslator:
  import AkkaCode._

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

    case PiRestriction(channel, body) =>
      for
        ctx <- StateT.get[IO, TransContext]
        (actorName, ctx1) = ctx.fresh(s"${channel}_chan")
        ctx2 = ctx1.bindChannel(channel, actorName)
        _ <- StateT.set[IO, TransContext](ctx2)
        bodyCode <- translate(body)
      yield Seq(List(
        Spawn(actorName, "Behaviors.receiveMessage { msg => ??? }"),
        bodyCode
      ))

    case PiReplication(body) =>
      for
        bodyCode <- translate(body)
      yield ActorDef(
        "replicator",
        SetupBehavior(
          Spawn("instance", codeToString(bodyCode)),
          ReceiveMsg(List("trigger" -> Seq(List(
            Spawn("instance", codeToString(bodyCode)),
            Tell("self", "trigger")
          ))))
        )
      )

    case PiChoice(left, right) =>
      for
        leftCode <- translate(left)
        rightCode <- translate(right)
      yield ActorDef(
        "chooser",
        ReceiveMsg(List(
          "left" -> leftCode,
          "right" -> rightCode
        ))
      )

  def codeToString(code: AkkaCode): String = code match
    case Stopped => "Behaviors.stopped"
    case Tell(t, m) => s"""$t ! "$m""""
    case Spawn(n, b) => s"""context.spawn($b, "$n")"""
    case Seq(steps) => steps.map(codeToString).mkString("; ")
    case ActorDef(n, b) => s"// Actor $n: ${behaviorToString(b)}"

  def behaviorToString(b: AkkaBehavior): String = b match
    case ReceiveMsg(hs) => s"Behaviors.receive { ${hs.map(_._1).mkString(", ")} => ... }"
    case SetupBehavior(i, n) => s"Behaviors.setup { ... }"
    case EmptyBehavior => "Behaviors.empty"

// ═══════════════════════════════════════════════════════════════════════════
// CODE GENERATOR
// ═══════════════════════════════════════════════════════════════════════════

object AkkaCodeGen:
  import AkkaCode._

  def generate(code: AkkaCode, indent: Int = 0): String =
    val pad = "  " * indent
    code match
      case Stopped =>
        s"${pad}Behaviors.stopped"

      case Tell(target, message) =>
        s"""${pad}$target ! "$message""""

      case Spawn(name, behavior) =>
        s"""${pad}val $name = context.spawn($behavior, "$name")"""

      case Seq(Nil) =>
        s"${pad}// no-op"

      case Seq(steps) =>
        steps.map(s => generate(s, indent)).mkString("\n")

      case ActorDef(name, behavior) =>
        s"""${pad}def ${name}Behavior: Behavior[String] =
${generateBehavior(behavior, indent + 1)}"""

  def generateBehavior(b: AkkaBehavior, indent: Int): String =
    val pad = "  " * indent
    b match
      case ReceiveMsg(handlers) =>
        val cases = handlers.map { case (msg, code) =>
          s"""${pad}  case "$msg" =>
${generate(code, indent + 2)}
${pad}    Behaviors.same"""
        }.mkString("\n")
        s"""${pad}Behaviors.receiveMessage[String] {
$cases
${pad}}"""

      case SetupBehavior(init, next) =>
        s"""${pad}Behaviors.setup[String] { context =>
${generate(init, indent + 1)}
${generateBehavior(next, indent + 1)}
${pad}}"""

      case EmptyBehavior =>
        s"${pad}Behaviors.empty"

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE USAGE
// ═══════════════════════════════════════════════════════════════════════════

object PiCalculusExample extends IOApp.Simple:

  def run: IO[Unit] =
    for
      _ <- IO.println("═" * 70)
      _ <- IO.println(" π-CALCULUS TO AKKA TRANSLATION")
      _ <- IO.println("═" * 70)
      _ <- IO.println("")

      // Example: Ping-Pong
      // (νping)(νpong)(
      //   ping(x).ponḡ⟨x⟩.0 
      //   | pong(y).pinḡ⟨y⟩.0 
      //   | pinḡ⟨"hello"⟩.0
      // )

      pingPong = PiRestriction("ping",
        PiRestriction("pong",
          PiParallel(
            PiInput("ping", "x", PiOutput("pong", "x", PiNil)),
            PiParallel(
              PiInput("pong", "y", PiOutput("ping", "y", PiNil)),
              PiOutput("ping", "hello", PiNil)
            )
          )
        )
      )

      _ <- IO.println("▶ ORIGINAL π-CALCULUS:")
      _ <- IO.println(s"  $pingPong")
      _ <- IO.println("")

      // Apply optimizations
      optimized = PiOptimizations.optimize(pingPong)

      _ <- IO.println("▶ AFTER OPTIMIZATION:")
      _ <- IO.println(s"  $optimized")
      _ <- IO.println("")

      // Translate to Akka
      (ctx, akkaCode) <- PiTranslator.translate(optimized).run(TransContext())

      _ <- IO.println("▶ GENERATED AKKA CODE:")
      _ <- IO.println("─" * 70)
      _ <- IO.println(AkkaCodeGen.generate(akkaCode))
      _ <- IO.println("─" * 70)
      _ <- IO.println("")

      // Show optimization benefits
      _ <- IO.println("▶ OPTIMIZATION BENEFITS:")
      _ <- IO.println("  • Dead process elimination: P | 0 → P")
      _ <- IO.println("  • Parallel normalization: (P|Q)|R → P|(Q|R)")
      _ <- IO.println("  • Scope narrowing: (νx)(P|Q) → P|(νx)Q when x∉fv(P)")
      _ <- IO.println("  • Sync fusion: (νx)(x̄⟨v⟩.P | x(y).Q) → P | Q[v/y]")
      _ <- IO.println("")

      _ <- IO.println("═" * 70)
    yield ()
