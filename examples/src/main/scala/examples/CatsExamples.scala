/*
 * Copyright (c) 2025 Mark Grechanik and Lone Star Consulting, Inc. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.
 */

package examples

import cats.*
import cats.data.*
import cats.effect.*
import cats.effect.unsafe.implicits.global
import cats.syntax.all.*

import profess.runtime.*
import profess.runtime.effects.*

/**
 * PROFESS + CATS EFFECT EXAMPLES
 *
 * Demonstrates:
 *   - Defining word handlers for transformation
 *   - Defining object handlers
 *   - Using given instances for handler injection
 *   - Executing transformations with IO
 */

// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 1: Simple Trading Domain
// ═══════════════════════════════════════════════════════════════════════════

object TradingExample:

  // ─── Domain Model ───

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
                         ):
    def withBroker(b: String): TradingState = copy(currentBroker = Some(b))
    def withAction(a: String): TradingState = copy(currentAction = Some(a))
    def withQuantity(q: Int): TradingState = copy(currentQuantity = Some(q))
    def withInstrument(i: String): TradingState = copy(currentInstrument = Some(i))
    def withPrice(p: Double): TradingState = copy(currentPrice = Some(p))

    def completeTrade: TradingState =
      (currentBroker, currentAction, currentQuantity, currentInstrument) match
        case (Some(b), Some(a), Some(q), Some(i)) =>
          val trade = Trade(b, a, q, i, currentPrice)
          copy(
            trades = trades :+ trade,
            currentBroker = None,
            currentAction = None,
            currentQuantity = None,
            currentInstrument = None,
            currentPrice = None
          )
        case _ => this

  // ─── Handlers using StateT ───

  type TradingF[A] = StateT[IO, TradingState, A]

  val brokerHandler = ObjectHandler[TradingF, Unit]("broker") { (name, ctx) =>
    StateT.modify[IO, TradingState](_.withBroker(name))
  }

  val stockHandler = ObjectHandler[TradingF, Unit]("stock") { (name, ctx) =>
    StateT.modify[IO, TradingState](_.withInstrument(name))
  }

  val soldHandler = WordHandler[TradingF, Unit]("sold") { (args, ctx) =>
    StateT.modify[IO, TradingState] { state =>
      val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
      state.withAction("sold").withQuantity(qty)
    }
  }

  val boughtHandler = WordHandler[TradingF, Unit]("bought") { (args, ctx) =>
    StateT.modify[IO, TradingState] { state =>
      val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
      state.withAction("bought").withQuantity(qty)
    }
  }

  val atHandler = WordHandler[TradingF, Unit]("at") { (args, ctx) =>
    StateT.modify[IO, TradingState] { state =>
      val price = args.collectFirst { case IRNumber(n) => n }.getOrElse(0.0)
      state.withPrice(price).completeTrade
    }
  }

  // Default handler for unknown words (just completes trade if enough info)
  val defaultWord: (String, List[IRNode], TransformContext) => TradingF[Unit] =
    (word, args, ctx) => StateT.modify[IO, TradingState](_.completeTrade)

  // ─── Registry ───

  val registry: HandlerRegistry[TradingF, Unit] =
    HandlerRegistry[TradingF, Unit](
      wordHandlers = List(soldHandler, boughtHandler, atHandler),
      objectHandlers = List(brokerHandler, stockHandler)
    ).withDefaultWordHandler(defaultWord)

  // ─── Run ───

  def interpret(expr: ProfessExpr): IO[List[Trade]] =
    val traverser = IRTraverser[TradingF, Unit](registry)
    for
      (finalState, _) <- traverser.traverse(expr).run(TradingState())
    yield finalState.completeTrade.trades


// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 2: Effect-Producing Handlers (Logging)
// ═══════════════════════════════════════════════════════════════════════════

object LoggingExample:

  // Handlers that produce logging effects

  val registry: HandlerRegistry[IO, Executable[Unit]] =
    HandlerDSL.handlers[IO, Executable[Unit]]
      .onObject("broker") { (name, ctx) =>
        IO.pure(Executable.Effect(IO.println(s"[BROKER] $name entered the trade")))
      }
      .onObject("stock") { (name, ctx) =>
        IO.pure(Executable.Effect(IO.println(s"[STOCK] Instrument: $name")))
      }
      .onWord("sold") { (args, ctx) =>
        val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
        IO.pure(Executable.Effect(IO.println(s"[ACTION] Selling $qty units")))
      }
      .onWord("bought") { (args, ctx) =>
        val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
        IO.pure(Executable.Effect(IO.println(s"[ACTION] Buying $qty units")))
      }
      .onWord("at") { (args, ctx) =>
        val price = args.collectFirst { case IRNumber(n) => n }.getOrElse(0.0)
        IO.pure(Executable.Effect(IO.println(s"[PRICE] @ $$$price")))
      }
      .onAnyWord { (word, args, ctx) =>
        IO.pure(Executable.Effect(IO.println(s"[WORD] $word")))
      }
      .build

  def run(expr: ProfessExpr): IO[Unit] =
    given Unit = ()
    EffectInterpreter[Unit].runIO(expr, registry)


// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 3: Given-Based Handler Injection
// ═══════════════════════════════════════════════════════════════════════════

object GivenInjectionExample:

  // Define domain handlers as givens

  object TradingDomain:
    // Result type for transformations
    case class TradeAction(description: String)

    given Monoid[TradeAction] with
      def empty: TradeAction = TradeAction("")
      def combine(x: TradeAction, y: TradeAction): TradeAction =
        TradeAction(s"${x.description}; ${y.description}".stripPrefix("; ").stripSuffix("; "))

    // Domain handlers injected via given
    given DomainHandlers[IO, TradeAction] with
      val registry: HandlerRegistry[IO, TradeAction] =
        HandlerDSL.handlers[IO, TradeAction]
          .onObject("broker") { (name, ctx) =>
            IO.pure(TradeAction(s"Broker $name"))
          }
          .onObject("stock") { (name, ctx) =>
            IO.pure(TradeAction(s"trades $name"))
          }
          .onWord("sold") { (args, ctx) =>
            val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
            IO.pure(TradeAction(s"sells $qty"))
          }
          .onWord("bought") { (args, ctx) =>
            val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
            IO.pure(TradeAction(s"buys $qty"))
          }
          .build

  // Usage: handlers are automatically injected
  def interpret(expr: ProfessExpr): IO[TradingDomain.TradeAction] =
    import TradingDomain.given
    runWithHandlers[IO, TradingDomain.TradeAction](expr)


// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 4: Composable Domain Handlers
// ═══════════════════════════════════════════════════════════════════════════

object ComposableDomain:

  // Build handlers from multiple sources

  trait TradingWords[F[_]]:
    def soldHandler: WordHandler[F, Unit]
    def boughtHandler: WordHandler[F, Unit]

  trait TradingObjects[F[_]]:
    def brokerHandler: ObjectHandler[F, Unit]
    def stockHandler: ObjectHandler[F, Unit]

  // Implementation for IO
  object IOTradingWords extends TradingWords[IO]:
    def soldHandler = WordHandler[IO, Unit]("sold") { (args, ctx) =>
      IO.println(s"SOLD ${args.collect { case IRNumber(n) => n.toInt }.mkString}")
    }
    def boughtHandler = WordHandler[IO, Unit]("bought") { (args, ctx) =>
      IO.println(s"BOUGHT ${args.collect { case IRNumber(n) => n.toInt }.mkString}")
    }

  object IOTradingObjects extends TradingObjects[IO]:
    def brokerHandler = ObjectHandler[IO, Unit]("broker") { (name, ctx) =>
      IO.println(s"Broker: $name")
    }
    def stockHandler = ObjectHandler[IO, Unit]("stock") { (name, ctx) =>
      IO.println(s"Stock: $name")
    }

  // Compose into registry
  def makeRegistry(
                    words: TradingWords[IO],
                    objects: TradingObjects[IO]
                  ): HandlerRegistry[IO, Unit] =
    HandlerRegistry[IO, Unit](
      wordHandlers = List(words.soldHandler, words.boughtHandler),
      objectHandlers = List(objects.brokerHandler, objects.stockHandler)
    )


// ═══════════════════════════════════════════════════════════════════════════
// EXAMPLE 5: Code Generation
// ═══════════════════════════════════════════════════════════════════════════

object CodeGenExample:

  // Generate Scala code from PROFESS expressions

  case class CodeFragment(code: String)

  given Monoid[CodeFragment] with
    def empty: CodeFragment = CodeFragment("")
    def combine(x: CodeFragment, y: CodeFragment): CodeFragment =
      CodeFragment(s"${x.code}\n${y.code}".trim)

  val registry: HandlerRegistry[IO, CodeFragment] =
    HandlerDSL.handlers[IO, CodeFragment]
      .onObject("broker") { (name, ctx) =>
        IO.pure(CodeFragment(s"""val broker = Broker("$name")"""))
      }
      .onObject("stock") { (name, ctx) =>
        IO.pure(CodeFragment(s"""val stock = Stock("$name")"""))
      }
      .onWord("sold") { (args, ctx) =>
        val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
        IO.pure(CodeFragment(s"""broker.sell($qty, stock)"""))
      }
      .onWord("bought") { (args, ctx) =>
        val qty = args.collectFirst { case IRNumber(n) => n.toInt }.getOrElse(0)
        IO.pure(CodeFragment(s"""broker.buy($qty, stock)"""))
      }
      .onWord("at") { (args, ctx) =>
        val price = args.collectFirst { case IRNumber(n) => n }.getOrElse(0.0)
        IO.pure(CodeFragment(s"""  .atPrice($price)"""))
      }
      .build

  def generateCode(expr: ProfessExpr): IO[String] =
    val traverser = IRTraverser[IO, CodeFragment](registry)
    traverser.traverse(expr).map(_.code)


// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

object CatsExamplesMain extends IOApp.Simple:

  def run: IO[Unit] =
    for
      _ <- IO.println("═" * 70)
      _ <- IO.println(" PROFESS + CATS EFFECT EXAMPLES")
      _ <- IO.println("═" * 70)
      _ <- IO.println("")

      // Build test expressions (simulated - in real use, plugin scaffolds these)
      trade1 = ProfessExpr(List(
        IRObject("broker", "Mark"),
        IRWord("sold"),
        IRNumber(700),
        IRObject("stock", "MSFT"),
        IRWord("at"),
        IRNumber(150)
      ))

      trade2 = ProfessExpr(List(
        IRObject("broker", "Jane"),
        IRWord("bought"),
        IRNumber(500),
        IRObject("stock", "AAPL"),
        IRWord("at"),
        IRNumber(175)
      ))

      // Example 1: Trading with StateT
      _ <- IO.println("▶ EXAMPLE 1: Trading with StateT")
      _ <- IO.println("─" * 70)
      trades <- TradingExample.interpret(trade1)
      _ <- IO.println(s"  Trades: $trades")
      _ <- IO.println("")

      // Example 2: Logging effects
      _ <- IO.println("▶ EXAMPLE 2: Logging Effects")
      _ <- IO.println("─" * 70)
      _ <- LoggingExample.run(trade1)
      _ <- IO.println("")

      // Example 3: Given injection
      _ <- IO.println("▶ EXAMPLE 3: Given-Based Injection")
      _ <- IO.println("─" * 70)
      action <- GivenInjectionExample.interpret(trade1)
      _ <- IO.println(s"  Result: ${action.description}")
      _ <- IO.println("")

      // Example 5: Code generation
      _ <- IO.println("▶ EXAMPLE 5: Code Generation")
      _ <- IO.println("─" * 70)
      code <- CodeGenExample.generateCode(trade1)
      _ <- IO.println("  Generated code:")
      _ <- IO.println(code.split("\n").map("    " + _).mkString("\n"))
      _ <- IO.println("")

      _ <- IO.println("═" * 70)
    yield ()
