package handlers

import cats.Id
import cats.Monoid
import cats.implicits.*
import profess.runtime.*

/**
 * Trading DSL — DSL-AUTHOR territory (not framework).
 *
 * Defines the state type (Book/Desk), navigation helpers over Clause roles,
 * and the ClauseRegistry that maps head relations to handler logic.
 * Mirrors domain-specs/trading-expert-view.md.
 */

/** A broker's book — the per-entity state that accumulates across sentences. */
final case class Book(
  holdings: Map[String, Double] = Map.empty, // ticker -> net shares (long +, short -)
  cash: Double = 0.0,
  dividends: Double = 0.0,
  log: List[String] = Nil
)

object Book:
  given Monoid[Book] with
    def empty: Book = Book()
    def combine(a: Book, b: Book): Book =
      Book(
        holdings = b.holdings.foldLeft(a.holdings) { case (m, (k, v)) =>
          m.updated(k, m.getOrElse(k, 0.0) + v)
        },
        cash = a.cash + b.cash,
        dividends = a.dividends + b.dividends,
        log = a.log ++ b.log
      )

/** The whole desk: broker name -> Book. Map-merge Monoid comes from cats. */
type Desk = Map[String, Book]

object TradingHandlers:

  // ── clause navigation helpers ─────────────────────────────────────────────

  private def subjectBroker(clause: Clause): Option[String] =
    clause.subject.collectFirst { case IRObject("broker", n) => n }

  private def argQty(clause: Clause): Option[Double] =
    clause.args.collectFirst { case IRNumber(v) => v }

  private def argStock(clause: Clause): Option[String] =
    clause.args.collectFirst { case IRObject("stock", n) => n }

  private def modFillersOf(clause: Clause, relation: String): List[IRNode] =
    clause.modifiers.find(_.relation == relation).map(_.fillers).getOrElse(Nil)

  private def priceDollars(clause: Clause): Option[Double] =
    modFillersOf(clause, "at").collectFirst { case IRUnitValue(v, "dollars") => v }

  private def one(broker: String, book: Book): Desk = Map(broker -> book)

  // buy/sell shape: subject is the broker, args carry qty and stock,
  // "at" modifier carries the price. shareSign/cashSign flip for sells vs buys.
  private def trade(clause: Clause, shareSign: Double, cashSign: Double): Desk =
    (subjectBroker(clause), argStock(clause), argQty(clause)) match
      case (Some(b), Some(s), Some(q)) =>
        val cashDelta = priceDollars(clause).map(p => cashSign * q * p).getOrElse(0.0)
        one(b, Book(
          holdings = Map(s -> shareSign * q),
          cash = cashDelta,
          log = List(s"$b ${clause.head} ${q.toLong} $s")
        ))
      case _ => Monoid[Desk].empty

  val registry: ClauseRegistry[Id, Desk] =
    ClauseRegistry.empty[Id, Desk]
      .onRelation("bought")  ((clause, _, _) => trade(clause, +1.0, -1.0))
      .onRelation("filled")  ((clause, _, _) => trade(clause, +1.0, -1.0))
      .onRelation("covered") ((clause, _, _) => trade(clause, +1.0, -1.0))
      .onRelation("sold")    ((clause, _, _) => trade(clause, -1.0, +1.0))
      .onRelation("shorted") ((clause, _, _) => trade(clause, -1.0, +1.0))
      // amount is a direct arg; "dividend" and "from" are zero/one-filler modifiers
      .onRelation("received") { (clause, _, _) =>
        subjectBroker(clause) match
          case Some(b) =>
            val amt = clause.args.collectFirst { case IRUnitValue(v, _) => v }.getOrElse(0.0)
            one(b, Book(cash = amt, dividends = amt, log = List(s"$b received $amt dividend")))
          case None => Monoid[Desk].empty
      }
      // source broker is the subject; destination broker is in the "to" modifier
      .onRelation("transferred") { (clause, _, _) =>
        val src = subjectBroker(clause)
        val dst = modFillersOf(clause, "to").collectFirst { case IRObject("broker", n) => n }
        (src, dst, argStock(clause), argQty(clause)) match
          case (Some(s), Some(d), Some(stock), Some(q)) =>
            Map(
              s -> Book(holdings = Map(stock -> -q), log = List(s"$s transferred ${q.toLong} $stock to $d")),
              d -> Book(holdings = Map(stock ->  q), log = List(s"$d received ${q.toLong} $stock from $s"))
            )
          case _ => Monoid[Desk].empty
      }


@main def runTradingHandlers(): Unit =
  import sentences.TradingPlayground.*

  val exprs = List(
    bought, sold, boughtTsla, filledPartial, dividend, deposit, commission,
    transfer, shorted, covered, hedged, rated, cancelled
  )

  val (desk, misses) =
    exprs.foldLeft((Monoid[Desk].empty, List.empty[String])) { case ((acc, ms), s) =>
      val (result, miss) = TradingHandlers.registry.run(s.toIR)
      (Monoid[Desk].combine(acc, result), ms ++ miss.toList)
    }

  println("══════════════════════════════════════════")
  println(" PER-BROKER BOOKS (consumer view)")
  println("══════════════════════════════════════════")
  desk.toSeq.sortBy(_._1).foreach { case (broker, book) =>
    println(s"$broker")
    println(s"  holdings : ${book.holdings.toSeq.sortBy(_._1).map { case (s, n) => s"$s ${n.toLong}" }.mkString(", ")}")
    println(s"  cash     : ${book.cash}")
    println(s"  dividends: ${book.dividends}")
    book.log.foreach(l => println(s"    · $l"))
  }

  println()
  println("══════════════════════════════════════════")
  println(" ENGINEER LOG: relations with no handler")
  println("══════════════════════════════════════════")
  misses.distinct.foreach(r => println(s"  - $r"))
