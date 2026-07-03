package handlers

import cats.Id
import cats.Monoid
import cats.implicits.*
import profess.runtime.*

/**
 * Healthcare DSL — DSL-AUTHOR territory (not framework).
 *
 * Defines the state type (MedRecord/Chart), navigation helpers over Clause roles,
 * and the ClauseRegistry that maps head relations to handler logic.
 * Mirrors domain-specs/healthcare-expert-view.md.
 */

/** A medication order — an intention. Fields are partial by nature. */
final case class Order(
  dose: Option[String] = None,
  route: Option[String] = None,
  frequency: Option[String] = None,
  duration: Option[String] = None,
  prescriber: Option[String] = None
)

/** A dose that was actually administered — a real event, kept separate from orders. */
final case class AdminEvent(drug: String, dose: Option[String], route: Option[String], nurse: String)

/** A patient's medication record — the per-entity state that accumulates. */
final case class MedRecord(
  active: Map[String, Order] = Map.empty,
  administrations: List[AdminEvent] = Nil,
  held: List[String] = Nil,
  refused: List[String] = Nil,
  discontinued: Set[String] = Set.empty,
  allergies: List[String] = Nil
)

object MedRecord:
  given Monoid[MedRecord] with
    def empty: MedRecord = MedRecord()
    def combine(a: MedRecord, b: MedRecord): MedRecord =
      MedRecord(
        active = a.active ++ b.active,
        administrations = a.administrations ++ b.administrations,
        held = a.held ++ b.held,
        refused = a.refused ++ b.refused,
        discontinued = a.discontinued ++ b.discontinued,
        allergies = a.allergies ++ b.allergies
      )

/** The whole ward: patient id -> record. Map-merge Monoid comes from cats. */
type Chart = Map[String, MedRecord]

object HealthcareHandlers:

  // ── clause navigation helpers ─────────────────────────────────────────────

  private def fmt(d: Double): String =
    if d == d.toLong then d.toLong.toString else d.toString

  private def subjectKind(clause: Clause, kind: String): Option[String] =
    clause.subject.collectFirst { case IRObject(k, n) if k == kind => n }

  private def argKind(clause: Clause, kind: String): Option[String] =
    clause.args.collectFirst { case IRObject(k, n) if k == kind => n }

  private def modKind(clause: Clause, relation: String, kind: String): Option[String] =
    clause.modifiers.find(_.relation == relation)
      .flatMap(_.fillers.collectFirst { case IRObject(k, n) if k == kind => n })

  private def modUnit(clause: Clause, relation: String): Option[IRUnitValue] =
    clause.modifiers.find(_.relation == relation)
      .flatMap(_.fillers.collectFirst { case u: IRUnitValue => u })

  // dose is a unit value in args (the first mg or ml value)
  private def dose(clause: Clause): Option[String] =
    clause.args.collectFirst { case IRUnitValue(v, u) if u == "mg" || u == "ml" => s"${fmt(v)}:$u" }

  // frequency is the filler of the "every" modifier
  private def freq(clause: Clause): Option[String] =
    modUnit(clause, "every").map(u => s"every ${fmt(u.value)}h")

  // duration is the filler of the "for" modifier when it carries a unit value
  private def duration(clause: Clause): Option[String] =
    modUnit(clause, "for").map(u => s"${fmt(u.value)} days")

  private def one(patient: String, rec: MedRecord): Chart = Map(patient -> rec)

  val registry: ClauseRegistry[Id, Chart] =
    ClauseRegistry.empty[Id, Chart]

      // drug in "of", patient in "to", route in "via", freq in "every", duration in "for"
      .onRelation("prescribed") { (clause, _, _) =>
        val patient = modKind(clause, "to", "patient")
        val drug    = modKind(clause, "of", "drug")
        (patient, drug) match
          case (Some(p), Some(d)) =>
            val order = Order(
              dose      = dose(clause),
              route     = modKind(clause, "via", "route"),
              frequency = freq(clause),
              duration  = duration(clause),
              prescriber = subjectKind(clause, "doctor")
            )
            one(p, MedRecord(active = Map(d -> order)))
          case _ => Monoid[Chart].empty
      }

      // nurse is the subject; drug in "of", patient in "to", route in "via"
      .onRelation("administered") { (clause, _, _) =>
        val patient = modKind(clause, "to", "patient")
        val drug    = modKind(clause, "of", "drug")
        val nurse   = subjectKind(clause, "nurse")
        (patient, drug, nurse) match
          case (Some(p), Some(d), Some(n)) =>
            one(p, MedRecord(administrations = List(AdminEvent(d, dose(clause), modKind(clause, "via", "route"), n))))
          case _ => Monoid[Chart].empty
      }

      // drug is a direct arg; patient is in the "for" modifier
      .onRelation("held") { (clause, _, _) =>
        val patient = modKind(clause, "for", "patient")
        val drug    = argKind(clause, "drug")
        (patient, drug) match
          case (Some(p), Some(d)) => one(p, MedRecord(held = List(d)))
          case _                  => Monoid[Chart].empty
      }

      // patient is the subject; drug is a direct arg
      .onRelation("refused") { (clause, _, _) =>
        val patient = subjectKind(clause, "patient")
        val drug    = argKind(clause, "drug")
        (patient, drug) match
          case (Some(p), Some(d)) => one(p, MedRecord(refused = List(d)))
          case _                  => Monoid[Chart].empty
      }

      // drug is a direct arg; patient is in the "for" modifier
      .onRelation("discontinued") { (clause, _, _) =>
        val patient = modKind(clause, "for", "patient")
        val drug    = argKind(clause, "drug")
        (patient, drug) match
          case (Some(p), Some(d)) => one(p, MedRecord(discontinued = Set(d)))
          case _                  => Monoid[Chart].empty
      }

      // allergy is a direct arg; patient is in the "for" modifier
      .onRelation("recorded") { (clause, _, _) =>
        val patient = modKind(clause, "for", "patient")
        val allergy = argKind(clause, "allergy")
        (patient, allergy) match
          case (Some(p), Some(a)) => one(p, MedRecord(allergies = List(a)))
          case _                  => Monoid[Chart].empty
      }


@main def runHealthcareHandlers(): Unit =
  import sentences.HealthcarePlayground.*

  val exprs = List(
    prescribedAmox, prescribedMorphine, gaveAmox1, gaveAmox2, gaveMorphine,
    held, refused, dispensed, allergy, discontinued, reported, titrated, noted
  )

  val (chart, misses) =
    exprs.foldLeft((Monoid[Chart].empty, List.empty[String])) { case ((acc, ms), s) =>
      val (result, miss) = HealthcareHandlers.registry.run(s.toIR)
      (Monoid[Chart].combine(acc, result), ms ++ miss.toList)
    }

  println("══════════════════════════════════════════")
  println(" PER-PATIENT MEDICATION RECORDS (consumer view)")
  println("══════════════════════════════════════════")
  chart.toSeq.sortBy(_._1).foreach { case (patient, rec) =>
    println(s"patient $patient")
    println("  active orders:")
    rec.active.toSeq.sortBy(_._1).foreach { case (drug, o) =>
      val parts = List(
        o.dose.map(d => s"dose $d"),
        o.route.map(r => s"route $r"),
        o.frequency,
        o.duration.map(d => s"for $d"),
        o.prescriber.map(p => s"by $p")
      ).flatten.mkString(", ")
      println(s"    · $drug — $parts")
    }
    println("  administered:")
    rec.administrations.foreach { a =>
      println(s"    · ${a.drug} ${a.dose.getOrElse("?")} via ${a.route.getOrElse("?")} by ${a.nurse}")
    }
    if rec.held.nonEmpty        then println(s"  held        : ${rec.held.mkString(", ")}")
    if rec.refused.nonEmpty     then println(s"  refused     : ${rec.refused.mkString(", ")}")
    if rec.discontinued.nonEmpty then println(s"  discontinued: ${rec.discontinued.mkString(", ")}")
    if rec.allergies.nonEmpty   then println(s"  allergies   : ${rec.allergies.mkString(", ")}")
  }

  println()
  println("══════════════════════════════════════════")
  println(" ENGINEER LOG: relations with no handler")
  println("══════════════════════════════════════════")
  misses.distinct.foreach(r => println(s"  - $r"))
