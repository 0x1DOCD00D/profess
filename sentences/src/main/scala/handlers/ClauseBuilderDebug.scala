package handlers

import profess.runtime.*

@main def runClauseBuilderDebug(): Unit =

  def printClause(expr: ProfessExpr): Unit =
    val ir     = expr.toIR
    val clause = ClauseBuilder.build(ir)
    println(s"  Plugin / Flat IR")
    println(IRDebug.renderTree(ir).linesIterator.map("    " + _).mkString("\n"))
    println(s"  ClauseBuilder output")
    println(s"    subject  : ${clause.subject.map(_.render).mkString(", ").ifEmpty("—")}")
    println(s"    head     : ${clause.head.ifEmpty("—")}")
    println(s"    args     : ${clause.args.map(_.render).mkString(", ").ifEmpty("—")}")
    if clause.modifiers.nonEmpty then
      clause.modifiers.foreach(p =>
        val fillers = if p.fillers.isEmpty then "—" else p.fillers.map(_.render).mkString(", ")
        println(s"    modifier : ${p.relation} → $fillers")
      )
    else
      println(s"    modifiers: —")
    println()

  extension (s: String) def ifEmpty(fallback: String): String = if s.isEmpty then fallback else s

  println("══════════════════════════════════════════════════════")
  println(" CLAUSE BUILDER OUTPUT — Trading domain (13 sentences)")
  println("══════════════════════════════════════════════════════")
  println()

  import sentences.TradingPlayground.*
  printClause(bought)
  printClause(sold)
  printClause(boughtTsla)
  printClause(filledPartial)
  printClause(dividend)
  printClause(deposit)
  printClause(commission)
  printClause(transfer)
  printClause(shorted)
  printClause(covered)
  printClause(hedged)
  printClause(rated)
  printClause(cancelled)

  println("══════════════════════════════════════════════════════")
  println(" CLAUSE BUILDER OUTPUT — Healthcare domain (13 sentences)")
  println("══════════════════════════════════════════════════════")
  println()

  import sentences.HealthcarePlayground.*
  printClause(prescribedAmox)
  printClause(prescribedMorphine)
  printClause(gaveAmox1)
  printClause(gaveAmox2)
  printClause(gaveMorphine)
  printClause(held)
  printClause(refused)
  printClause(dispensed)
  printClause(allergy)
  printClause(discontinued)
  printClause(reported)
  printClause(titrated)
  printClause(noted)
