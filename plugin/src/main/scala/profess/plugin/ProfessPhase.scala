/*
 * Copyright (c) 2025 Mark Grechanik and Lone Star Consulting, Inc. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.
 */

package profess.plugin

import dotty.tools.dotc.ast.tpd
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.plugins.*
import dotty.tools.dotc.report

/** PROFESS Transform Phase
  *
  * Runs after parser, before typer.
  */
class ProfessPhase(debug: Boolean, dumpAst: Boolean) extends PluginPhase:

  val phaseName: String = "profess"

  override val runsAfter: Set[String] = Set("parser")
  override val runsBefore: Set[String] = Set("typer")

  override def prepareForUnit(tree: tpd.Tree)(using Context): Context =
    val unit = ctx.compilationUnit
    val fileName = unit.source.file.name
    // Always echo so P02 "prints diagnostic message" is visible in sbt
    report.echo(s"[PROFESS] processed $fileName")
    if debug then
      report.echo(s"[PROFESS] (debug) phase ran for $fileName")
    val untypedTree = unit.untpdTree

    if untypedTree != null then
      val fessCallSites = FessCallCollector.collect(untypedTree)
      val containsFessCall = fessCallSites.nonEmpty

      if debug && containsFessCall then
        report.echo(s"[PROFESS] FESS call sites for $fileName")
        fessCallSites.foreach { site =>
          val owner = site.owner.getOrElse("<none>")
          report.echo(
            s"[PROFESS]   owner=$owner at ${site.position.line}:${site.position.column} input=${site.sourceText}"
          )
        }

      if dumpAst && containsFessCall then
        report.echo(s"[PROFESS] AST before transform for $fileName")
        report.echo(ASTPrinter.print(untypedTree))

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
          !declared.contains(id) && !ScalaKeywords.contains(id)
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
          val transformed =
            ASTInjector.inject(untypedTree, scaffolding, declared)

          unit.untpdTree = transformed

      if dumpAst && containsFessCall then
        report.echo(s"[PROFESS] AST after transform for $fileName")
        report.echo(ASTPrinter.print(unit.untpdTree))

    ctx
