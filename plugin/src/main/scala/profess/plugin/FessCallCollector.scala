package profess.plugin

import dotty.tools.dotc.ast.untpd.*
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Constants.Constant

import scala.collection.mutable

final case class SourcePoint(line: Int, column: Int)

final case class FessCallSite(
    owner: Option[String],
    sourceText: String,
    position: SourcePoint
)

object FessCallCollector:
  // Temporary parser-phase collector for FESS(...) call sites.
  // This runs on the untyped tree, so matching is intentionally name-based.
  // Once FESS is resolved as a stable symbol after typer, this should move to
  // symbol-based matching against the fully-qualified runtime entrypoint.
  def collect(tree: Tree)(using Context): List[FessCallSite] =
    val traverser = new Collector
    traverser.traverse(tree)
    traverser.result

  def isFessFunction(tree: Tree): Boolean =
    tree match
      case Ident(name) =>
        name.toString == "FESS"
      case Select(_, name) =>
        name.toString == "FESS"
      case TypeApply(fun, _) =>
        isFessFunction(fun)
      case _ =>
        false

  def extractStringArgFromTree(treeOpt: Option[Tree]): Option[String] =
    treeOpt.flatMap {
      case Literal(Constant(value: String)) => Some(value)
      case Typed(expr, _) => extractStringArgFromTree(Some(expr))
      case NamedArg(_, expr) => extractStringArgFromTree(Some(expr))
      case _ => None
    }

  private final class Collector(using Context) extends UntypedTreeTraverser:
    private val ownerStack = mutable.ArrayBuffer.empty[String]
    private val sites = mutable.ListBuffer.empty[FessCallSite]

    def result: List[FessCallSite] = sites.toList

    override def traverse(tree: Tree)(using Context): Unit =
      tree match
        case vd @ ValDef(name, _, _) =>
          withOwner(name.toString) {
            traverseChildren(vd)
          }

        case dd @ DefDef(name, _, _, _) =>
          withOwner(name.toString) {
            traverseChildren(dd)
          }

        case applyTree @ Apply(fun, args) if isFessFunction(fun) =>
          extractStringArgFromTree(args.headOption).foreach { text =>
            sites += FessCallSite(
              owner = if ownerStack.nonEmpty then Some(ownerStack.mkString(".")) else None,
              sourceText = text,
              position = sourcePoint(applyTree)
            )
          }
          traverseChildren(applyTree)

        case _ =>
          traverseChildren(tree)

    private def withOwner[A](owner: String)(body: => A): A =
      ownerStack += owner
      try body
      finally ownerStack.remove(ownerStack.size - 1)

    // Name-based matching is acceptable only at this untyped parser-phase stage.
    // It may over-match unrelated FESS identifiers until this collector is moved
    // to a typed/symbol-aware phase.
    private def sourcePoint(tree: Tree): SourcePoint =
      if tree.span.exists then
        val pos = ctx.source.atSpan(tree.span)
        SourcePoint(pos.line + 1, pos.column + 1)
      else SourcePoint(-1, -1)
