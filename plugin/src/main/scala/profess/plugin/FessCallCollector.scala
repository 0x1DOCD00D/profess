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
  def collect(tree: Tree)(using Context): List[FessCallSite] =
    val traverser = new Collector
    traverser.traverse(tree)
    traverser.result

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
          extractStringArg(args.headOption).foreach { text =>
            sites += FessCallSite(
              owner = ownerStack.lastOption,
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

    private def isFessFunction(tree: Tree): Boolean =
      tree match
        case Ident(name) =>
          name.toString == "FESS"
        case Select(_, name) =>
          name.toString == "FESS"
        case TypeApply(fun, _) =>
          isFessFunction(fun)
        case _ =>
          false

    private def extractStringArg(treeOpt: Option[Tree]): Option[String] =
      treeOpt.flatMap {
        case Literal(Constant(value: String)) => Some(value)
        case Typed(expr, _) => extractStringArg(Some(expr))
        case NamedArg(_, expr) => extractStringArg(Some(expr))
        case _ => None
      }

    private def sourcePoint(tree: Tree): SourcePoint =
      if tree.span.exists then
        val pos = ctx.source.atSpan(tree.span)
        SourcePoint(pos.line + 1, pos.column + 1)
      else SourcePoint(-1, -1)
