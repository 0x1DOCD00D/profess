package profess.plugin

import dotty.tools.dotc.ast.untpd
import dotty.tools.dotc.core.Contexts.*

object ASTPrinter:
  def print(tree: untpd.Tree)(using Context): String =
    tree.show
