package profess.plugin

import dotty.tools.dotc.ast.untpd.*
import dotty.tools.dotc.core.Constants.Constant
import dotty.tools.dotc.core.Contexts.*
import dotty.tools.dotc.core.Decorators.*

object FessIrLowering:

  sealed trait ParsedToken
  final case class ParsedObject(kind: String, name: String) extends ParsedToken
  final case class ParsedWord(word: String) extends ParsedToken
  final case class ParsedNumber(value: Double) extends ParsedToken
  final case class ParsedUnit(value: Double, unit: String) extends ParsedToken
  final case class ParsedString(value: String) extends ParsedToken

  def rewrite(tree: Tree)(using Context): Tree =
    object Rewriter extends UntypedTreeMap:
      override def transform(tree: Tree)(using Context): Tree =
        tree match
          case applyTree @ Apply(fun, args) if FessCallCollector.isFessFunction(fun) =>
            FessCallCollector.extractStringArgFromTree(args.headOption) match
              case Some(source) =>
                parseSentence(source) match
                  case Right(tokens) => buildProfessExpr(tokens).withSpan(applyTree.span)
                  case Left(_) => super.transform(tree)
              case None =>
                super.transform(tree)
          case _ =>
            super.transform(tree)

    Rewriter.transform(tree)

  def parseSentence(input: String): Either[String, List[ParsedToken]] =
    val tokens = scala.collection.mutable.ListBuffer.empty[ParsedToken]
    var i = 0

    def skipWhitespace(): Unit =
      while i < input.length && input.charAt(i).isWhitespace do i += 1

    def parseIdentifier(): Option[String] =
      if i >= input.length then None
      else
        val start = i
        val ch = input.charAt(i)
        if ch.isLetter || ch == '_' then
          i += 1
          while i < input.length && {
              val c = input.charAt(i)
              c.isLetterOrDigit || c == '_'
            }
          do i += 1
          Some(input.substring(start, i))
        else None

    def parseNameIdentifier(): Option[String] =
      if i >= input.length then None
      else
        val start = i
        val ch = input.charAt(i)
        if ch.isLetterOrDigit || ch == '_' then
          i += 1
          while i < input.length && {
              val c = input.charAt(i)
              c.isLetterOrDigit || c == '_'
            }
          do i += 1
          Some(input.substring(start, i))
        else None

    def parseNumberLiteral(): Option[String] =
      if i >= input.length then None
      else
        val start = i
        if input.charAt(i) == '-' then i += 1
        var digits = 0
        while i < input.length && input.charAt(i).isDigit do
          digits += 1
          i += 1
        if i < input.length && input.charAt(i) == '.' then
          i += 1
          while i < input.length && input.charAt(i).isDigit do
            digits += 1
            i += 1
        if digits == 0 then
          i = start
          None
        else Some(input.substring(start, i))

    def parseQuoted(): Either[String, String] =
      val sb = new StringBuilder
      i += 1
      var escaped = false
      while i < input.length do
        val ch = input.charAt(i)
        if escaped then
          sb += ch
          escaped = false
        else if ch == '\\' then
          escaped = true
        else if ch == '"' then
          i += 1
          return Right(sb.toString)
        else
          sb += ch
        i += 1
      Left("unterminated string literal")

    while
      skipWhitespace()
      i < input.length
    do
      input.charAt(i) match
        case '(' =>
          i += 1
          skipWhitespace()
          val parsed =
            for
              kind <- parseIdentifier().toRight("expected kind identifier")
              _ = skipWhitespace()
              name <- parseNameIdentifier().toRight("expected object name")
              _ = skipWhitespace()
              _ <-
                if i < input.length && input.charAt(i) == ')' then
                  i += 1
                  Right(())
                else Left("expected ')'")
            yield ParsedObject(kind, name)
          parsed match
            case Right(token) => tokens += token
            case Left(err) => return Left(err)

        case '"' =>
          parseQuoted() match
            case Right(value) => tokens += ParsedString(value)
            case Left(err) => return Left(err)

        case ch if ch == '-' || ch.isDigit =>
          val start = i
          parseNumberLiteral() match
            case Some(numberLit) =>
              if i < input.length && input.charAt(i) == ':' then
                i += 1
                parseIdentifier() match
                  case Some(unit) =>
                    tokens += ParsedUnit(numberLit.toDouble, unit)
                  case None =>
                    return Left("expected unit identifier after ':'")
              else
                tokens += ParsedNumber(numberLit.toDouble)
            case None =>
              return Left(s"invalid numeric literal at index $start")

        case _ =>
          parseIdentifier() match
            case Some(word) => tokens += ParsedWord(word)
            case None => return Left(s"unexpected character '${input.charAt(i)}' at index $i")

    Right(tokens.toList)

  private def buildProfessExpr(tokens: List[ParsedToken])(using Context): Tree =
    val irTrees = tokens.map(buildIrNode)
    Apply(
      runtimeSelect("ProfessExpr"),
      List(
        Apply(
          Select(
            Select(Ident("_root_".toTermName), "scala".toTermName),
            "List".toTermName
          ),
          irTrees
        )
      )
    )

  private def buildIrNode(token: ParsedToken)(using Context): Tree =
    token match
      case ParsedObject(kind, name) =>
        Apply(runtimeSelect("IRObject"), List(stringLit(kind), stringLit(name)))
      case ParsedWord(word) =>
        Apply(runtimeSelect("IRWord"), List(stringLit(word)))
      case ParsedNumber(value) =>
        Apply(runtimeSelect("IRNumber"), List(doubleLit(value)))
      case ParsedUnit(value, unit) =>
        Apply(runtimeSelect("IRUnitValue"), List(doubleLit(value), stringLit(unit)))
      case ParsedString(value) =>
        Apply(runtimeSelect("IRString"), List(stringLit(value)))

  private def runtimeSelect(term: String)(using Context): Tree =
    Select(
      Select(
        Select(Ident("_root_".toTermName), "profess".toTermName),
        "runtime".toTermName
      ),
      term.toTermName
    )

  private def stringLit(value: String)(using Context): Tree =
    Literal(Constant(value))

  private def doubleLit(value: Double)(using Context): Tree =
    Literal(Constant(value))
