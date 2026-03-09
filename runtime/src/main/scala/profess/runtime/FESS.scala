package profess.runtime

import scala.quoted.*
import scala.util.parsing.combinator.RegexParsers

sealed trait NLAst
case class NLObject(kind: String, name: String) extends NLAst
case class NLWord(word: String) extends NLAst
case class NLNumber(value: Double) extends NLAst
case class NLUnit(value: Double, unit: String) extends NLAst
case class NLString(value: String) extends NLAst
case class NLSentence(parts: List[NLAst])

object NLSentenceParser extends RegexParsers:
  override val skipWhitespace: Boolean = true

  private def kindId: Parser[String] = """[A-Za-z_][A-Za-z0-9_]*""".r
  private def ident: Parser[String] = """[A-Za-z_][A-Za-z0-9_]*""".r
  private def nameId: Parser[String] = """[A-Za-z0-9_]+""".r
  private def number: Parser[Double] = """-?\d+(?:\.\d+)?""".r ^^ (_.toDouble)
  private def quoted: Parser[String] =
    "\"([^\"\\\\]|\\\\.)*\"".r ^^ { raw =>
      raw.substring(1, raw.length - 1)
    }

  private def obj: Parser[NLAst] =
    "(" ~> kindId ~ nameId <~ ")" ^^ { case kind ~ name => NLObject(kind, name) }

  private def unitValue: Parser[NLAst] =
    number ~ ":" ~ ident ^^ { case value ~ _ ~ unit => NLUnit(value, unit) }

  private def str: Parser[NLAst] = quoted ^^ (NLString(_))
  private def num: Parser[NLAst] = number ^^ (NLNumber(_))
  private def word: Parser[NLAst] = ident ^^ (NLWord(_))
  private def token: Parser[NLAst] = obj | unitValue | str | num | word

  def parseSentence(input: String): Either[String, NLSentence] =
    parseAll(rep1(token), input) match
      case Success(result, _) => Right(NLSentence(result))
      case Failure(msg, next) =>
        Left(s"$msg at line ${next.pos.line}, column ${next.pos.column}")
      case Error(msg, next) =>
        Left(s"$msg at line ${next.pos.line}, column ${next.pos.column}")

object FESSMacro:
  def fessImpl(sentenceExpr: Expr[String])(using Quotes): Expr[ProfessExpr] =
    import quotes.reflect.*

    sentenceExpr.value match
      case None =>
        report.errorAndAbort("FESS requires a string literal, e.g. FESS(\"(broker Mark) sold 700\")")
      case Some(sentence) =>
        NLSentenceParser.parseSentence(sentence) match
          case Left(err) =>
            report.errorAndAbort(s"[FESS] parse error: $err")
          case Right(sentenceAst) =>
            val debugTokens = sentenceAst.parts.map {
              case NLObject(k, n) => s"Object($k,$n)"
              case NLWord(w) => s"Word($w)"
              case NLNumber(v) => s"Number($v)"
              case NLUnit(v, u) => s"Unit($v:$u)"
              case NLString(s) => s"""String("$s")"""
            }.mkString("[", ", ", "]")
            report.info(
              s"[FESS] macro input: $sentence | parsed tokens: $debugTokens",
              Position.ofMacroExpansion
            )
            val irExprs: List[Expr[IRNode]] = sentenceAst.parts.map {
              case NLObject(kind, name) => '{ IRObject(${ Expr(kind) }, ${ Expr(name) }) }
              case NLWord(word) => '{ IRWord(${ Expr(word) }) }
              case NLNumber(value) => '{ IRNumber(${ Expr(value) }) }
              case NLUnit(value, unit) => '{ IRUnitValue(${ Expr(value) }, ${ Expr(unit) }) }
              case NLString(value) => '{ IRString(${ Expr(value) }) }
            }
            '{ ProfessExpr(${ Expr.ofList(irExprs) }) }

inline def FESS(inline sentence: String): ProfessExpr =
  ${ FESSMacro.fessImpl('sentence) }
