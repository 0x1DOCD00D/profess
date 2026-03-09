object ProfessPreprocessorSupport {
  import java.io.File
  // Build-time preprocessor used by sbt source generation.
  // It rewrites raw PROFESS-like assignment RHS into FESS("...") so Scala sees valid syntax.
  private val MarkerStart = "@:-"
  private val MarkerEnd = "-:@"
  private val Ws = """\s+""".r
  // Grammar atoms for preprocessor detection (structural, domain-agnostic).
  // Extend these patterns when introducing new DSL surface forms.
  private val Obj = """\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+(!?[A-Za-z_][A-Za-z0-9_]*)\s*\)""".r
  private val Quoted = """"([^"\\]|\\.)*"""".r
  private val UnitValue = """-?\d+(?:\.\d+)?:[A-Za-z_][A-Za-z0-9_]*""".r
  private val Number = """-?\d+(?:\.\d+)?""".r
  private val VarRef = """![A-Za-z_][A-Za-z0-9_]*""".r
  private val Word = """[A-Za-z_][A-Za-z0-9_]*""".r

  private sealed trait NlToken
  private final case class ObjTok(kind: String, name: String) extends NlToken
  private final case class UnitTok(value: String, unit: String) extends NlToken
  private final case class WordTok(word: String) extends NlToken
  private final case class NumTok(value: String) extends NlToken
  private final case class StrTok(value: String) extends NlToken
  private final case class VarTok(name: String) extends NlToken

  def isProfessEnabled(content: String): Boolean =
    content.linesIterator.exists(_.trim == "// @profess")

  private def nextToken(input: String, from: Int): Option[(NlToken, Int)] = {
    val tail = input.substring(from)

    val matchers =
      List(
        Obj.findPrefixMatchOf(tail).map(m => (ObjTok(m.group(1), m.group(2)): NlToken, m.end)),
        Quoted.findPrefixMatchOf(tail).map(m => (StrTok(m.matched): NlToken, m.end)),
        UnitValue.findPrefixMatchOf(tail).map { m =>
          val parts = m.matched.split(":", 2)
          (UnitTok(parts(0), parts(1)): NlToken, m.end)
        },
        Number.findPrefixMatchOf(tail).map(m => (NumTok(m.matched): NlToken, m.end)),
        VarRef.findPrefixMatchOf(tail).map(m => (VarTok(m.matched.drop(1)): NlToken, m.end)),
        Word.findPrefixMatchOf(tail).map(m => (WordTok(m.matched): NlToken, m.end))
      ).flatten

    matchers.headOption.map { case (tok, len) => (tok, from + len) }
  }

  private def parseNlTokens(input: String): Option[List[NlToken]] = {
    val tokens = scala.collection.mutable.ListBuffer.empty[NlToken]
    var i = 0

    while (i < input.length) {
      Ws.findPrefixMatchOf(input.substring(i)) match {
        case Some(m) => i += m.end
        case None =>
          nextToken(input, i) match {
            case Some((tok, j)) =>
              tokens += tok
              i = j
            case None =>
              return None
          }
      }
    }

    Some(tokens.toList)
  }

  // Conservative rewrite gate:
  // 1) tokenization must fully succeed
  // 2) at least one structured token must appear (object/unit/var/string)
  // This avoids rewriting normal Scala expressions accidentally.
  private def looksLikeProfessRhs(rhs: String, profile: ProfessDslProfile.Profile): Boolean = {
    parseNlTokens(rhs).exists { tokens =>
      val hasStructured =
        tokens.exists {
          case ObjTok(_, _) | UnitTok(_, _) | VarTok(_) => true
          case _ => false
        }

      val profileCompatible =
        if (profile.isEmpty) true
        else
          tokens.forall {
            case ObjTok(kind, _) =>
              profile.kinds.isEmpty || profile.kinds.contains(kind)
            case WordTok(word) =>
              profile.words.isEmpty || profile.words.contains(word)
            case UnitTok(_, unit) =>
              profile.units.isEmpty || profile.units.contains(unit)
            case _ =>
              true
          }

      hasStructured && tokens.nonEmpty && profileCompatible
    }
  }

  private def escapeForScalaString(s: String): String =
    s.flatMap {
      case '"'  => "\\\""
      case '\\' => "\\\\"
      case '\n' => "\\n"
      case '\r' => "\\r"
      case c    => c.toString
    }

  private def wrapDelimitedBlocks(content: String): String = {
    val out = new StringBuilder
    var i = 0
    var inString = false
    var escaped = false

    def findEndOutsideString(from: Int): Int = {
      var j = from
      var localInString = false
      var localEscaped = false
      while (j < content.length) {
        val c = content.charAt(j)
        if (localInString) {
          if (localEscaped) localEscaped = false
          else if (c == '\\') localEscaped = true
          else if (c == '"') localInString = false
          j += 1
        } else if (c == '"') {
          localInString = true
          j += 1
        } else if (content.startsWith(MarkerEnd, j)) {
          return j
        } else {
          j += 1
        }
      }
      -1
    }

    while (i < content.length) {
      val ch = content.charAt(i)
      if (inString) {
        out.append(ch)
        if (escaped) escaped = false
        else if (ch == '\\') escaped = true
        else if (ch == '"') inString = false
        i += 1
      } else if (ch == '"') {
        inString = true
        out.append(ch)
        i += 1
      } else if (content.startsWith(MarkerStart, i)) {
        val end = findEndOutsideString(i + MarkerStart.length)
        if (end < 0) {
          out.append(content.substring(i))
          i = content.length
        } else {
          val inner = content.substring(i + MarkerStart.length, end).trim
          out.append(s"""FESS("${escapeForScalaString(inner)}")""")
          i = end + MarkerEnd.length
        }
      } else {
        out.append(ch)
        i += 1
      }
    }

    out.toString
  }

  def preprocessProfessSource(content: String, sourceFile: Option[File] = None): String = {
    val profile = ProfessDslProfile.fromSource(content, sourceFile)
    val assignPattern =
      """^(\s*)(val|var)\s+([A-Za-z_][A-Za-z0-9_]*)(\s*:\s*[^=]+)?\s*=\s*(.*)$""".r
    val withDelimitedBlocksWrapped = wrapDelimitedBlocks(content)
    val lines = withDelimitedBlocksWrapped.split("\\n", -1).toList
    val out = scala.collection.mutable.ListBuffer.empty[String]

    var i = 0
    while (i < lines.length) {
      lines(i) match {
        case assignPattern(indent, keyword, name, tpeRaw, rhsRaw) =>
          val tpe = Option(tpeRaw).getOrElse("")
          val rhs = Option(rhsRaw).getOrElse("").trim

          if (rhs.nonEmpty) {
            if (rhs.contains("FESS(") || !looksLikeProfessRhs(rhs, profile))
              out += s"$indent$keyword $name$tpe = $rhs"
            else
              out += s"""$indent$keyword $name$tpe = FESS("${escapeForScalaString(rhs)}")"""
            i += 1
          } else if (i + 1 < lines.length) {
            val next = lines(i + 1).trim
            if (next.startsWith("FESS("))
              out += s"$indent$keyword $name$tpe = $next"
            else if (next.nonEmpty && looksLikeProfessRhs(next, profile))
              out += s"""$indent$keyword $name$tpe = FESS("${escapeForScalaString(next)}")"""
            else
              out += lines(i)
            i += 2
          } else {
            out += lines(i)
            i += 1
          }
        case _ =>
          out += lines(i)
          i += 1
      }
    }

    out.mkString("\n")
  }
}
