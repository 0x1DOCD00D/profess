package profess.preprocessor

object ProfessPreprocessorSupport {
  // Delimiter-only mode: only @:- ... -:@ blocks are rewritten to FESS("...").
  private val MarkerStart = "@:-"
  private val MarkerEnd = "-:@"

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

  def preprocessProfessSource(content: String): String =
    wrapDelimitedBlocks(content)
}
