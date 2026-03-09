object ProfessDslProfile {
  import java.io.File
  import sbt.io.IO

  final case class Profile(
      kinds: Set[String],
      words: Set[String],
      units: Set[String]
  ) {
    val isEmpty: Boolean = kinds.isEmpty && words.isEmpty && units.isEmpty
  }

  val Empty: Profile = Profile(Set.empty, Set.empty, Set.empty)

  private val KindsPrefix = "// @profess-kinds:"
  private val WordsPrefix = "// @profess-words:"
  private val UnitsPrefix = "// @profess-units:"
  private val ProfilePathPrefix = "// @profess-profile:"

  private def parseCsv(raw: String): Set[String] =
    raw
      .split(",")
      .iterator
      .map(_.trim)
      .filter(_.nonEmpty)
      .toSet

  private def fromLines(lines: List[String]): Profile = {
    val kinds = lines
      .collectFirst { case l if l.trim.startsWith(KindsPrefix) =>
        parseCsv(l.trim.stripPrefix(KindsPrefix))
      }
      .getOrElse(Set.empty)

    val words = lines
      .collectFirst { case l if l.trim.startsWith(WordsPrefix) =>
        parseCsv(l.trim.stripPrefix(WordsPrefix))
      }
      .getOrElse(Set.empty)

    val units = lines
      .collectFirst { case l if l.trim.startsWith(UnitsPrefix) =>
        parseCsv(l.trim.stripPrefix(UnitsPrefix))
      }
      .getOrElse(Set.empty)

    Profile(kinds = kinds, words = words, units = units)
  }

  private def merge(base: Profile, overrideBy: Profile): Profile =
    Profile(
      kinds = if (overrideBy.kinds.nonEmpty) overrideBy.kinds else base.kinds,
      words = if (overrideBy.words.nonEmpty) overrideBy.words else base.words,
      units = if (overrideBy.units.nonEmpty) overrideBy.units else base.units
    )

  private def resolveProfileFile(profilePath: String, sourceFile: File): File = {
    val raw = new File(profilePath)
    if (raw.isAbsolute) raw
    else new File(sourceFile.getParentFile, profilePath)
  }

  def fromSource(content: String, sourceFile: Option[File] = None): Profile = {
    val lines = content.linesIterator.toList
    val inline = fromLines(lines)
    val externalPath = lines.collectFirst {
      case l if l.trim.startsWith(ProfilePathPrefix) =>
        l.trim.stripPrefix(ProfilePathPrefix).trim
    }.filter(_.nonEmpty)

    (sourceFile, externalPath) match {
      case (Some(src), Some(path)) =>
        val file = resolveProfileFile(path, src)
        if (file.exists() && file.isFile) {
          val fileLines = IO.read(file).linesIterator.toList
          merge(fromLines(fileLines), inline)
        } else inline
      case _ =>
        inline
    }
  }
}
