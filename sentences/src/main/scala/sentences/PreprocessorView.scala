package sentences

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

object PreprocessorView:
  private val managedSuffix =
    Paths.get("src_managed", "main", "profess", "sentences", "SentencePlayground.scala")

  def findManagedSource(root: Path): Option[Path] =
    if !Files.exists(root) then None
    else
      val paths = Files.walk(root)
      try
        paths.iterator().asScala.find { p =>
          p.endsWith(managedSuffix)
        }
      finally
        paths.close()

@main def runPreprocessorView(): Unit =
  val targetDir = Paths.get("sentences", "target")
  PreprocessorView.findManagedSource(targetDir) match
    case Some(path) =>
      println(s"Managed source: $path")
      println("=== Rewritten ===")
      println(Files.readString(path))
    case None =>
      println("Managed source not found. Run `sbt sentences/compile` first.")
