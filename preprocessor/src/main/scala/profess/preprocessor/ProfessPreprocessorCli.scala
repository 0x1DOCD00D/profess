package profess.preprocessor

import java.nio.file.{Files, Path}

object ProfessPreprocessorCli {
  private def rewriteFile(inputPath: String, outputPath: String): Unit = {
    val inPath = Path.of(inputPath)
    val outPath = Path.of(outputPath)
    val content = Files.readString(inPath)
    val rewritten = ProfessPreprocessorSupport.preprocessProfessSource(content)
    val parent = outPath.getParent
    if (parent != null) Files.createDirectories(parent)
    Files.writeString(outPath, rewritten)
  }

  def main(args: Array[String]): Unit = {
    args.toList match
      case "--self-test" :: Nil =>
        ProfessPreprocessorSelfTest.run(msg => println(s"[preprocessor-self-test] $msg"))
      case fileArgs if fileArgs.nonEmpty && fileArgs.length % 2 == 0 =>
        fileArgs.grouped(2).foreach {
          case inputPath :: outputPath :: Nil =>
            rewriteFile(inputPath, outputPath)
          case _ =>
            throw new IllegalStateException("internal CLI argument grouping failure")
        }
      case _ =>
        Console.err.println(
          "Usage:\n" +
            "  ProfessPreprocessorCli <input> <output.scala> [<input> <output.scala> ...]\n" +
            "  ProfessPreprocessorCli --self-test"
        )
        sys.exit(2)
  }
}
