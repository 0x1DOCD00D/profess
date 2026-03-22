package profess.preprocessor

import java.nio.file.{Files, Path}

object ProfessPreprocessorCli {
  def main(args: Array[String]): Unit = {
    args.toList match
      case "--self-test" :: Nil =>
        ProfessPreprocessorSelfTest.run(msg => println(s"[preprocessor-self-test] $msg"))
      case inputPath :: outputPath :: Nil =>
        val inPath = Path.of(inputPath)
        val outPath = Path.of(outputPath)
        val content = Files.readString(inPath)
        val rewritten = ProfessPreprocessorSupport.preprocessProfessSource(content)
        val parent = outPath.getParent
        if (parent != null) Files.createDirectories(parent)
        Files.writeString(outPath, rewritten)
      case _ =>
        Console.err.println(
          "Usage:\n" +
            "  ProfessPreprocessorCli <input.scala> <output.scala>\n" +
            "  ProfessPreprocessorCli --self-test"
        )
        sys.exit(2)
  }
}
