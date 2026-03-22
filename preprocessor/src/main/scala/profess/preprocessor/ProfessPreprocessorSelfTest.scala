package profess.preprocessor

object ProfessPreprocessorSelfTest {
  def run(log: String => Unit): Unit = {
    def checkCase(name: String, input: String, expected: String): Unit = {
      val actual = ProfessPreprocessorSupport.preprocessProfessSource(input)
      if (actual != expected) {
        val msg =
          s"""Preprocessor case failed: $name
             |--- Input ---
             |$input
             |--- Expected ---
             |$expected
             |--- Actual ---
             |$actual
             |""".stripMargin
        throw new IllegalStateException(msg)
      } else {
        log(s"PASS: $name")
      }
    }

    checkCase(
      "unclosed marker is left unchanged",
      """|object X:
         |  val trade = @:- (broker Mark) sold 700 (stock MSFT) at 150:dollars
         |""".stripMargin,
      """|object X:
         |  val trade = @:- (broker Mark) sold 700 (stock MSFT) at 150:dollars
         |""".stripMargin
    )

    checkCase(
      "stray end marker is left unchanged",
      """|object X:
         |  val trade = (broker Mark) sold 700 (stock MSFT) at 150:dollars -:@
         |""".stripMargin,
      """|object X:
         |  val trade = (broker Mark) sold 700 (stock MSFT) at 150:dollars -:@
         |""".stripMargin
    )

    checkCase(
      "adjacent marker blocks rewrite independently",
      """|object X:
         |  val a = @:- (broker Mark) sold 1 (stock MSFT) at 1:dollars -:@
         |  val b = @:- (broker Jane) bought 2 (stock AAPL) at 2:dollars -:@
         |""".stripMargin,
      """|object X:
         |  val a = FESS("(broker Mark) sold 1 (stock MSFT) at 1:dollars")
         |  val b = FESS("(broker Jane) bought 2 (stock AAPL) at 2:dollars")
         |""".stripMargin
    )

    checkCase(
      "marker tokens inside string are not rewritten",
      """|object X:
         |  val s = "marker @:- keep -:@ string"
         |""".stripMargin,
      """|object X:
         |  val s = "marker @:- keep -:@ string"
         |""".stripMargin
    )

    checkCase(
      "normal scala assignment remains unchanged",
      """|object X:
         |  val n = 1 + 2
         |""".stripMargin,
      """|object X:
         |  val n = 1 + 2
         |""".stripMargin
    )

    checkCase(
      "multiline marker escapes content safely",
      """|object X:
         |  val p = @:-
         |    (broker Mark) said "hello\\path"
         |    then moved
         |  -:@
         |""".stripMargin,
      """|object X:
         |  val p = FESS("(broker Mark) said \"hello\\\\path\"\n    then moved")
         |""".stripMargin
    )

    checkCase(
      "mixed scala and non-delimited DSL stays unchanged",
      """|object X:
         |  val x = 42
         |  val t = (broker Mark) sold 700 (stock MSFT) at 150:dollars
         |  def inc(v: Int): Int = v + 1
         |""".stripMargin,
      """|object X:
         |  val x = 42
         |  val t = (broker Mark) sold 700 (stock MSFT) at 150:dollars
         |  def inc(v: Int): Int = v + 1
         |""".stripMargin
    )

    checkCase(
      "single-line delimited expression rewrites",
      """|object X:
         |  val t = @:- (broker Mark) sold 700 (stock MSFT) at 150:dollars -:@
         |""".stripMargin,
      """|object X:
         |  val t = FESS("(broker Mark) sold 700 (stock MSFT) at 150:dollars")
         |""".stripMargin
    )
  }
}
