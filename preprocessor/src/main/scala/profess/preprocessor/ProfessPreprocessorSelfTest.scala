package profess.preprocessor

object ProfessPreprocessorSelfTest {
  final case class TestCase(name: String, input: String, expected: String)

  val cases: List[TestCase] = List(
    TestCase(
      "unclosed marker is left unchanged",
      """|object X:
         |  val trade = @:- (broker Mark) sold 700 (stock MSFT) at 150:dollars
         |""".stripMargin,
      """|object X:
         |  val trade = @:- (broker Mark) sold 700 (stock MSFT) at 150:dollars
         |""".stripMargin
    ),
    TestCase(
      "stray end marker is left unchanged",
      """|object X:
         |  val trade = (broker Mark) sold 700 (stock MSFT) at 150:dollars -:@
         |""".stripMargin,
      """|object X:
         |  val trade = (broker Mark) sold 700 (stock MSFT) at 150:dollars -:@
         |""".stripMargin
    ),
    TestCase(
      "adjacent marker blocks rewrite independently",
      """|object X:
         |  val a = @:- (broker Mark) sold 1 (stock MSFT) at 1:dollars -:@
         |  val b = @:- (broker Jane) bought 2 (stock AAPL) at 2:dollars -:@
         |""".stripMargin,
      """|object X:
         |  val a = FESS("(broker Mark) sold 1 (stock MSFT) at 1:dollars")
         |  val b = FESS("(broker Jane) bought 2 (stock AAPL) at 2:dollars")
         |""".stripMargin
    ),
    TestCase(
      "marker tokens inside string are not rewritten",
      """|object X:
         |  val s = "marker @:- keep -:@ string"
         |""".stripMargin,
      """|object X:
         |  val s = "marker @:- keep -:@ string"
         |""".stripMargin
    ),
    TestCase(
      "normal scala assignment remains unchanged",
      """|object X:
         |  val n = 1 + 2
         |""".stripMargin,
      """|object X:
         |  val n = 1 + 2
         |""".stripMargin
    ),
    TestCase(
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
    ),
    TestCase(
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
    ),
    TestCase(
      "single-line delimited expression rewrites",
      """|object X:
         |  val t = @:- (broker Mark) sold 700 (stock MSFT) at 150:dollars -:@
         |""".stripMargin,
      """|object X:
         |  val t = FESS("(broker Mark) sold 700 (stock MSFT) at 150:dollars")
         |""".stripMargin
    )
  )

  def renderFailure(testCase: TestCase, actual: String): String =
    s"""Preprocessor case failed: ${testCase.name}
       |--- Input ---
       |${testCase.input}
       |--- Expected ---
       |${testCase.expected}
       |--- Actual ---
       |$actual
       |""".stripMargin

  def run(log: String => Unit): Unit = {
    cases.foreach { testCase =>
      val actual = ProfessPreprocessorSupport.preprocessProfessSource(testCase.input)
      if (actual != testCase.expected) {
        throw new IllegalStateException(renderFailure(testCase, actual))
      } else {
        log(s"PASS: ${testCase.name}")
      }
    }
  }
}
