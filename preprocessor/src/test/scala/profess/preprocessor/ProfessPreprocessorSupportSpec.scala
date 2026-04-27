package profess.preprocessor

import org.scalatest.funsuite.AnyFunSuite

final class ProfessPreprocessorSupportSpec extends AnyFunSuite {
  ProfessPreprocessorSelfTest.cases.foreach { testCase =>
    test(testCase.name) {
      val actual = ProfessPreprocessorSupport.preprocessProfessSource(testCase.input)
      withClue(ProfessPreprocessorSelfTest.renderFailure(testCase, actual)) {
        assert(actual == testCase.expected)
      }
    }
  }
}
