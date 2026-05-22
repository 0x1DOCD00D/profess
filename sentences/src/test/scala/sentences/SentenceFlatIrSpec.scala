package sentences

import org.scalatest.funsuite.AnyFunSuite
import profess.runtime.*

class SentenceFlatIrSpec extends AnyFunSuite:

  test("trade sentence lowers to flat IR sequence") {
    assert(
      SentencePlayground.sentence.toIR ==
        IRSequence(
          List(
            IRObject("broker", "Mark"),
            IRWord("sold"),
            IRNumber(700.0),
            IRObject("stock", "MSFT"),
            IRWord("at"),
            IRUnitValue(150.0, "dollars")
          )
        )
    )
  }

  test("quoted string sentence lowers to flat IR sequence") {
    assert(
      SentencePlayground.sentenceWithString.toIR ==
        IRSequence(
          List(
            IRObject("broker", "Mark"),
            IRWord("said"),
            IRString("hello world")
          )
        )
    )
  }

  test("negative unit sentence lowers to flat IR sequence") {
    assert(
      SentencePlayground.sentenceWithNegative.toIR ==
        IRSequence(
          List(
            IRObject("broker", "Mark"),
            IRWord("sold"),
            IRNumber(700.0),
            IRObject("stock", "MSFT"),
            IRWord("at"),
            IRUnitValue(-150.5, "dollars")
          )
        )
    )
  }

  test("multiline sentence lowers to flat IR sequence") {
    assert(
      SentencePlayground.sentenceWithBlockMarkers.toIR ==
        IRSequence(
          List(
            IRObject("unit", "3rd_platoon"),
            IRWord("deployed"),
            IRWord("to"),
            IRObject("location", "firebase_alpha"),
            IRWord("with"),
            IRObject("equipment", "armored_truck")
          )
        )
    )
  }

  test("tree renderer exposes structural IR, not just source-like render") {
    val rendered = IRDebug.renderTree(SentencePlayground.sentence.toIR)

    assert(rendered.contains("IRSequence("))
    assert(rendered.contains("IRObject(kind=broker, name=Mark)"))
    assert(rendered.contains("IRWord(word=sold)"))
    assert(rendered.contains("IRUnitValue(value=150.0, unit=dollars)"))
  }
