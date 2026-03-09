package sentences

import profess.runtime.*

// @profess
// @profess-profile: profess.profile
object SentencePlayground:

  // Raw PROFESS sentences; preprocessor rewrites these to FESS("...").
  val sentence: ProfessExpr =
    (broker Mark) sold 700 (stock MSFT) at 150:dollars
  val sentenceWithString: ProfessExpr =
    @:- (broker Mark) said "hello world" -:@
  val sentenceWithNegative: ProfessExpr =
    (broker Mark) sold 700 (stock MSFT) at -150.5:dollars
  val sentenceWithBlockMarkers: ProfessExpr =
    @:-
      (unit 3rd_platoon)
      deployed to (location firebase_alpha)
      with (equipment armored_truck)
    -:@


@main def runSentencesPlayground(): Unit =
  println("Running PROFESS sentence playground")
  println(s"Sentence IR: ${SentencePlayground.sentence.toIR.render}")
  println(s"Sentence With String IR: ${SentencePlayground.sentenceWithString.toIR.render}")
  println(s"Sentence With Negative IR: ${SentencePlayground.sentenceWithNegative.toIR.render}")
  println(s"Sentence With Block Markers IR: ${SentencePlayground.sentenceWithBlockMarkers.toIR.render}")
