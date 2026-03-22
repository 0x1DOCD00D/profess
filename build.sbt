import scala.sys.process.*

// ═══════════════════════════════════════════════════════════════════════════
// PROFESS - Programming Rule Oriented Formalized English Sentence Specifications
// A Scala 3 Framework for Domain-Specific Languages with Formal Semantics
// Copyright © 2025-12-15 21:14:28 CST Lone Star Consulting, Inc. and Mark Grechanik. All rights reserved.
// ═══════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Versions (Updated December 2025)
// ─────────────────────────────────────────────────────────────────────────────

val scala3Version = "3.7.4" // Update to your preferred version
val logbackVersion = "1.5.18"
val typeSafeConfigVersion = "1.4.4"
val sfl4sVersion = "2.1.0-alpha1"
val typesafeConfigVersion = "1.4.4"
val apacheCommonIOVersion = "2.20.0"
val scalacticVersion = "3.2.19"
val nscalatimeVersion = "3.0.0"
val apacheCommonMathVersion = "3.6.1"
val guavaVersion = "33.4.8-jre"
val catsVersion = "2.13.0"
val catsEffectVersion = "3.6.3"
val snakeYamlVersion = "2.4"
val xmlVersion = "2.4.0"
val scalaReflectVersion = "2.13.16"
val scalaCompilerVersion = "2.13.16"
val ollama4jVersion = "1.0.73"
val scalaMetaVersion = "4.13.8"
val record4sVersion = "0.13.0"
val okHttpVersion = "5.1.0"
val uPickleVersion = "4.2.1"
val nameofVersion = "5.0.0"
val commonsRngSimpleVersion = "1.6"
val doobieVersion = "1.0.0-RC8"
val clickhouseJdbcVersion = "0.9.2"
val http4sVersion = "0.23.32"
val circeVersion = "0.14.15"
val log4catsVersion = "2.7.1"
val log4catsTestingVersion = "2.7.1"
val neo4jVersion = "6.8.0"
val anthropicVersion = "2.8.1"
val blackNiniaJepVersion = "4.2.2"
val pureConfigVersion = "0.17.9"
val nameOfVersion = "4.0.0"
val catsEffectsTestingVersion = "1.7.0"
val fs2Version = "3.11.0"
val declineVersion = "2.4.1"
val sttpVersion = "3.9.8"
val tapirVersion = "1.11.11"
val catsSTMVersion = "0.13.4"
val osLibVersion = "0.11.4"
val scalaParserCombinatorsVersion = "2.4.0"

// ─────────────────────────────────────────────────────────────────────────────
// Dependencies
// ─────────────────────────────────────────────────────────────────────────────

val commonId = "commons-io" % "commons-io" % apacheCommonIOVersion
val logbackCore = "ch.qos.logback" % "logback-core" % logbackVersion
val logbackClassic = "ch.qos.logback" % "logback-classic" % logbackVersion
val sfl4jApt = "org.slf4j" % "slf4j-api" % sfl4sVersion
val catsCore = "org.typelevel" %% "cats-core" % catsVersion
val catsEffect = "org.typelevel" %% "cats-effect" % catsEffectVersion
val catsLaws = "org.typelevel" %% "cats-laws" % catsVersion % Test
val catsEffects = "org.typelevel" %% "cats-effect" % catsEffectVersion
val scalactic = "org.scalactic" %% "scalactic" % scalacticVersion
val scalatest = "org.scalatest" %% "scalatest" % scalacticVersion % Test
val pureConfig =
  "com.github.pureconfig" %% "pureconfig-core" % pureConfigVersion
val pureConfGeneric =
  "com.github.pureconfig" %% "pureconfig-generic-scala3" % pureConfigVersion
val catsEffectsTest =
  "org.typelevel" %% "cats-effect-testing-scalatest" % catsEffectsTestingVersion
val log4CatsCore = "org.typelevel" %% "log4cats-core" % log4catsVersion
val log4CatsSfl = "org.typelevel" %% "log4cats-slf4j" % log4catsVersion
val log4CatsTest =
  "org.typelevel" %% "log4cats-testing" % log4catsTestingVersion % Test
val scalaParserCombinators =
  "org.scala-lang.modules" %% "scala-parser-combinators" % scalaParserCombinatorsVersion

// ─────────────────────────────────────────────────────────────────────────────
// Global Settings
// ─────────────────────────────────────────────────────────────────────────────

ThisBuild / organization := "com.profess"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := scala3Version

// Publishing metadata
ThisBuild / homepage := Some(url("https://github.com/0x1DOCD00D/profess"))
ThisBuild / licenses := List(
  "Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0")
)
ThisBuild / developers := List(
  Developer(
    id = "0x1DOCD00D",
    name = "Mark Grechanik",
    email = "0x1DOCD00D@drmark.tech",
    url = url("https://github.com/0x1DOCD00D")
  )
)
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/0x1DOCD00D/profess"),
    "git@github.com:0x1DOCD00D/profess.git"
  )
)

// ─────────────────────────────────────────────────────────────────────────────
// Common Settings
// ─────────────────────────────────────────────────────────────────────────────

lazy val commonSettings = Seq(
  scalacOptions ++= Seq(
    "-deprecation",
    "-feature",
    "-unchecked",
    "-language:dynamics",
    "-language:higherKinds",
    "-language:implicitConversions"
  ),
  libraryDependencies ++= Seq(
    commonId,
    logbackCore,
    logbackClassic,
    sfl4jApt,
    catsCore,
    catsEffect,
    catsLaws,
    catsEffects,
    scalactic,
    scalatest,
    pureConfig,
    pureConfGeneric,
    catsEffectsTest,
    log4CatsCore,
    log4CatsSfl,
    log4CatsTest
  ),
  // Test settings
  Test / parallelExecution := false,
  Test / fork := true,
  testFrameworks += new TestFramework("munit.Framework")
)

lazy val preprocessorSelfTest =
  taskKey[Unit]("Run edge-case checks for PROFESS source preprocessing.")

// Use the in-repo compiler plugin jar directly instead of external resolution.
def inRepoProfessPluginOptions: Seq[Def.Setting[?]] = Seq(
  Compile / scalacOptions += {
    val pluginJar = (plugin / Compile / packageBin).value
    s"-Xplugin:${pluginJar.getAbsolutePath}"
  },
  // Fail fast if the plugin could not be loaded by scalac.
  Compile / scalacOptions += "-Xplugin-require:profess"
)

def hasProfessDelimiter(content: String): Boolean = {
  val marker = "@:-"
  var i = 0
  var inString = false
  var escaped = false

  while (i < content.length) {
    val ch = content.charAt(i)
    if (inString) {
      if (escaped) escaped = false
      else if (ch == '\\') escaped = true
      else if (ch == '"') inString = false
      i += 1
    } else if (ch == '"') {
      inString = true
      i += 1
    } else if (content.startsWith(marker, i)) {
      return true
    } else {
      i += 1
    }
  }

  false
}

// ─────────────────────────────────────────────────────────────────────────────
// Plugin Module
// Scala 3 compiler plugin for PROFESS syntax scaffolding
// ─────────────────────────────────────────────────────────────────────────────

lazy val plugin = project
  .in(file("plugin"))
  .settings(commonSettings)
  .settings(
    name := "profess-plugin",
    libraryDependencies ++= Seq(
      "org.scala-lang" %% "scala3-compiler" % scalaVersion.value
    ),
    // Mark as compiler plugin
    Compile / packageBin / packageOptions +=
      Package.ManifestAttributes("Scala-Compiler-Plugin" -> "true")
  )

// ─────────────────────────────────────────────────────────────────────────────
// Preprocessor Module
// Dedicated tool that rewrites @:- ... -:@ blocks to FESS("...")
// ─────────────────────────────────────────────────────────────────────────────

lazy val preprocessor = project
  .in(file("preprocessor"))
  .settings(
    name := "profess-preprocessor",
    scalaVersion := scala3Version,
    publish / skip := true
  )

// ─────────────────────────────────────────────────────────────────────────────
// Runtime Module
// Core runtime library with IR types, expressions, and Cats Effect integration
// ─────────────────────────────────────────────────────────────────────────────

lazy val runtime = project
  .in(file("runtime"))
  .settings(commonSettings)
  .settings(
    name := "profess-runtime",
    libraryDependencies += scalaParserCombinators
  )

// ─────────────────────────────────────────────────────────────────────────────
// Examples Module
// Usage examples demonstrating PROFESS features
// ─────────────────────────────────────────────────────────────────────────────

lazy val examples = project
  .in(file("examples"))
  .dependsOn(runtime)
  .settings(commonSettings)
  .settings(inRepoProfessPluginOptions)
  .settings(
    name := "profess-examples",
    // Don't publish examples
    publish / skip := true,
    // Uncomment to see debug output (per-unit phase echo) or comment to remove debug flag:
//    scalacOptions += "-P:profess:debug",
    Compile / compile := (Compile / compile)
      .dependsOn(plugin / Compile / packageBin)
      .value
  )

// ─────────────────────────────────────────────────────────────────────────────
// Sentences Module
// Playground for writing English-like PROFESS sentences and running them
// ─────────────────────────────────────────────────────────────────────────────

lazy val sentences = project
  .in(file("sentences"))
  .dependsOn(runtime)
  .settings(commonSettings)
  .settings(inRepoProfessPluginOptions)
  .settings(
    name := "profess-sentences",
    publish / skip := true,
    Compile / unmanagedSources := {
      val sources = (Compile / unmanagedSources).value
      sources.filterNot { file =>
        file.ext == "scala" && hasProfessDelimiter(IO.read(file))
      }
    },
    Compile / sourceGenerators += Def.task {
      val log = streams.value.log
      val srcDir = (Compile / scalaSource).value
      val outDir = (Compile / sourceManaged).value / "profess"
      val preprocessorCp = (preprocessor / Compile / fullClasspath).value.files
        .map(_.getAbsolutePath)
        .mkString(java.io.File.pathSeparator)
      val rawSources = (srcDir ** "*.scala").get
      val enabledSources =
        rawSources.filter(f => hasProfessDelimiter(IO.read(f)))

      enabledSources.map { inFile =>
        val relPath = inFile.relativeTo(srcDir).get.getPath
        val outFile = outDir / relPath
        IO.createDirectory(outFile.getParentFile)
        val cmd = Seq(
          "java",
          "-cp",
          preprocessorCp,
          "profess.preprocessor.ProfessPreprocessorCli",
          inFile.getAbsolutePath,
          outFile.getAbsolutePath
        )
        val exitCode = Process(cmd, baseDirectory.value).!
        if (exitCode != 0) {
          sys.error(s"Preprocessor CLI failed for ${inFile.getAbsolutePath}")
        } else {
          log.debug(s"Preprocessed ${inFile.getName} -> ${outFile.getAbsolutePath}")
        }
        outFile
      }
    }.taskValue,
    preprocessorSelfTest := {
      val preprocessorCp = (preprocessor / Compile / fullClasspath).value.files
        .map(_.getAbsolutePath)
        .mkString(java.io.File.pathSeparator)
      val cmd = Seq(
        "java",
        "-cp",
        preprocessorCp,
        "profess.preprocessor.ProfessPreprocessorCli",
        "--self-test"
      )
      val exitCode = Process(cmd, baseDirectory.value).!
      if (exitCode != 0) sys.error("Preprocessor self-test failed")
    },
    Compile / compile := (Compile / compile)
      .dependsOn(plugin / Compile / packageBin)
      .value
  )

// ─────────────────────────────────────────────────────────────────────────────
// Documentation Module
// For generating documentation with mdoc
// Requires sbt-mdoc plugin in project/plugins.sbt
// ─────────────────────────────────────────────────────────────────────────────

lazy val docs = project
  .in(file("docs-build"))
  .dependsOn(runtime)
  .enablePlugins(MdocPlugin)
  .settings(
    name := "profess-docs",
    publish / skip := true,
    mdocIn := file("docs"),
    mdocOut := file("target/docs")
  )

// ─────────────────────────────────────────────────────────────────────────────
// Root Project
// Aggregates all modules
// ─────────────────────────────────────────────────────────────────────────────

lazy val root = project
  .in(file("."))
  .aggregate(plugin, preprocessor, runtime, examples, sentences)
  .settings(
    name := "profess",
    // Don't publish the root project
    publish / skip := true
  )

// ─────────────────────────────────────────────────────────────────────────────
// Command Aliases
// ─────────────────────────────────────────────────────────────────────────────

addCommandAlias("fmt", "scalafmtAll")
addCommandAlias("fmtCheck", "scalafmtCheckAll")
addCommandAlias("build", "compile; test")
addCommandAlias("runExamples", "examples/run")
addCommandAlias("runSentences", "sentences/run")
