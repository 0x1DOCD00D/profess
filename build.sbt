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
val pureConfig = "com.github.pureconfig" %% "pureconfig-core" % pureConfigVersion
val pureConfGeneric = "com.github.pureconfig" %% "pureconfig-generic-scala3" % pureConfigVersion
val catsEffectsTest = "org.typelevel" %% "cats-effect-testing-scalatest" % catsEffectsTestingVersion
val log4CatsCore = "org.typelevel" %% "log4cats-core" % log4catsVersion
val log4CatsSfl = "org.typelevel" %% "log4cats-slf4j" % log4catsVersion
val log4CatsTest = "org.typelevel" %% "log4cats-testing" % log4catsTestingVersion % Test


// ─────────────────────────────────────────────────────────────────────────────
// Global Settings
// ─────────────────────────────────────────────────────────────────────────────

ThisBuild / organization := "com.profess"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := scala3Version

// Publishing metadata
ThisBuild / homepage := Some(url("https://github.com/0x1DOCD00D/profess"))
ThisBuild / licenses := List("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))
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
// Runtime Module
// Core runtime library with IR types, expressions, and Cats Effect integration
// ─────────────────────────────────────────────────────────────────────────────

lazy val runtime = project
  .in(file("runtime"))
  .settings(commonSettings)
  .settings(
    name := "profess-runtime"
  )

// ─────────────────────────────────────────────────────────────────────────────
// Examples Module
// Usage examples demonstrating PROFESS features
// ─────────────────────────────────────────────────────────────────────────────

lazy val examples = project
  .in(file("examples"))
  .dependsOn(runtime)
  .settings(commonSettings)
  .settings(
    name := "profess-examples",
    // Don't publish examples
    publish / skip := true,
    // Enable the PROFESS compiler plugin
    scalacOptions ++= Seq(
      s"-Xplugin:${(plugin / Compile / packageBin).value.getAbsolutePath}"
    ),
    // Ensure plugin is compiled before examples
    Compile / compile := ((Compile / compile) dependsOn (plugin / Compile / packageBin)).value
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
  .aggregate(plugin, runtime, examples)
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
