package profess.runtime

object IRDebug:
  def renderTree(node: IRNode, indent: String = ""): String =
    node match
      case IRObject(kind, name) =>
        s"${indent}IRObject(kind=$kind, name=$name)"
      case IRWord(word) =>
        s"${indent}IRWord(word=$word)"
      case IRNumber(value) =>
        s"${indent}IRNumber(value=$value)"
      case IRString(value) =>
        s"""${indent}IRString(value="$value")"""
      case IRUnitValue(value, unit) =>
        s"${indent}IRUnitValue(value=$value, unit=$unit)"
      case IRBoolean(value) =>
        s"${indent}IRBoolean(value=$value)"
      case IRReference(name) =>
        s"${indent}IRReference(name=$name)"
      case IRBinding(name, value) =>
        s"""${indent}IRBinding(name=$name)
${renderTree(value, indent + "  ")}"""
      case IRTuple(elements) =>
        val rendered =
          elements.map(renderTree(_, indent + "  ")).mkString("\n")
        s"""${indent}IRTuple(
$rendered
${indent})"""
      case IRSequence(nodes) =>
        val rendered =
          nodes.map(renderTree(_, indent + "  ")).mkString("\n")
        s"""${indent}IRSequence(
$rendered
${indent})"""
      case IRConditional(condition, consequent, alternative) =>
        val elsePart = alternative match
          case Some(alt) => s"\n${indent}  else =\n${renderTree(alt, indent + "    ")}"
          case None => ""
        s"""${indent}IRConditional(
${indent}  condition =
${renderTree(condition, indent + "    ")}
${indent}  then =
${renderTree(consequent, indent + "    ")}$elsePart
${indent})"""
      case IRParamBlock(params) =>
        val rendered =
          params.map { case (name, value) =>
            val label = name.getOrElse("<positional>")
            s"${indent}  $label =\n${renderTree(value, indent + "    ")}"
          }.mkString("\n")
        s"""${indent}IRParamBlock(
$rendered
${indent})"""
      case IRAttributes(target, attrs) =>
        s"""${indent}IRAttributes(attrs=${attrs.mkString("[", ", ", "]")})
${renderTree(target, indent + "  ")}"""
