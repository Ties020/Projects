package repls

import scala.collection.mutable
class MultiSetREPL extends REPLBase {
    // Have a REPL of a MutliSet of strings
    override type Base = MultiSet[String]
    override val replName: String = "multiset-repl"

    var mapOfBinding : Map[String,Any] = Map.empty    //maybe do string to a int, better for revPolish

    override def readEval(command: String): String = {
        val exp = SplitExpressionString.splitExpressionString(command)

        if(exp.size > 1 && (exp.head == "@" || exp(1) == "=" )){
            if(exp(1) == "="){
                val inRPN = MultiSet.revPolishMultiset(exp.drop(2))
                val newBinding = MultiSet.reversePolishToResMultiset(inRPN)
                val assignedVar = exp.head
                mapOfBinding += (assignedVar -> newBinding)
                s"$assignedVar = $newBinding"
            }
            else {
                val inRPN = MultiSet.revPolishMultiset(exp.drop(1))
                val treeExp = MultiSet.reversePolishToExpression(inRPN)
                var simplifiedExp = MultiSet.simplify(treeExp)
                var canBeSimplified = true
                while (canBeSimplified) {
                    val moreSimplified = MultiSet.simplify(simplifiedExp)
                    if (simplifiedExp == moreSimplified) canBeSimplified = false
                    else simplifiedExp = moreSimplified
                }
                simplifiedExp.toString
            }
        }
        else {
            val inRPN = MultiSet.revPolishMultiset(exp)
            val resultPostOperations = MultiSet.reversePolishToResMultiset(inRPN)
            resultPostOperations.toString
        }

    }

    case class MultiSet[T](multiplicity: Map[T, Int]) {

        def isBiggerThanZero(i: Int): Boolean = i > 0

        def *(that: MultiSet[T]): MultiSet[T] = {
            val oldMap = multiplicity
            val intMap = that.multiplicity
            val newMap = oldMap.flatMap {
                case (key, value) =>
                    if (intMap.contains(key) && value > intMap(key) && isBiggerThanZero(value) && isBiggerThanZero(intMap(key))) {
                        Some(key -> intMap(key))
                    }
                    else if (intMap.contains(key) && value < intMap(key) && isBiggerThanZero(value) && isBiggerThanZero(intMap(key))) {
                        Some(key -> value)
                    }
                    else if (intMap.contains(key) && value == intMap(key) && isBiggerThanZero(value) && isBiggerThanZero(intMap(key))) {
                        Some(key -> value)
                    }
                    else None // excludes the key-value pair from the newMap
            }
            MultiSet[T](newMap)
        }

        def +(that: MultiSet[T]): MultiSet[T] = {
            val oldMap = multiplicity
            val addMap = that.multiplicity
            val newMap = oldMap ++ addMap.map { case (key, value) => key -> (value + oldMap.getOrElse(key, 0)) } //++ is used to merge 2 maps, map value of key to oldMap if it is already present, else add it to the map with the value of addMap
            MultiSet[T](newMap)
        }

        def -(that: MultiSet[T]): MultiSet[T] = {
            val oldMap = multiplicity
            val subMap = that.multiplicity
            val newMap = oldMap.map {
                case (key, value) =>
                    if (subMap.contains(key)) key -> (value - subMap(key))
                    else key -> value
            }
            MultiSet[T](newMap)
        }

        def toSeq: Seq[T] = multiplicity.toSeq.flatMap { case (letter, count) => Seq.fill(count)(letter) }

        val MaxCountForDuplicatePrint = 5

        override def toString: String = {
            def elemToString(elem: T): String = {
                val count = multiplicity(elem)
                if (count >= MaxCountForDuplicatePrint)
                    elem.toString + " -> " + count.toString
                else Seq.fill(count)(elem).mkString(",")
            }

            val keyStringSet = multiplicity.keySet.map(elemToString)
            "{" + keyStringSet.toSeq.sorted.mkString(",") + "}"
        }
    }

    abstract class Expression {
        def describe: String = "To a string " + toString
    }

    case class Constant(n: MultiSet[Any]) extends Expression {
        def value: MultiSet[Any] = n
        override def toString: String = n.toString
    }

    case class Operator(lhs: Expression, operatorName: String, rhs: Expression) extends Expression {
        private def operatorByName(opName: String, lhs: Int, rhs: Int) = {
            if (opName == "+") lhs + rhs
            else if (opName == "*") lhs * rhs
            else lhs - rhs
        }
        override def toString: String = lhs.toString + " " + operatorName + " " + rhs.toString
    }

    case class multisetVar(str: String) extends Expression {
        def value: Any = mapOfBinding(str)

        override def toString: String = str
    }

    object MultiSet {
        def performOperation[T](lhs: MultiSet[T], opName: String, rhs: MultiSet[T]): MultiSet[T] = {
            opName match {
                case "*" => lhs * rhs
                case "+" => lhs + rhs
                case "-" => lhs - rhs
            }
        }

        def reversePolishToResMultiset[T](expression: Seq[String]): MultiSet[T] = {
            val s: mutable.Stack[MultiSet[T]] = mutable.Stack.empty[MultiSet[T]]  //only push Multisets on here, if operator is encounter from rpn exp-> do operation
            for (el <- expression) {
                if (isOperator(el)) {
                    val rhs = s.last
                    s.remove(s.size - 1)
                    val lhs = s.last
                    s.remove(s.size - 1)
                    val res = performOperation(lhs, el, rhs)
                    val removedNonsenseValues = MultiSet.apply(res.toSeq)
                    s += removedNonsenseValues
                }
                else {        //transform string to MultiSet
                    if(el == "{}") {       //needed to handle empty multisets
                        val newMultiSet = fromSeq[T](Seq.empty)
                        s+= newMultiSet
                    }
                    else {
                        val elements = el.filterNot(c => c == '{' || c == '}').split(",").map(_.trim) // filter out { and } characters
                        val newMultiSet = fromSeq[T](elements.toSeq)
                        s += newMultiSet
                    }
                }
            }
            s.last
        }


        def simplify(exp: Expression): Expression = {
            exp match {
                case Operator(lhs, "*", rhs)  if (lhs == rhs && !isMultiset(rhs.toString)) => multisetVar(lhs.toString)
                case Operator(lhs, "-",rhs) => Constant(MultiSet(Seq.empty)) //this is for subtracting sequences by itself for example: (a+b)-(a+b), where a and b are vars
                case Operator(l@multisetVar(a), "*", r@multisetVar(b)) if a != b => Constant(MultiSet(Seq.empty))
                case Operator(l@Constant(_), op, r@Constant(_)) => {
                    val simplifiedValue = performOperation(l.value, op, r.value)
                    Constant(simplifiedValue)
                }

                case Operator(e, "+", Constant(o)) if (o == MultiSet(Seq.empty)) => simplify(e)
                case Operator(Constant(o), "+", e) if (o == MultiSet(Seq.empty)) => simplify(e)
                case Operator(e, "*", Constant(o)) if (o == MultiSet(Seq.empty)) => Constant(o)
                case Operator(Constant(o), "*", e) if (o == MultiSet(Seq.empty)) => Constant(o)
                case Operator(e, "-", e2) if e == e2 => Constant(MultiSet(Seq.empty))
                case Operator(multisetVar(o), "*", multisetVar(p)) if (o == p) => multisetVar(o)
                case Operator(l, op, r) => {
                    val simplifiedLHS = simplify(l)
                    val simplifiedRHS = simplify(r)
                    Operator(simplifiedLHS, op, simplifiedRHS)
                }
                case _ => exp
            }
        }

        def reversePolishToExpression[T](expression: Seq[String]): Expression = {
            val s: mutable.Stack[Expression] = mutable.Stack.empty[Expression]
            for (el <- expression) {
                if (isOperator(el)) {
                    val rhs = s.last
                    s.remove(s.size - 1)
                    val lhs = s.last
                    s.remove(s.size - 1)
                    val res = Operator(lhs, el, rhs)
                    s += res
                }
                else if (el == "{}") { //needed to handle empty multisets
                    val newMultiSet = fromSeq[Any](Seq.empty)
                    s += Constant(newMultiSet)
                }
                else if (isMultiset(el)) {
                    val elements = el.filterNot(c => c == '{' || c == '}').split(",").map(_.trim) // filter out { and } characters
                    val newMultiSet = fromSeq[Any](elements.toSeq)
                    s += Constant(newMultiSet)
                }
                else s += multisetVar(el)
            }
            s.last
        }

        def isMultiset(str: String) : Boolean = str.contains("{")

        def fromSeq[T](elements: Seq[String]): MultiSet[T] = {    //used to convert elements to type T to allow for multiple types to be used in Multiset
            val convertedElements = elements.map { elem => elem.asInstanceOf[T] }
            apply(convertedElements)
        }

        private def isOperator(s: String) = s == "+" || s == "*" || s == "-" || s == "(" || s == ")"

        def revPolishMultiset(exp: Seq[String]): Seq[String] = {
            val operatorStack = mutable.Stack.empty[String]
            val wholeStack = mutable.Stack.empty[String]
            for (expression <- exp) {
                if (isOperator(expression)) {
                    if (operatorStack.isEmpty) {
                        operatorStack += expression
                    } else {
                        expression match {
                            case "(" =>
                                operatorStack += expression
                            case ")" =>
                                while (operatorStack.nonEmpty && operatorStack.last != "(") {
                                    wholeStack += operatorStack.last
                                    operatorStack.remove(operatorStack.size - 1)
                                }
                                operatorStack.remove(operatorStack.size - 1) // Pop ( and discard
                            case "+" | "-" =>
                                while (operatorStack.nonEmpty && (operatorStack.last == "*" || operatorStack.last == "+" || operatorStack.last == "-")) {
                                    wholeStack += operatorStack.last
                                    operatorStack.remove(operatorStack.size - 1)
                                }
                                operatorStack += expression
                            case "*" =>
                                while (operatorStack.nonEmpty && operatorStack.last == "*") {
                                    wholeStack += operatorStack.last
                                    operatorStack.remove(operatorStack.size - 1)
                                }
                                operatorStack += expression
                        }
                    }
                }
                else if (mapOfBinding.contains(expression)){
                    wholeStack += mapOfBinding(expression).toString
                }
                else wholeStack += expression

            }
            while (operatorStack.nonEmpty) {
                wholeStack += operatorStack.last
                operatorStack.remove(operatorStack.size - 1)
            }
            wholeStack.toSeq
        }


        def empty[T]: MultiSet[T] = MultiSet(Map[T, Int]())

        def apply[T](elements: Seq[T]): MultiSet[T] = {
            val groupedElements = elements.foldLeft(Map.empty[T, Int].withDefaultValue(0)) { //from left to right, update the map in each iteration, when element is seen-> increment counter by 1
                (map, element) => map + (element -> (map(element) + 1))
            }
            MultiSet(groupedElements)
        }
    }
}
