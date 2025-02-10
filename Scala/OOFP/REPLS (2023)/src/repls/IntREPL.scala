package repls

import scala.collection.mutable

class IntREPL extends REPLBase {
    // Have a REPL of type Int
    type Base = Int
    override val replName: String = "int-repl"

    var mapOfBinding : Map[String,Int] = Map.empty

    override def readEval(command: String): String = {
        val exp: Seq[String] = SplitExpressionString.splitExpressionString(command)

        if(exp.size > 1 && (exp.head == "@" || exp(1) == "=" )){

            if(exp.head == "@"){
                val rpnExpression: Seq[String] = ReversePolish.shuntingYardAlgorithm(exp.drop(1)) //Convert Infix expression to RPN, works!
                val treeExp = ReversePolish.reversePolishToExpression(rpnExpression)
                var simplifiedExp: Expression = PatternMatch.simplify(treeExp)
                var canBeSimplified = true
                while (canBeSimplified){
                    val moreSimplified = PatternMatch.simplify(simplifiedExp)
                    if(simplifiedExp == moreSimplified) canBeSimplified = false
                    else simplifiedExp = moreSimplified
                }
                val addBrackets = PatternMatch.traverseExpressionAndCheckDistr(simplifiedExp)
                addBrackets.toString
            }
            else {
                val rpnExpression: Seq[String] = ReversePolish.shuntingYardAlgorithm(exp.drop(2)) //Convert Infix expression to RPN, works!
                val newBinding = ReversePolish.reversePolishToInt(rpnExpression)
                val assignedVar = exp.head
                mapOfBinding += (assignedVar->newBinding)
                s"$assignedVar = $newBinding"
            }

        }

        else {
            val rpnExpression: Seq[String] = ReversePolish.shuntingYardAlgorithm(exp) //Convert Infix expression to RPN, works!
            val resultNormal = ReversePolish.reversePolishToInt(rpnExpression)
            resultNormal.toString
        }
    }

    abstract class Expression {
        def eval(bindings: Map[String, Int]): Int
        def describe: String = "To a string " + toString
    }

    case class Negate(arg: Expression) extends Expression {
        override def eval(bindings: Map[String, Int]): Int = -arg.eval(bindings)
    }


    case class Constant(n: Int) extends Expression {
        override def eval(bindings: Map[String, Int]): Int = n
        def value: Int = n
        override def toString: String = n.toString
    }

    case class Operator(lhs: Expression, operatorName: String, rhs: Expression) extends Expression {
        private def operatorByName(opName: String, lhs: Int, rhs: Int) = {
            if (opName == "+") lhs + rhs
            else if (opName == "*") lhs * rhs
            else lhs - rhs
        }
        override def toString: String = lhs.toString +  " " + operatorName + " " + rhs.toString
        override def eval(bindings: Map[String, Int]): Int = {
            val l = lhs.eval(bindings)
            val r = rhs.eval(bindings)
            PatternMatch.operatorByName(l, operatorName, r)
        }
    }

    case class bracketsExp(lhs: Expression, operatorName: String, rhs: Expression) extends Expression {
        override def toString: String = "( " + lhs.toString + " " + operatorName + " " + rhs.toString + " )"

        override def eval(bindings: Map[String, Int]): Int = {
            val l = lhs.eval(bindings)
            val r = rhs.eval(bindings)
            PatternMatch.operatorByName(l, operatorName, r)
        }
    }

    case class Var(s: String) extends Expression {
        override def eval(bindings: Map[String, Int]): Int = bindings(s)
        def value: Int = mapOfBinding(s)
        override def toString: String = s
    }

    object ReversePolish {

        def shuntingYardAlgorithm(exp: Seq[String]): Seq[String] = {
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
                } else if (isNumber(expression)) wholeStack += expression
                else if (mapOfBinding.contains(expression)) wholeStack += Var(expression).value.toString
                else wholeStack += expression
            }
            while (operatorStack.nonEmpty) {
                wholeStack += operatorStack.last
                operatorStack.remove(operatorStack.size - 1)
            }
            wholeStack.toSeq
        }

        def isNumber(s: String): Boolean = {
            if (s.isEmpty) return false
            var startIndex = 0
            if (s(0) == '-') {  //check if the number is negative
                if (s.length == 1) return false
                startIndex = 1
            }
            for (i <- startIndex until s.length) {
                if (!s(i).isDigit) return false
            }
            true
        }

        def reversePolishToExpression(expression: Seq[String]): Expression = {
            val s: mutable.Stack[Expression] = mutable.Stack.empty[Expression]
            for (el <- expression) {
                if (isOperator(el)) {
                    val rhs = s.last
                    s.remove(s.size-1)
                    val lhs = s.last
                    s.remove(s.size-1)
                    val res = Operator(lhs, el, rhs)
                    s += res
                } else if (isNumber(el)) s += Constant(el.toInt) else s += Var(el)
            }
            s.last
        }
        def performOperation(lhs : Int , opNName : String , rhs : Int) : Int = {
            if (opNName == "+") lhs + rhs
            else if (opNName == "*") lhs * rhs
            else lhs - rhs
        }

        def reversePolishToInt(expression: Seq[String]): Int = {
            val s: mutable.Stack[Int] = mutable.Stack.empty[Int]
            for (el <- expression) {
                if (isOperator(el)) {
                    val rhs = s.last
                    s.remove(s.size - 1)
                    val lhs = s.last
                    s.remove(s.size - 1)
                    val res = performOperation(lhs, el, rhs)
                    s += res
                } else if (isNumber(el)) s += el.toInt
            }
            s.last
        }

        private def isOperator(s: String) = s == "+" || s == "*" || s == "-" || s == "(" || s == ")"
    }

    object PatternMatch {

        def traverseExpressionAndCheckDistr(exp: Expression): Expression = {
            exp match {
                case o: Operator =>
                    val lhs = traverseExpressionAndCheckDistr(o.lhs)
                    val rhs = traverseExpressionAndCheckDistr(o.rhs)
                    val operatorName = o.operatorName
                    val checkedExp = checkDistr(Operator(lhs, operatorName, rhs))
                    checkedExp
                case _ => exp
            }
        }


        def checkDistr(exp : Expression) : Expression = {
            exp match {
                case Operator(lhs, "*", rhs) =>
                    rhs match{
                        case Operator(l, "+", r) =>
                            lhs match {
                                case Operator(ll,"+",rr) =>
                                    Operator(bracketsExp(ll,"+", rr),"*",bracketsExp(l,"+",r))
                                case _ => Operator(lhs,"*",bracketsExp(l,"+",r))
                            }
                        case Constant(d) =>
                            lhs match {
                                case Operator(ll, "+", rr) =>
                                    Operator(bracketsExp(ll, "+", rr), "*", Constant(d))
                                case _ => Operator(lhs,"*",rhs)
                            }
                        case _ => Operator(lhs,"*",rhs)
                    }
                case _ => exp
            }
        }

        def operatorByName(l: Int, name: String, r: Int): Int = {
            name match {
                case "+" => l + r
                case "-" => l - r
                case "*" => l * r
            }
        }

        def eval(bindings: Map[String, Int], exp: Expression): Int =
            exp match {
                case Constant(i) => i
                case Var(s) => bindings(s)
                case Negate(arg) => -eval(bindings, arg)
                case Operator(lhs, op, rhs) =>
                    operatorByName(eval(bindings, lhs), op, eval(bindings, rhs))
            }

        def simplify(exp: Expression): Expression = {
            exp match {
                case Operator(l @ Constant(_), op, r @ Constant(_)) => {
                    val simplifiedValue = operatorByName(l.value, op, r.value)
                    Constant(simplifiedValue)
                }
                //check distributivity
                case Operator(Operator(lhs, "*", rhs), "+", Operator(llhs, "*", rrhs)) if lhs == llhs => simplify(Operator(lhs, "*", Operator(rhs, "+", rrhs)))
                case Operator(Operator(lhs, "*", rhs), "+", Operator(llhs, "*", rrhs)) if rhs == llhs => simplify(Operator(rhs, "*", Operator(lhs, "+", rrhs)))
                case Operator(Operator(lhs, "*", rhs), "+", Operator(llhs, "*", rrhs)) if lhs == rrhs => simplify(Operator(lhs, "*", Operator(rhs, "+", llhs)))
                case Operator(Operator(lhs, "*", rhs), "+", Operator(llhs, "*", rrhs)) if rhs == rrhs => simplify(Operator(rhs, "*", Operator(lhs, "+", llhs)))

                case Operator(e, "+", Constant(0)) => simplify(e)
                case Operator(Constant(0), "+", e) => simplify(e)
                case Operator(e, "*", Constant(1)) => simplify(e)
                case Operator(Constant(1), "*", e) => simplify(e)
                case Operator(e, "*", Constant(0)) => Constant(0)
                case Operator(Constant(0), "*", e) => Constant(0)
                case Operator(e, "-", e2) if e == e2 => simplify(Constant(0))
                case Operator(l, op, r) => {
                    val simplifiedLHS = simplify(l)
                    val simplifiedRHS = simplify(r)
                    Operator(simplifiedLHS, op, simplifiedRHS)
                }

                case Negate(Negate(e)) => simplify(e)
                case Negate(e) => Negate(simplify(e))
                case _ => exp
            }
        }
    }
}