package repls

import repls.MultiSet.empty

import scala.collection.mutable

case class MultiSet[T] (multiplicity: Map[T, Int]){

    def isBiggerThanZero(i : Int) : Boolean = i > 0

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
                else None                                // excludes the key-value pair from the newMap
        }
        MultiSet[T](newMap)
    }

    def +(that: MultiSet[T]): MultiSet[T] = {
        val oldMap = multiplicity
        val addMap = that.multiplicity
        println("Doing", oldMap, "+", addMap)
        val newMap = oldMap ++ addMap.map { case (key, value) => key -> (value + oldMap.getOrElse(key, 0)) }   //++ is used to merge 2 maps, map value of key to oldMap if it is already present, else add it to the map with the value of addMap
        MultiSet[T](newMap)
    }

    def -(that: MultiSet[T]): MultiSet[T] = {
        val oldMap = multiplicity
        val subMap = that.multiplicity
        val newMap = oldMap.map {
            case (key, value) =>
                if(subMap.contains(key)) key -> (value - subMap(key))
                else key -> value
        }
        MultiSet[T](newMap)
    }

    def toSeq: Seq[T] = multiplicity.toSeq.flatMap{ case (letter, count) => Seq.fill(count)(letter) }

    val MaxCountForDuplicatePrint = 5

    override def toString: String = {
        def elemToString(elem : T) : String = {
            val count = multiplicity(elem)
            if(count >= MaxCountForDuplicatePrint)
                elem.toString + " -> " + count.toString
            else Seq.fill(count)(elem).mkString(",")
        }
        val keyStringSet = multiplicity.keySet.map(elemToString)
        "{" + keyStringSet.toSeq.sorted.mkString(",") + "}"
    }
}

object MultiSet {

    private def isOperator(s: String) = s == "+" || s == "*" || s == "-" || s == "(" || s == ")"

    def OperatorMult[T](lhs: MultiSet[T], opName: String, rhs: MultiSet[T]): MultiSet[T] = {
        opName match {
            case "*" => lhs * rhs
            case "+" => lhs + rhs
            case "-" => lhs - rhs
        }
    }

    def reversePolishCalc[T] (expression: Seq[String]): MultiSet[T] = {
        val s: mutable.Stack[MultiSet[T]] = mutable.Stack.empty[MultiSet[T]]
        for (el <- expression) {
            if (isOperator(el)) {
                val rhs = s.last
                s.remove(s.size - 1)
                val lhs = s.last
                s.remove(s.size - 1)
                val res = OperatorMult(rhs,el,lhs)
                println("Post", lhs, el, rhs, "=", res)
                s += res
            } else {
                val newMultiSet: MultiSet[T] = MultiSet[T](Seq(el.asInstanceOf[T]))
                s += newMultiSet
            }
        }
        s.last
    }

    def revPolishMultiset (exp: Seq[String]): Seq[String] = {
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
            } else wholeStack += expression
        }
        while (operatorStack.nonEmpty) {
            wholeStack += operatorStack.last
            operatorStack.remove(operatorStack.size - 1)
        }
        wholeStack.toSeq
    }

    def empty[T] : MultiSet[T] = MultiSet(Map[T,Int]())

    def apply[T](elements: Seq[T]): MultiSet[T] = {
        val groupedElements = elements.foldLeft(Map.empty[T, Int].withDefaultValue(0)) {  //from left to right, update the map in each iteration, when element is seen-> increment counter by 1
            (map, element) => map + (element -> (map(element) + 1))
        }
        MultiSet(groupedElements)
    }
}
