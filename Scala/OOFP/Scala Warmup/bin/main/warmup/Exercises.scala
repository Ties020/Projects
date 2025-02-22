package warmup


import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer, Map}
import scala.util.control.Breaks.break

object Exercises {


  /* assignment 1:

      calculate the _index_ of the first of occurrence of the maximum of an array, give -1 if the array is empty
      (there is no index of the maximum)

      implement this using a for loop,
      do not use library functions such as max and indexOf
      (we will subtract the points for this assignment if you do)

      examples:
        indexOfMax(Array(6,5,4,3,6)) == 0
        indexOfMax(Array(4,5,6,3,6)) == 2
        indexOfMax(Array()) == -1

  Indication of solution function body length : 8 lines

   */
  def indexOfMax(a : Array[Int]) : Int =
  {
    val maxIndex: Int = if (a.length <=0 ) -1 else
    {
      var currMax : Int = a(0)
      var maxInd : Int = 0
      for (currIndex <- 0 until a.length)
      {
        if (a(currIndex) > currMax)
        {
          currMax = a(currIndex)
          maxInd = currIndex
        }
      }
      return maxInd
    }
    return maxIndex
  }


  /* Assignment 2:

  calculate the average exam score from a string containing a list of grades. A grade is either a number in the 1-10
  range or the string "NS" for no show. No show grades have no effect on the average grade.

  examples:

  averageGrade("NS NS 9 NS NS 5") == 7
  averageGrade("8 8.2 NS NS 9 4 1 5.5") == 5.95

  Use string.split(" ") to split a string in to an array of strings using " " as a separator
  "a bb cc".split(" ") -> Array("a", "bb", "cc")

  Indication of solution length : 9 lines
   */
  def averageGrade(grades : String) : Double = {
    var stringArray = Array[String]()
    stringArray = grades.split(" ")
    var sum : Double = 0
    var numElements = 0
    for(currIndex <- 0 until  stringArray.length)
    {
      if (stringArray(currIndex)!= "NS")
      {
        sum += stringArray(currIndex).toDouble
        numElements +=1
      }
    }
    return sum / numElements
  }

    /* Assignment 3:

      An infamous unsolved problem in discrete math is the
      "Collatz conjecture" (https://en.wikipedia.org/wiki/Collatz_conjecture).


      For any positive integer n, the corresponding "Collatz sequence" starts with the number n.
      The next number in the sequence is defined as follows:
          if n is even, the next number is n/2
          if n is odd,  the next number is 3n + 1
      The subsequencent numbers are constructed in the same way.
      For example, if we start with 11, the corresponding Collatz sequence is:
      11,34,17,52,26,13,40,20,10,5,16,8,4,2,1,4,2,1,4,...
      Once a Collatz sequence has reached 1, the sequence will repeat the pattern 4,2,1 forever.

      The Collatz conjecture states that, no matter which integer we start with, the collatz sequence will
      eventually reach 1 and repeat the pattern 4,2,1 forever. The conjecture is probably true, and has been checked
      by computers for all starting values up to 87×2^60(!). However, there is no mathematical proof for seemingly
      simple statement. There have been  mathematicians that have spent years of continued study on the conjecture,
      without success.Mathematical professor Jeffrey Lagarias has written a book about Collatz conjecture, and stated
      that "this is an extraordinarily difficult problem, completely out of reach of present day mathematics".

      Luckily, programming the Collatz sequence is much easier! Please implement a function which when given a starting
       number, returns the number of elements in the sequence before reaching 1. For example:
       collatzLength(11) = 14
       collatzLength(1) = 0
       collatzLength(4) = 2
       collatzLength(7) = 16
       collatzLength(27) = 111

    Indication of solution length : 9 lines
     */



  def collatzLength(start : Long) : Long =
  {
    var numElements : Long = 0
    var currNum : Long = start
    while (currNum != 1)
    {
      numElements +=1
      if (currNum % 2 == 0)
      {
        currNum = currNum/2
      }
      else
      {
        currNum = currNum*3 + 1
      }
    }
    return numElements
  }


  /* Assignment 4: Write a function that produces "ASCII art" of a diamond of height n such the following diamond
   of height 7 :

   #
  ###
 #####
#######
 #####
  ###
   #


Diamond of height 12

     #
    ###
   #####
  #######
 #########
###########
###########
 #########
  #######
   #####
    ###
     #

 (You can use multiplication on strings: "ab" * 3 -> "ababab")

Note that the last character of each line should be # and that hence the line should not end in spaces.

Indication of length : 12 lines
   */


  def diamondString(height : Int) : String = {
    var diamond : String = ""
    var topPart : Int = math.ceil(height.toDouble/2.0).toInt
    var bottomPart: Int = math.floor(height.toDouble/2.0).toInt
    var counter : Int = 0
    var numTags : Int = 1
    var spaces : Int = 0
    if (height % 2 == 0) spaces = topPart - 1 else spaces = bottomPart
    while (counter < topPart)           //first build the part of the diamond down to the middle
    {
      diamond += " " * spaces
      diamond += "#" * numTags
      diamond += "\n"
      counter +=1
      spaces -=1
      numTags +=2
    }
    counter = 0
    if (height % 2 == 0) numTags -= 2 else numTags-= 4    //use this to check if the height of the diamond is even, if it is:
    spaces = topPart - bottomPart                       //the first line that will be created will be the same as the last one of the top/first part
    while (counter < bottomPart)
    {
      diamond += " " * spaces
      diamond += "#" * numTags
      diamond += "\n"
      counter += 1
      spaces +=1
      numTags -= 2
    }
    return diamond
  }


  /* Assignment 5: Implement a function word count that gives the word counts for all words in a string.
  Ignore whitespace and capitalization.

   Examples:
    "the monkey hugs the monkey" gives Map(the -> 2, monkey -> 2, hugs -> 1)

  "A horse, a horse! My kingdom for a horse!" gives Map(for -> 1, a -> 3, my -> 1, horse -> 3, kingdom -> 1)


  "A woman is but a woman. A man is but a man. A child is but a child."

  gives

  Map(is -> 3, but -> 3, man -> 2, child -> 2, woman -> 2, a -> 6)

Use string.split("(\\s|\\.|,|!)+") to separate words in a string into a list
(split gets a regular expression, this expression says a single character to split on is:
  * whitespace (\s, we need to escape the \, so it becomes \\s)
  * OR . (. means any character in regular expression, so we escape it for the literal \\.)
  * OR , OR !
 the + indicates we can have any number >= 1 of the characters above)
Use string.toLowerCase to convert a string to lowercase

Indication of length : 7 lines
   */


  def wordCount(text : String) : mutable.Map[String,Int] =
  {
    val map1 = text.split {"(\\s|\\.|,|!)+" }
    val map2 = mutable.Map[String,Int]()
    for (i<-0 until map1.length)
    {
      map1(i) = map1(i).toLowerCase()
      if(map2.contains(map1(i)))        //check if word is already in the map(2), if it is: increase count of that word else insert it into the map
      {
        map2(map1(i)) += 1
      }
      else
      {
        map2 += (map1(i) -> 1)
      }
    }
    return map2
  }




  /* Assignment 6: Elo ratings

  In chess and various other forms of competition the skill of a player is indicated using
  the Elo-rating system. The elo-rating is a number in the 0-3000 range, where 0 is an absolute beginner and
  2900 is the world champion.

  When given the ratings of two players rA and rB, the probability of player A winning the game is calculated as follows:

  eA = 1 / ( 1 + 10^((rB - rA) / 400)))

  This gives a number between 0 (100 % sure that A loses) and 1 (100 % sure that A wins).

  After playing a number of games, the elo ratings of the player are updated based on their wins and loses. The
  new elo rate delta (change in elo rate) of a player is calculated based on the probability of a winning eA and the
  actual outcome aA.

  drA = k * (aA - eA)

  where k is a constant controlling the importance of the game (typically between 16-32). We use k=24. The change in elo
  rating of a player A is the sum of all the elo rate deltas of the individual games that player A played.

  For example, if player A has an elo rating of 2000 and player B has a elo rating of 2200, then the probability of
  player A winning is 1 / ( 1 + 10^((2200 - 2000) / 400))) = 0.24. Suppose player A wins the game, so aA=1.0.
  Then the new elo rate delta of player A is 24 * (1.0 - 0.24) = 18.24. If we would update elo ratings after this single
  game, the elo rate of player A would be 2000 + 18.24 = 2018.24. For player B the computation is reversed,
  his probability of winning was (1 - 0.24) = 0.76. The elo rate delta score of player B is
24 * (0.0 - 0.76) = -18.24. If we would update elo ratings after this single
   game, the elo rate of player A would be 2200 - 18.24 = 2181.76.

When playing a tournament, the elo rates are kept fixed during the tournament, and are updated afterwards based on all
the games that were played. The elo score change of each player A is the sum of all the elo rate deltas from the individual
that A played. The elo scores used when computing these deltas are the elo scores from _before_ the tournament.

  Program a method which updates the elo ratings of all players. You can (and should) add method to classes below.


  We test this as follows:

    val jaap = new Player("Jaap",2000)
    val piet = new Player("Piet", 2200)
    val game = new Game(jaap,piet,0.0)
    Exercises.updateEloScores(List(jaap,piet), List(game) )
    assert(jaap.rating === 2018.24 +- 0.01)
    assert(piet.rating === 2181.76 +- 0.01)

Indication of length 12 added lines
   */

  val eloK = 24

  def updateEloScores(players : List[Player] , games : List[Game]) : Unit =
  {
    for (i<- 0 until games.length)
    { //go through list of games, check the two names and don't update the scores of the two players of that game yet but update the deltas
      games(i).playerA.rating +=  games(i).newRatingA
      games(i).playerB.rating +=  games(i).newRatingB
    }                                //when all games have been read, update elo rating with sum of delta's per player. Use map to store deltas?
  }

  class Player( val name : String, var rating : Double)
  {
  }

  class Game( val playerA : Player,
              val playerB : Player,
              val outcome : Double,
            )
  {
    var outcomeA : Double = 0
    var outcomeB : Double = 0
    if (outcome == 0)
    {                   // 0 means playerA won so give value 1 to outcomeA, 1 means playerB won, 0.5 means draw
      outcomeA = 1
      outcomeB = 0
    }
    else if (outcome == 1)
    {
      outcomeA = 0
      outcomeB = 1
    }
    else
    {
      outcomeA = 0.5
      outcomeB = 0.5
    }
    val probAwinning : Double = 1.0 / (1.0 + scala.math.pow(10.0,((playerB.rating - playerA.rating)) / 400.0))
    val probBwinning: Double = 1 - probAwinning
    var newRatingA : Double= eloK * (outcomeA - probAwinning)
    var newRatingB : Double = eloK * (outcomeB - probBwinning)
  }

  /* Assignment 7: List speed offenders.

  Two sets of cameras have been set up on the A2 highway near Utrecht to find speed offenders. The sets of cameras,
  named "A" and "B", are 1500 meters apart and the speed limit is 100 km/h between 6:00 and 19:00 and 120 km/h between
  19:00 and 6:00. If a car passed both cameras in a (single) 6:00 to 19:00 period then it is going too fast if its speed
  is over 100 km/h. If a car passed either camera in between 19:00 and 6:00, then it is going too fast if its speed is over
  120 km/h.

  The cameras are only positioned on the North-bound direction of the highway, and the cars first pass the "A" cameras
  and then the "B" cameras. After processing, the cameras produce a list of cars and the times they passed as
  follows (definitions of Observation and Time below):

  Observation(cameraSet = "A", licensePlate = "DX-98-DW", time = Time(18492, 13, 3, 5.0))
  Observation(cameraSet = "B", licensePlate = "WW-11-XX", time = Time(18492, 13, 3, 6.0))
  Observation(cameraSet = "A", licensePlate = "MV-33-PP", time = Time(18492, 13, 3, 6.2))
  Observation(cameraSet = "A", licensePlate = "33-XX-RR", time = Time(18492, 13, 3, 6.7))
  ....
  Observation(cameraSet = "A", licensePlate = "DX-98-DW", time = Time(18492, 13, 4, 1.0))

  Program a method that takes the observations of the cameras and produces a list of speed offenders and their speed
  in the order that the cars passed cameraset B.

   ----> Use a map of type Map[String,Time] to store & look up for each license plate at which time cameraset "A" was passed.

Indication of length: 25 added lines. Add extra functions for conceptually definitions to organize your code.
   */

  // The used epoch is 1 January 1970. An epoch is an instant in time chosen as the origin of a time scale.
  // (see https://en.wikipedia.org/wiki/Epoch)
  case class Time(daysSinceEpoch : Int, hours : Int, minutes : Int, seconds : Double)
  {
    var totalSeconds : Double = (daysSinceEpoch * 86400) + (hours * 3600) + (minutes * 60) + seconds //store this with the license plate
  }
  // case class means (among other things) that you do not have to type new to create one
  // so instead of new Time(43,6,3,0) you just type Time(43,6,3,0)
  // equality and pretty printing are also defined for you

  case class Observation(cameraSet : String, licensePlate : String, time : Time )
  // to convert your speed of type double to an Int use Math.round(speed).toInt
  case class SpeedOffender(licensePlate : String, speed : Int)

  def speedOffenders(observations: Seq[Observation]) : ArrayBuffer[SpeedOffender] =
  {
    val startTimeOfCar : mutable.Map[String,Double] = new mutable.HashMap()
    val result :  ArrayBuffer[SpeedOffender] = new ArrayBuffer()
    for(observation <- observations) {
      if(observation.cameraSet == "A") startTimeOfCar += (observation.licensePlate -> observation.time.totalSeconds) //store time car passed A in map
      else
      {
        var diff : Double = observation.time.totalSeconds- startTimeOfCar.get(observation.licensePlate).get //extra .get to get rid of option[Int] but get Int
        var speed : Int = (((1500 / diff) * 3600) / 1000).toInt
        if(observation.time.hours > 6 && observation.time.hours < 19 && speed > 100) result += SpeedOffender(observation.licensePlate,speed) //if and else if statement for different times when driver is speeding
        else if (speed > 120) result += SpeedOffender(observation.licensePlate, speed)
      }
    }
    result
  }

  /* Assignment 8: Program a method that split any non-empty array into two arrays.

  If the input array has length l , the first array should have length (l / 2),
  the second array should have length (l - (l / 2)). For example, for
  length 7, the first array has length 3, while the second has length 4.

  If we put the second resulting array behind the first one, we should get the original array back.

  implement this using a for loop (or two),
  do not use library functions such as splitAt
  (we will subtract the points for this assignment if you do)

  example: [5,3,10,5,6,1,3] -> ([5,3,10],[5,6,1,3])

Use "new Array[Int](length)" to create a new int array

Indication of length: 12 lines

   */
  def splitArray(a : Array[Int]) : (Array[Int],Array[Int]) =
  {
    val array1 = new Array[Int](a.length / 2 )
    val array2 : Array[Int] = new Array[Int](a.length - (a.length / 2))
    for (i<-0 until a.length / 2 )
    {
      array1(i) = a(i)
    }
    var indexArray2 : Int = 0
    for (j <- 0 until (a.length - (a.length / 2)))
    {
      array2(indexArray2) = a(j + (a.length/2))
      indexArray2 += 1
    }
    return (array1, array2)
  }


  /* Assignment 9:

     Program a method that when given two sorted arrays a and b, returns a new sorted array c that has the elements from
     array a and array b. For example
     when given
     a = [1,3,5,6,10]
     b = [1,4,6,8]

     the resulting array should be:

     c = [1,1,3,4,5,6,6,9,10]

     This method should not call a sorting method. Instead the resulting array should be produced by "zipping" the
     two input arrays together. We repeatedly select the least element that we did not consider before from a and b
     and include this in c.

     For example:

     a = [1,3,5,6,10]
              ^
     b = [1,4,6,8]
            ^
     c = [1,1,3,...]
      the arrows (^) point to the lowest element we did not consider before. Of these, the element 4 from b is less
      then the element 5 from a. For this reason, we select 4 as the next element and advance the arrow ^ for b to point
      to 6.


    implement this using a loop
    do not use library functions such as sort
    (we will subtract the points for this assignment if you do)

Indication of length: 15 lines
   */
  def mergeSortedArrays(a : Array[Int], b : Array[Int]) : Array[Int] =
  {
    val resultArray = new Array[Int](a.length + b.length)
    var counter = 0
    var index1 = 0
    var index2 = 0

    while (index1 < a.length && index2 < b.length)        //go through both arrays, stop if one has reached its end
    {
      if(a(index1) <= b(index2))
      {
        resultArray(counter) = a(index1)
        index1+=1
      }
      else
      {
        resultArray(counter) = b(index2)
        index2+=1
      }
      counter+=1
    }
    if(index1 < a.length)                     //if array1(a) hasn't reached its end yet, put the rest of the numbers in the final array
    {
      for(k<-index1 until a.length)
      {
        resultArray(counter) = a(k)
        counter +=1
      }
    }
    else                                    //same principle as block above, but for if there are elements left in array2
    {
      for (l <- index2 until b.length)
      {
        resultArray(counter) = b(l)
        counter += 1
      }
    }
    return resultArray
  }





  /* Assignment 10 : Mergesort

  Implement mergesort. Mergesort is a recursive sorting algorithm that works by splitting arrays and merging them.
  More precisely, mergeSort works as follows:

   If the input array is of length <= 1, then the array is already sorted an the input array is returned.
   If the input array is of length >= 2, then the input array is split in two (using the splitArray function),
   the two smaller arrays are sorted using mergesort, and the resulting two sorted arrays are merged using
   mergeSortedArrays.

   Example:
        7 5 6 1    -- split
       /      \     -- merge sort left and right recursively
     7 5     6 1     -- split both sides
     /  \    /  \
    7   5   6   1     -- merge both sides
    \   /   \  /    -- left and right merge sort done
    5 7     1 6    -- merge
     \       /
      1 5 6 7

Indication of length: 5 lines
   */

  def mergeSort(a : Array[Int]) : Array[Int] =
  {
    if(a.length == 1) return a                        //this is the base case
    var midIndex : Int = Math.floor(a.length / 2).toInt
    var leftSide : Array[Int] = mergeSort(a.slice(0, midIndex))       //sort the left side of the original array
    var rightSide : Array[Int] = mergeSort(a.slice(midIndex,a.length))  //sort right side of the original array
    return mergeSortedArrays(leftSide, rightSide)               //return the sorted array which is obtained by merging the two sorted array left and right
  }



}

