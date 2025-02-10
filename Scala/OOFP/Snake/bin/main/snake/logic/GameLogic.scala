package snake.logic

import engine.random.{RandomGenerator, ScalaRandomGen}
import snake.logic.GameLogic._

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks.break

/** To implement Snake, complete the ``TODOs`` below.
 *
 * If you need additional files,
 * please also put them in the ``snake`` package.
 */
class GameLogic(val random: RandomGenerator,val gridDims : Dimensions)
{
  //arrays used for storing all game elements (snake body, head etc.) at every step in the game
  val arraySnakeBodies = ArrayBuffer[ArrayBuffer[Point]]()
  val arrayHeads  = ArrayBuffer[Point]()
  val arrayApplePos = ArrayBuffer[Point]()
  val arrayAddApples = ArrayBuffer[Int]()
  val dirArrayGamestate = ArrayBuffer[Direction]()

  //initialize direction and put it in the arrays which keep track of it
  val dirArray = ArrayBuffer[Direction]() //used in case quickswitch happens
  var dir : Direction = East()
  dirArray += dir
  dirArrayGamestate += dir

  var player : Point = Point(2, 0)
  arrayHeads += player

  var currSnakeBody = ArrayBuffer[Point]()
  currSnakeBody += Point(1, 0)
  currSnakeBody += Point(0, 0)
  arraySnakeBodies += currSnakeBody.clone()

  var apple : Point = _
  val possibleAppleLocations = scala.collection.mutable.Map[Int, Point]()
  var applesToAdd : Int = 0
  arrayAddApples += applesToAdd
  updateApple()
  arrayApplePos += apple

  var gameOver : Boolean = false
  var indexGamestate : Int = 0   //used to keep track of current game state to reverse when needed
  var toBeAppended : Point = _
  var reverse : Boolean = false

  def setReverse(r : Boolean): Unit = {
    reverse = r
    if(!reverse) arraySnakeBodies(indexGamestate) = currSnakeBody.clone() //save current snake body in the array of snake bodies when setting reverse to false
  }

  def changeDir(d : Direction) : Unit = {
    if(dirArray.length == 1) {   //check if this is the first time changeDir was called since the last step()
      dir = d
      dirArray += dir
    }

    else {                     //make the last valid direction that was given before step() is called, the next direction (quickswitch case)
      dirArray += d
      for(i<-1 until dirArray.length) {
        if(dirArray(0).opposite != dirArray(i)) dir = dirArray(i)
      }
    }
  }

  def getCellType(cell : Point): CellType = {
    if(cell == player) SnakeHead(dir)
    else if(currSnakeBody.contains(cell)) SnakeBody(0.0f)
    else if(cell == apple) Apple()
    else Empty()
  }

  def updateApple() : Unit = {
    var freeSpotIndex : Int = 0
    //go through the grid and mark empty spots with indices where apple can be spawned
    for(y <- 0 until gridDims.height; x <- 0 until gridDims.width) {
      if (getCellType(Point(x, y)) == Empty()) {
        possibleAppleLocations += (freeSpotIndex -> Point(x, y))
        freeSpotIndex += 1
      }
    }
    val randInt : Int = random.randomInt(freeSpotIndex+1)
    if(freeSpotIndex != 0) apple = possibleAppleLocations(randInt) //condition is not met when there are no free spaces left for the apple to be spawned
  }

  def step() : Unit = {
    if(gameOver && reverse || reverse) reverseStep()
    if(!reverse && !gameOver) {
      if (applesToAdd > 0) {      //let snake grow by 1 each step after eating apple
        currSnakeBody += toBeAppended
        applesToAdd -= 1
      }
      toBeAppended = currSnakeBody(currSnakeBody.length-1)

      for (i <- currSnakeBody.length - 1 to 1 by -1) currSnakeBody(i) = currSnakeBody(i - 1) //update the snake's current body
      currSnakeBody(0) = player
      moveSnakeHead()

      if (apple == player){ //when snake eats apple->respawn apple and increment snake's body by 1
        updateApple()
        applesToAdd += 3
      }
      updateGameStateArrays()
    }
  }

  def updateGameStateArrays() : Unit = {
    indexGamestate += 1
    arraySnakeBodies += currSnakeBody.clone() //use clone to not alter the snake body of the previous game state when changing the current one
    arrayHeads += player
    arrayApplePos += apple
    arrayAddApples += applesToAdd
    dirArrayGamestate += dir
  }

  def reverseStep() : Unit = {
    if(indexGamestate > 0 && reverse) {
        //remove all game elements of the previous game state from the arrays to avoid going back to this when continuing
        arraySnakeBodies.remove(indexGamestate)
        arrayHeads.remove(indexGamestate)
        arrayApplePos.remove(indexGamestate)
        arrayAddApples.remove(indexGamestate)
        dirArrayGamestate.remove(indexGamestate)

        //update game state
        indexGamestate -= 1
        gameOver = false
        currSnakeBody = arraySnakeBodies(indexGamestate)
        player = arrayHeads(indexGamestate)
        apple = arrayApplePos(indexGamestate)
        applesToAdd = arrayAddApples(indexGamestate)
        dir = dirArrayGamestate(indexGamestate)
    }
  }

  def moveSnakeHead() : Unit = {
    dir match {
      case West() =>
        if (currSnakeBody(1) == Point(player.x - 1, player.y) && currSnakeBody(1) != Point(0,player.y)){  //in case player tries to move left while going right
          //keep moving right
          if (player.x == gridDims.width - 1) player = Point(0, player.y) else player = Point(player.x + 1, player.y)
          dir = dir.opposite
        }
        //condition below is used to give game over if snake head hits body
        else if (currSnakeBody.contains(Point(player.x - 1, player.y)) || (player.x == 0 && currSnakeBody.contains(Point(gridDims.width - 1, player.y)))) gameOver = true
        else if (player.x == 0) player = Point(gridDims.width - 1, player.y) else player = Point(player.x - 1, player.y)

      case East() =>
        if (currSnakeBody(1) == Point(player.x + 1, player.y) && currSnakeBody(1) != Point(gridDims.width-1,player.y)){
          if (player.x == 0) player = Point(gridDims.width - 1, player.y) else player = Point(player.x - 1, player.y)
          dir = dir.opposite
        }

        else if (currSnakeBody.contains(Point(player.x + 1, player.y)) || (player.x == gridDims.width - 1 && currSnakeBody.contains(Point(0, player.y)))) gameOver = true
        else if (player.x == gridDims.width - 1) player = Point(0, player.y) else player = Point(player.x + 1, player.y)

      case North() =>
        if (currSnakeBody(1) == Point(player.x, player.y - 1) && currSnakeBody(1) != Point(player.x,0)){  //in case player tries to move up while going down
          if (player.y == gridDims.height - 1) player = Point(player.x, 0) else player = Point(player.x, player.y + 1)
          dir = dir.opposite
        }
        else if (currSnakeBody.contains(Point(player.x, player.y - 1)) || (player.y == 0 && currSnakeBody.contains(Point(player.x, gridDims.width - 1)))) gameOver = true
        else if (player.y == 0) player = Point(player.x, gridDims.height - 1) else player = Point(player.x, player.y - 1)

      case South() =>
        if (currSnakeBody(1) == Point(player.x, player.y + 1) && currSnakeBody(1) != Point(player.x,gridDims.height-1)){
          if (player.y == 0) player = Point(player.x, gridDims.height - 1) else player = Point(player.x, player.y - 1)
          dir = dir.opposite
        }
        else if (currSnakeBody.contains(Point(player.x, player.y + 1)) || (player.y == gridDims.width - 1 && currSnakeBody.contains(Point(player.x, 0)))) gameOver = true
        else if (player.y == gridDims.height - 1) player = Point(player.x, 0) else player = Point(player.x, player.y + 1)
    }
    dirArray.clear()  //update array to only have the direction in which the snake actually went
    dirArray += dir
  }
}

/** GameLogic companion object */
object GameLogic {

  val FramesPerSecond: Int = 5 // change this to increase/decrease speed of game

  val DrawSizeFactor = 1.0 // increase this to make the game bigger (for high-res screens)
  // or decrease to make game smaller

  // These are the dimensions used when playing the game.
  // When testing the game, other dimensions are passed to
  // the constructor of GameLogic.
  //
  // DO NOT USE the variable DefaultGridDims in your code!
  //
  // Doing so will cause tests which have different dimensions to FAIL!
  //
  // In your code only use gridDims.width and gridDims.height
  // do NOT use DefaultGridDims.width and DefaultGridDims.height
  val DefaultGridDims
  : Dimensions =
  Dimensions(width = 25, height = 25)  // you can adjust these values to play on a different sized board
}