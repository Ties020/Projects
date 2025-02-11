package tetris.logic

import engine.random.{RandomGenerator, ScalaRandomGen}
import tetris.logic.TetrisLogic._

import scala.collection.mutable.ArrayBuffer

class TetrisLogic(val randomGen: RandomGenerator,
                  val gridDims : Dimensions,
                  val initialBoard: Seq[Seq[CellType]]) {

  def this(random: RandomGenerator, gridDims: Dimensions) =
    this(random, gridDims, makeEmptyBoard(gridDims))

  def this() =
    this(new ScalaRandomGen(), DefaultDims, makeEmptyBoard(DefaultDims))

  private class gameState(var board: Seq[Seq[CellType]]) { //this class contains the board where the tetrominoes aren't currently being moved

    def convertBoardToPoints(usedForAddingBlock: Boolean): Vector[Point] = { //use this to check for moving/rotating into other tetrominoes
      var tetrominoPoints = Vector.empty[Point]
      for (rowIndex <- board.indices) {
        for (colIndex <- board(rowIndex).indices) {
          if (!usedForAddingBlock && board(rowIndex)(colIndex) != Empty) tetrominoPoints :+= Point(colIndex, rowIndex)
          else if (usedForAddingBlock) tetrominoPoints :+= Point(colIndex, rowIndex)
        }
      }
      tetrominoPoints
    }

    private def clearFullLines(): Unit = {
      if (countFullLines() != 0) {
        val boardAfterRemoval: ArrayBuffer[ArrayBuffer[CellType]] = ArrayBuffer.fill(gridDims.height)(ArrayBuffer.fill(gridDims.width)(Empty))
        for (row <- 0 until 1; col <- 0 until gridDims.width) boardAfterRemoval(row)(col) = Empty //fill top row with empty cells
        for (row <- 1 until indexOfFullRow() + 1; col <- 0 until gridDims.width) { //move down all cells on the board, until the to be removed line is met
          val currentPoint = Point(col, row - 1)
          boardAfterRemoval(row)(col) = getCellTypeFromBoard(currentPoint)
        }
        if (indexOfFullRow() != gridDims.height - 1) { //cells after full line stay the same on the board
          for (row <- indexOfFullRow() + 1 until gridDims.height; col <- 0 until gridDims.width) {
            val currentPoint = Point(col, row)
            boardAfterRemoval(row)(col) = getCellTypeFromBoard(currentPoint)
          }
        }
        board = boardAfterRemoval.map(_.toSeq).toSeq
        clearFullLines()                //recursive call to remove any remaining full lines
      }
    }

    private def indexOfFullRow(): Int = board.indexWhere(row => !row.contains(Empty))

    private def countFullLines(): Int = board.count(row => !row.contains(Empty))

    def addTetrominoToBoard(): Unit = {
      val boardPostAddition: Seq[Seq[CellType]] = board.zipWithIndex.map { case (row, rowIndex) =>
        row.zipWithIndex.map { case (_, colIndex) => getCellType(Point(colIndex, rowIndex)) }          //board is updated with the celltypes of the newly added tetromino
      }
      board = boardPostAddition
      clearFullLines()                 //check if the new board has any full lines that need to be removed
    }

    private def getCellTypeFromBoard(point: Point): CellType = {      //this is needed to only remove cells from the full lines that are on the board before generating a new tetromino
      if (point.y >= 0 && point.y < board.length && point.x >= 0 && point.x < board(point.y).length) board(point.y)(point.x)     //if the point/cell is on the board -> return the cellrype of that point
      else Empty                                                                                                                 // default value when the cell is out of bounds
    }
  }

  class currTetromino(var tetrominoVec: Vector[Point], var relativeToAnchor: Vector[Point]) {

    def postRotation(rhs: Vector[Point]): Vector[Point] = rhs.map { point => Point(anchor.x + point.x, anchor.y + point.y) }

    def convToRelative(): Vector[Point] = { //used for rotations
      val updatedRelative: Vector[Point] = currentTetromino.tetrominoVec.map { point =>
        if (point != anchor) Point(point.x - anchor.x, point.y - anchor.y) else Point(0, 0)
      }
      updatedRelative
    }

    def rotateLeft(icellCase: Int): Unit = {
      relativeToAnchor = convToRelative()
      val transformedVector = relativeToAnchor.map(point => Point(point.y, -point.x + icellCase))
      val rotatedBlock = postRotation(transformedVector)
      if (!tetrominoIsOutOfBounds(rotatedBlock, "all") && !tetrominoShouldBeAddedToBoard(rotatedBlock)) {
        currentTetromino.tetrominoVec = rotatedBlock
        currentTetromino.relativeToAnchor = transformedVector
      }
    }

    def rotateRight(icellCase: Int): Unit = {
      relativeToAnchor = convToRelative()
      val transformedVector = relativeToAnchor.map(point => Point(-point.y + icellCase, point.x))
      val rotatedBlock = postRotation(transformedVector)
      if (!tetrominoIsOutOfBounds(rotatedBlock, "all") && !tetrominoShouldBeAddedToBoard(rotatedBlock)) {
        currentTetromino.tetrominoVec = rotatedBlock
        currentTetromino.relativeToAnchor = transformedVector
      }
    }
  }

  private val initialGameState = new gameState(initialBoard)
  private val middleColumn: Int = math.ceil(gridDims.width / 2).toInt
  private var anchor: Point = if (isEven(gridDims.width)) Point(middleColumn - 1, 1) else Point(middleColumn, 1)
  private var currCellType: CellType = _
  private val currentTetromino = new currTetromino(generateBlock(), null)

  private def isEven(x: Int) = x % 2 == 0

  private def checkGameOver(): Boolean = {
    val boardPoints = initialGameState.convertBoardToPoints(false)
    val filteredVector = boardPoints.filter(currentTetromino.tetrominoVec.contains)
    if (filteredVector.nonEmpty) true else false
  }

  private def generateBlock(): Vector[Point] = {
    val nrTetrominoes: Int = 7
    val randInt: Int = randomGen.randomInt(nrTetrominoes)
    randInt match {
      case 0 =>
        currCellType = ICell
        Vector(anchor, anchor.blockToTheLeft, anchor.blockToTheRight, anchor.blockToTheRightTwice)
      case 1 =>
        currCellType = JCell
        Vector(anchor, anchor.blockLeftUp, anchor.blockToTheLeft, anchor.blockToTheRight)
      case 2 =>
        currCellType = LCell
        Vector(anchor, anchor.blockToTheLeft, anchor.blockToTheRight, anchor.blockRightUp)
      case 3 =>
        currCellType = OCell
        Vector(anchor, anchor.blockToTheRight, anchor.blockRightUp, anchor.blockAbove)
      case 4 =>
        currCellType = SCell
        Vector(anchor, anchor.blockToTheLeft, anchor.blockAbove, anchor.blockRightUp)
      case 5 =>
        currCellType = TCell
        Vector(anchor, anchor.blockToTheLeft, anchor.blockAbove, anchor.blockToTheRight)
      case 6 =>
        currCellType = ZCell
        Vector(anchor, anchor.blockLeftUp, anchor.blockAbove, anchor.blockToTheRight)
    }
  }

  private def tetrominoIsOutOfBounds(newBlock: Vector[Point], dir: String): Boolean = {
    dir match {
      case "right" => newBlock.exists(point => point.x >= gridDims.width)
      case "left" => newBlock.exists(point => point.x < 0)
      case "down" => newBlock.exists(point => point.y > gridDims.height)
      case _ => newBlock.exists(point => point.x < 0) || newBlock.exists(point => point.x >= gridDims.width)
    }
  }

  private def tetrominoShouldBeAddedToBoard(vector: Vector[Point]): Boolean = {
    val boardPoints = initialGameState.convertBoardToPoints(false)
    val filteredVector = boardPoints.filter(vector.contains)
    filteredVector.map(point => Point(point.x, point.y + 1))
    if (filteredVector.nonEmpty) return true               //add tetromino to board if a Point of the tetromino isn't already on the board (doesn't have a celltype other than Empty)
    if (vector.exists { case Point(_, y) => y == gridDims.height }) true else false   //add tetromino to the board if bottom row is reached
  }

  def rotateLeft(): Unit = {
    if (!checkGameOver()) {
      currCellType match {
        case JCell | SCell | LCell | TCell | ZCell =>
          currentTetromino.rotateLeft(0)

        case ICell =>
          currentTetromino.rotateLeft(1)
        case OCell => return
      }
    }
  }

  def rotateRight(): Unit = {
    if (!checkGameOver()) {
      currCellType match {
        case JCell | SCell | LCell | TCell | ZCell =>
          currentTetromino.rotateRight(0)
        case ICell =>
          currentTetromino.rotateRight(1)
        case OCell => return
      }
    }
  }

  def moveLeft(): Unit = {
    if (!checkGameOver()) {
      val newTetromino = currentTetromino.tetrominoVec.map(point => Point(point.x - 1, point.y))
      if (!tetrominoIsOutOfBounds(newTetromino, "left")) {
        anchor = Point(anchor.x - 1, anchor.y)
        if (!tetrominoShouldBeAddedToBoard(newTetromino)) currentTetromino.tetrominoVec = newTetromino
        else anchor = Point(anchor.x + 1, anchor.y)
      }
    }
  }

  def moveRight(): Unit = {
    if (!checkGameOver()) {
      val newTetromino = currentTetromino.tetrominoVec.map(point => Point(point.x + 1, point.y))
      if (!tetrominoIsOutOfBounds(newTetromino, "right")) {
        anchor = Point(anchor.x + 1, anchor.y)
        if (!tetrominoShouldBeAddedToBoard(newTetromino)) currentTetromino.tetrominoVec = newTetromino
        else anchor = Point(anchor.x - 1, anchor.y)
      }
    }
  }

  def moveDown(): Unit = {
    if (!checkGameOver()) {
      val newTetromino = currentTetromino.tetrominoVec.map(point => Point(point.x, point.y + 1))
      if (!tetrominoIsOutOfBounds(newTetromino, "down")) {
        anchor = Point(anchor.x, anchor.y + 1)
        if (!tetrominoShouldBeAddedToBoard(newTetromino)) currentTetromino.tetrominoVec = newTetromino
        else {                                                                                        //this is when the block hits another block or the last row is reached
          initialGameState.addTetrominoToBoard()
          anchor = if (isEven(gridDims.width)) Point(middleColumn - 1, 1) else Point(middleColumn, 1)
          currentTetromino.tetrominoVec = generateBlock()
          currentTetromino.relativeToAnchor = currentTetromino.convToRelative()
        }
      }
    }
  }

  def doHardDrop(): Unit = {
    if (!checkGameOver()) {
      while (!tetrominoShouldBeAddedToBoard(currentTetromino.tetrominoVec.map(point => Point(point.x, point.y + 1)))) currentTetromino.tetrominoVec = currentTetromino.tetrominoVec.map(point => Point(point.x, point.y + 1))
      initialGameState.addTetrominoToBoard()
      anchor = if (isEven(gridDims.width)) Point(middleColumn - 1, 1) else Point(middleColumn, 1)
      currentTetromino.tetrominoVec = generateBlock()
      currentTetromino.relativeToAnchor = currentTetromino.convToRelative()
    }
  }

  def isGameOver: Boolean = checkGameOver()

  def getCellType(p: Point): CellType = if (currentTetromino.tetrominoVec.contains(p)) currCellType else initialGameState.board(p.y)(p.x)

}

object TetrisLogic {
  val FramesPerSecond: Int = 5
  val DrawSizeFactor = 1.0

  def makeEmptyBoard(gridDims : Dimensions): Seq[Seq[CellType]] = {
    val emptyLine = Seq.fill(gridDims.width)(Empty)
    Seq.fill(gridDims.height)(emptyLine)
  }
  val DefaultWidth: Int = 10
  val NrTopInvisibleLines: Int = 4
  val DefaultVisibleHeight: Int = 20
  val DefaultHeight: Int = DefaultVisibleHeight + NrTopInvisibleLines
  val DefaultDims : Dimensions = Dimensions(width = DefaultWidth, height = DefaultHeight)

  def apply() = new TetrisLogic(new ScalaRandomGen(),
    DefaultDims,
    makeEmptyBoard(DefaultDims))
}
