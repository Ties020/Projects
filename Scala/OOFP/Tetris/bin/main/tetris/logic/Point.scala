package tetris.logic

// you can alter this file!

case class Point( x : Int, y : Int) {
  def +(rhs: Point): Point = Point(x + rhs.x, y + rhs.y)

  //make some functions to generate the initial tetromino based on anchor position
  def blockToTheLeft : Point = copy(x = x-1)
  def blockToTheRight : Point = copy(x = x+1)
  def blockToTheRightTwice : Point = copy(x = x+2)
  def blockAbove : Point = copy(y = y-1)
  def blockLeftUp : Point = copy(x = x-1, y = y-1)
  def blockRightUp : Point = copy(x = x+1, y = y-1)
}
