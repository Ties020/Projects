#include "Maze.h"
#include "Coordinates.h"
#include <iostream>
#include <cstdlib>

Maze::Maze(int rows, int columns)
{
   numRows = rows;
   numCols = columns;
   grid = std::vector<std::vector<Cell>>(numRows, std::vector<Cell>(numCols)); //fill 2d vector with cells
}

Cell& Maze::at(Coordinates coord)  //makes you able to call cell by its coordinates
{
   return grid.at(coord.y).at(coord.x); 
}

void Maze::removeWalls(Coordinates coord,Coordinates nextCell) //removes walls between two cells
{  
   if (coord.x == nextCell.x && coord.y < nextCell.y)
   {
      at(coord).bottomWall = false;
      at(nextCell).topWall = false;
   }
   if (coord.x == nextCell.x && coord.y > nextCell.y)
   {
      at(coord).topWall = false;
      at(nextCell).bottomWall = false;
   }
   if (coord.x < nextCell.x && coord.y == nextCell.y)
   {
      at(coord).rightWall = false;
      at(nextCell).leftWall = false;
   }
   if (coord.x > nextCell.x && coord.y == nextCell.y)
   {
      at(coord).leftWall = false;
      at(nextCell).rightWall = false;
   }
}

bool Maze::isValidCoordinate(Coordinates coord) //checks if coordinates of to be visited cell is within the maze
{
   if (coord.x < 0 || coord.y < 0) return false;
   if (coord.y >= numRows || coord.x >= numCols) return false;
   return true;
}

std::vector <Coordinates> Maze::validNeighbours(Coordinates coord) //returns list of all valid neighbours
{
   std::vector <Coordinates> tempvec;

   Coordinates above(coord.x, coord.y - 1);
   if (isValidCoordinate(above) && !at(above).visited) tempvec.push_back(above);

   Coordinates below(coord.x, coord.y + 1);
   if (isValidCoordinate(below) && !at(below).visited) tempvec.push_back(below);

   Coordinates left(coord.x - 1, coord.y);
   if (isValidCoordinate(left) && !at(left).visited) tempvec.push_back(left);

   Coordinates right(coord.x + 1, coord.y);
   if (isValidCoordinate(right) && !at(right).visited) tempvec.push_back(right);

   return tempvec;
}

void Maze::genRandomMaze(Coordinates coord)
{
   at(coord).visited = true;
   visitedCells += 1;
   if (visitedCells == numCols * numRows)
   {
      return;
   }
   while (!validNeighbours(coord).empty()) 
   {
      int randIndex = rand() % validNeighbours(coord).size();
      Coordinates nextCell = validNeighbours(coord).at(randIndex);
      removeWalls(coord, nextCell);
      genRandomMaze(nextCell);
   }
}

void Maze::allCellsVisistedToFalse()
{
   for(int i=0;i< numRows;++i)
   {
      for(int j=0; j< numCols;++j)
      {
         Coordinates coord(j,i);
         at(coord).visited = false;
      }
   }
}

std::vector <Coordinates> Maze::neighBours(Coordinates coord)  //returns list of valid neighbours when finding a path, so no wall between from and to
{
   std::vector <Coordinates> neighbourList;

   Coordinates above(coord.x, coord.y - 1);
   if (isValidCoordinate(above) && !at(above).bottomWall && !at(coord).topWall) neighbourList.push_back(above);

   Coordinates below(coord.x, coord.y + 1);
   if (isValidCoordinate(below) && !at(below).topWall && !at(coord).bottomWall) neighbourList.push_back(below);

   Coordinates left(coord.x - 1, coord.y);
   if (isValidCoordinate(left) && !at(left).rightWall && !at(coord).leftWall) neighbourList.push_back(left);

   Coordinates right(coord.x + 1, coord.y);
   if (isValidCoordinate(right) && !at(right).leftWall && !at(coord).rightWall) neighbourList.push_back(right);

   return neighbourList;
}

bool Maze::findPath(Coordinates coord,Coordinates endcoord)
{  
   at(coord).visited = true;
   if (coord.x == endcoord.x && coord.y==endcoord.y) //if cell in bottom right corner is reached, path is found
   {
      return true;
   }
   std::vector<Coordinates> neighbours = neighBours(coord); //list of neighbours
   for(int i=0; i < neighbours.size();++i)  //goes through neighbours
   {
      Coordinates coord = neighbours.at(i);
      if (at(coord).visited == false)
      {
         if (findPath(coord,endcoord)) //call recursive function until path is found
         {
            return true;
         }
      }
   }
   at(coord).visited = false;
   return false;   
}

void Maze::printPath()
{
   for(int i=0;i< numRows;++i)
   {
      for(int j=0; j< numCols;++j) //print top wall first
      {
         Coordinates coord(j,i);
         at(coord).printTop();
      }
      std::cout << "+";
      std::cout<<std::endl;

      for(int k=0; k< numCols;++k) //print left wall and visited mark
      {
         Coordinates coord(k,i);
         at(coord).printLeft();
         at(coord).printContent();
      }
      std::cout << "|";  //print right wall when the last column is reached
      std::cout<<std::endl;
   }
   for(int j=0; j< numCols;++j) //print bottom wall
   {
      std::cout << "+---";
   }
   std::cout << "+" << std::endl;
}