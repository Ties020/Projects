#ifndef Maze_H
#define Maze_H

#include <vector>
#include <stack>
#include "Cell.h"
#include "Coordinates.h"

class Maze
{
   public:
      Maze(int rows, int columns);
      Cell& at(Coordinates coord); 
      void genRandomMaze(Coordinates coord);
      bool findPath(Coordinates coord, Coordinates endcoord);
      void printPath();
      void removeWalls(Coordinates coord, Coordinates nextCell);
      int rows() const {return numRows;};
      int columns() const {return numCols;};
      std::vector<Coordinates> validNeighbours(Coordinates coord);
      std::vector <Coordinates> neighBours(Coordinates coord);
      bool isValidCoordinate(Coordinates coord);
      void allCellsVisistedToFalse();

   private:
      std::vector<std::vector<Cell>> grid; //so every coordinate is a cell, insert one cell at the time
      int numRows;
      int numCols;
      int visitedCells = 0;
      std::stack<Coordinates> visCoords;
      
};
#endif
