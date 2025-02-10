#include "Cell.h"
#include "Coordinates.h"
#include <iostream>

Cell::Cell ()
{
   topWall = true;
   bottomWall = true;
   leftWall = true;
   rightWall = true;
   visited = false;
}

void Cell::printTop() 
{
   if (topWall) std::cout << "+---";
   else std::cout << "+   ";
}

void Cell::printLeft() 
{
   if (leftWall) std::cout << "|";
   else std::cout << " ";
}

void Cell::printContent()  
{
   if (visited) std::cout << " . ";
   else std::cout << "   ";
}