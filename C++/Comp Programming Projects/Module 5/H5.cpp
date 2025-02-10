#include <iostream>
#include <vector> 
#include <sstream>
#include <string>
#include <cmath>
#include <stdexcept> 
#include "Maze.h"
#include "Cell.h"
#include "Coordinates.h"

int main(int argc, char* argv[]) 
{
   try
   {
      if(argc == 3 || argc == 4)
      {
           //still need error-handling for input other than integers
         int rowsin = std::stoi(argv[1]);
         int colsin = std::stoi(argv[2]);

         if (argc == 4) 
         {
            int seed = std::stoi(argv[3]);
            srand(seed);
         }
         else 
         {
            srand(time(0));
         }

         Maze m(rowsin,colsin);
         Coordinates coord(0,0);
         Coordinates endcoord(colsin-1,rowsin-1);
         m.genRandomMaze(coord);  //generate random maze
         m.allCellsVisistedToFalse();  //set all cells to unvisited
         m.findPath(coord,endcoord);
         m.printPath();
         return 0;
         
      }
      else
      {
         throw std::runtime_error("invalid parameter list");
      }
   }
   catch (std::runtime_error &excpt)  
   { 
      std::cout << excpt.what() << std::endl; 
   }
   catch (std::out_of_range &excpt)  
   { 
      std::cout << excpt.what() << std::endl; 
   }
   return 0;
}