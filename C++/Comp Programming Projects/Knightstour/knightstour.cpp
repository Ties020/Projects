#include <iostream>
#include <vector> 
#include <sstream>
#include <string>
#include <cmath>

const std::vector <int> letterMove{2,1,-1,-2,-2,-1,1,2};  //represent all possible moves
const std::vector <int> integerMove{1,2,2,1,-1,-2,-2,-1};

bool isvalid(int nextlet, int nextnum, std::vector<std::vector <int>> boardVector, const int boardSize) //checks whether coordinates of the next move are inside the chess board and not visited yet
{  
   if (nextlet >= 0 && nextlet < boardSize && nextnum >= 0 && nextnum < boardSize && boardVector.at(nextlet).at(nextnum) == -1)
   {
         return true;
   }
   return false;
}

bool findKnightsTour(std::vector<std::vector<int>>& boardVector, int& startCoordLetterAsInt, int& startCoordInt, const int endCoordLetterAsInt, const int endCoordInt, const int boardSize)
{
   int nextlet, nextnum;
   if (startCoordLetterAsInt == endCoordLetterAsInt && startCoordInt == endCoordInt)
   {
      return true;
   }
   
   else
   {
      for (int i = 0; i < 8; ++i) //loop through all 8 possible moves
      {
         nextlet = startCoordLetterAsInt + letterMove.at(i);
         nextnum = startCoordInt + integerMove.at(i);          //these are coordinates of the next square after a move
         if (isvalid(nextlet, nextnum, boardVector, boardSize))
         {
            boardVector.at(nextlet).at(nextnum) = boardVector.at(startCoordLetterAsInt).at(startCoordInt) + 1;  //mark the visited square with the number of the previous square + 1 
            bool success = findKnightsTour(boardVector, nextlet, nextnum, endCoordLetterAsInt, endCoordInt, boardSize); //recursively call function findKnightsTour to find tour
            if (success) // if tour is found, so if start coordinates are the same as the end coordinates return true
            {
               return true;
            }
            boardVector.at(nextlet).at(nextnum) = -1; //remove the mark that was placed when square was visited if move didn't lead to valid tour
         }
      }
   }
   return false;

}

void knightsTour(int boardSize, int startCoordLetterAsInt, int startCoordInt, int endCoordLetterAsInt, int endCoordInt)
{
   std::vector<std::vector<int>> boardVector(boardSize, std::vector<int> (boardSize, -1)); //this is the 2D vector, the chess board, with every square initially having value -1 
   boardVector.at(startCoordLetterAsInt).at(startCoordInt) = 0; //start square is marked as visited
   
   if (findKnightsTour(boardVector, startCoordLetterAsInt, startCoordInt, endCoordLetterAsInt, endCoordInt, boardSize))
   {
      int checkedAs = 0;
      for (int i=0; i < boardSize;i++)      //loop that goes through whole board and checks if square has been visited and outputs it, it is in order by using checkedAs as variable that compares value of square which is when they were visited
       {
         if (checkedAs == boardVector.at(endCoordLetterAsInt).at(endCoordInt))
         {
            char endLetter = endCoordLetterAsInt + 97;  //transform coordinate of the last visited square that was an integer back to a letter like it was in the input.
            std::cout << endLetter << endCoordInt+1 << std::endl;
            break;
         }         
         for (int j=0; j< boardSize;j++)
         {
            if (boardVector.at(i).at(j) == checkedAs)
            {
               char charm = i + 97;
               std::cout << charm << j+1 <<" ";
               checkedAs += 1;
               i = 0;
               j = 0;
            }
         }
      }
   }
   else if(!(findKnightsTour(boardVector, startCoordLetterAsInt, startCoordInt, endCoordLetterAsInt, endCoordInt, boardSize)))
   {
      std::cout << "could not find a knight's tour";
   }
}

void checkinput(std::string boardSizestr, std::string startCoord, std::string endCoord)
{
   char startCoordLetter, endCoordLetter;
   int startCoordInt, endCoordInt, boardSize;;
 
   
   std::istringstream boardSizestrStream(boardSizestr);
   boardSizestrStream >> boardSize;

   std::istringstream startCoordStream(startCoord);
   startCoordStream >> startCoordLetter;
   startCoordStream >> startCoordInt;

   int startCoordLetterAsInt = startCoordLetter - 97; //make character have an integer value, related to ASCII table

   std::istringstream endCoordStream(endCoord);
   endCoordStream >> endCoordLetter;
   endCoordStream >> endCoordInt;

   int endCoordLetterAsInt = endCoordLetter - 97;
   
   if (startCoordInt > boardSize || endCoordInt > boardSize || (startCoordLetterAsInt + 1)> boardSize || (endCoordLetterAsInt +1) > boardSize || boardSize > 26) //error handling, for if coordinates are outside board and max board size is 26
   { 
      throw std::runtime_error("invalid parameter list");  
   }
   startCoordInt -= 1; 
   endCoordInt -= 1;
   knightsTour(boardSize, startCoordLetterAsInt, startCoordInt , endCoordLetterAsInt, endCoordInt);
}

int main(int argc, char* argv[]) 
{ 
   try
   {
      if (argc == 4) //command parameter is good, now check whether the coordinates are within the board size
      {
      std::string boardSizestr = argv[1];
      std::string startCoord = argv[2];
      std::string endCoord = argv[3];
      checkinput(boardSizestr,startCoord,endCoord);
      }
     else if (argc != 4)
      { 
         throw std::runtime_error("invalid parameter list");  
      } 
   }
   catch (std::runtime_error &excpt)  
   { 
         std::cout << excpt.what() << std::endl; 
   }
   return 0;
}