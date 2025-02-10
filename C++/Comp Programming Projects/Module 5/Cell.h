#ifndef Cell_H
#define Cell_H

class Cell
{
   public:
      Cell(); 
      void printTop();
      void printLeft();
      void printContent();

      bool visited;
      bool topWall, bottomWall, leftWall, rightWall;  
};

#endif