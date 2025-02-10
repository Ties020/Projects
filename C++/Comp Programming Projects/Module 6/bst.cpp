#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

class Coordinates
{
   public:
      Coordinates(int newX, int newY);
      int x, y;
};

Coordinates::Coordinates(int newX, int newY)
{
   x = newX;
   y = newY;
}

class Cell
{
   public:
      Cell();
      void printContent();
      int numofNode;
      bool hasNum;
};

Cell::Cell ()
{
  hasNum = false;
}

void Cell::printContent()  
{
   if(hasNum)
   {  
      if(numofNode > 9 && numofNode < 100)
      {
         std::cout<< "  " << numofNode;
      }
      else if(numofNode > 99 && numofNode < 1000)
      {
         std::cout<< " " << numofNode; 
      }
      else if(numofNode < 10 && numofNode >= 0)
      {
         std::cout<< "   " << numofNode; 
      }
      else if(numofNode < 0 && numofNode >-10)
      {
         std::cout<< "  " << numofNode;
      }
      else if(numofNode < -9 && numofNode >-100)
      {
         std::cout<< " " << numofNode;
      }
      else if (numofNode < -99 && numofNode >-1000)
      {
         std::cout << numofNode;
      }

   }
   else
   {
      std::cout << "    ";
   }
}

class Node
{
   public:
      int root;
      Node* rightp;
      Node* leftp;
      Node(int val) : root(val), leftp(nullptr), rightp(nullptr){};
   private:
};

class BST
{
    public:
        BST(){root = nullptr;};
        ~BST();
        void insertKey(int newKey);
        bool hasKey(int searchKey);
        int getHeight();
        std::vector<int> inOrder();
        void prettyPrint();
        
        
    private:
         void removeTree(Node* node);
         void insertKey(int newKey, Node *&node);
         bool hasKey(int searchKey, Node* node); 
         int getHeight(Node* node);
         void prettyPrint(int sizevec);
         void makeMatrix();
         std::vector<int> inOrder(Node* Node);
         std::vector<int> orderedvec;
         Node* root;
         std::vector<std::vector<Cell>> table;
         Cell& at(Coordinates coord);
         void fillTable(Node* node);
         int row = 0;
};

Cell& BST::at(Coordinates coord)  //makes you able to call cell by its coordinates
{
   return table.at(coord.y).at(coord.x); 
}
BST::~BST()
{
   removeTree(root);
}

void BST::makeMatrix()
{
   table = std::vector<std::vector<Cell>>(getHeight(), std::vector<Cell>(orderedvec.size()));   
}

void BST::removeTree(Node* node)
{
   if(node != NULL)
   {
      if(node->leftp != NULL)
      {
         removeTree(node->leftp);
      }
      if(node->rightp != NULL)
      {
         removeTree(node->rightp);
      }
      delete node;
   }
}

void BST::insertKey(int newKey)
{
   insertKey(newKey, root);
}

void BST::insertKey(int newKey, Node *&node) 
{
   if (node == nullptr) 
   {
      node = new Node(newKey);
   }
   else if(node->root < newKey)
   {
      insertKey(newKey, node->rightp);
   }
   else 
   {
      insertKey(newKey, node->leftp);
   }
}

std::vector<int> BST::inOrder()
{
   orderedvec.clear();
   return inOrder(root);
}

std::vector<int> BST::inOrder(Node* node){
   if(node != NULL)
   {
      if(node->leftp != NULL)
      {
         inOrder(node->leftp);
      }
      orderedvec.push_back(node->root);
      if(node->rightp != NULL)
      {  //checks if it is possible to go right in the tree
         inOrder(node->rightp);
      }

   }
   return orderedvec;
}

int BST::getHeight()
{
   return getHeight(root);
}

int BST::getHeight(Node* node)
{
    if (node == NULL) return 0;
    else 
    {
        int leftheight = getHeight(node->leftp);
        int rightheight = getHeight(node->rightp);
 
        if (leftheight > rightheight) 
        {
            return (leftheight + 1); //also include the top node
        }
        else 
        {
            return (rightheight + 1);
        }
    }
}

bool BST::hasKey(int searchkey)
{
   return hasKey(searchkey, root);
}

bool BST::hasKey(int searchkey, Node* node)
{
   if(node == nullptr)
   {
      return false;
   }
   if(searchkey == node->root)
   {
      return true;
   }
   if(searchkey < node->root)
   {
      hasKey(searchkey, node->leftp);
   }
   else 
   {
      hasKey(searchkey, node->rightp);
   }
   return false;
}

void BST::fillTable(Node* node)
{
   if(node != NULL)
   {
      int col;
      for(int i=0; i < inOrder().size();++i)
      {
         if(inOrder().at(i) == node->root)
         {
            col = i;
            break;
         }
      }
      Coordinates coord(col, row);
      at(coord).numofNode = node->root;
      at(coord).hasNum = true;
      row +=1;
      fillTable(node->leftp);
      fillTable(node->rightp);
      row -=1;
      }
   
}

void BST::prettyPrint()
{
   int sizevec = inOrder().size();
   makeMatrix();
   fillTable(root);
   prettyPrint(sizevec);
}

void BST::prettyPrint(int sizevec)
{
   for(int i=0;i< getHeight();++i)
   {
      for(int j=0; j< sizevec;++j) //print top wall first
      {
         std::cout<< "-----";
      }
      std::cout << "-";
      std::cout<<std::endl;

      for(int k=0; k< sizevec;++k) //print left wall and content of node
      {
         Coordinates coord(k,i);
         std::cout <<"|";
         at(coord).printContent();
         
      }
      std::cout << "|";  
      std::cout<<std::endl;
   }
   for(int j=0; j< sizevec;++j) //print bottom wall
   {
      std::cout << "-----";
   }
   std::cout << "-" << std::endl;

}

int main()
{
   int input,searchkey;
   BST t;
   std::cout << "Enter the numbers to be stored:";
   while(std::cin >> input)
   {
      t.insertKey(input);
   }
   std::cout <<"The numbers in sorted order: ";
   int sizevec = t.inOrder().size();
   if(sizevec == 0) return 0;
   for(int i=0; i <sizevec; ++i)
   {
      std::cout << t.inOrder().at(i)<<" ";
   }
   std::cout<<std::endl;
   t.prettyPrint(); 
   std::cout<<std::endl;
   return 0; 
}