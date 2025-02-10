#include <iostream>
#include <vector>
#include <limits>





class Node{
   public:
      int root;
      Node* rightp;
      Node* leftp;
      Node(int val) : root(val), leftp(nullptr), rightp(nullptr){};
   private:
};

class BST{
    public:
        BST(){};
        ~BST();
        void insertKey(int newKey);
        bool hasKey(int searchKey);
        int getHeight();
        std::vector<int> inOrder();
        
        
    private:
         void removeTree(Node* node);
         void insertKey(int newKey, Node *&node);
         bool hasKey(int searchKey, Node* node); 
         int getHeight(Node* node);
         std::vector<int> inOrder(Node* Node);
         std::vector<int> orderedvec;
         Node* root;
};
BST::~BST()
{
   removeTree(root);
}

void BST::removeTree(Node* node)
{
   if(node != NULL){
      if(node->leftp != NULL){
         removeTree(node->leftp);
      }
      if(node->rightp != NULL){
         removeTree(node->rightp);
      }
      delete node;
   }
}

void BST::insertKey(int newKey){
   insertKey(newKey, root);
}

void BST::insertKey(int newKey, Node *&node) {
   if (node == nullptr) {
      node = new Node(newKey);
   }
   else if(node->root < newKey){
      insertKey(newKey, node->rightp);
   }
   else {
      insertKey(newKey, node->leftp);
   }
}

std::vector<int> BST::inOrder(){
   return inOrder(root);
}

std::vector<int> BST::inOrder(Node* node){
   if(node != NULL){
      if(node->leftp != NULL){
         inOrder(node->leftp);
      }
      orderedvec.push_back(node->root);
      if(node->rightp != NULL){  //checks if it is possible to go right in the tree
         inOrder(node->rightp);
      }

   }
   return orderedvec;
}

int BST::getHeight(){
   return getHeight(root);
}

int BST::getHeight(Node* node){
    if (node == NULL)
        return 0;
    else {
        int leftheight = getHeight(node->leftp);
        int rightheight = getHeight(node->rightp);
 
        if (leftheight > rightheight) {
            return (leftheight + 1); //also include the top node
        }
        else {
            return (rightheight + 1);
        }
    }
}

bool BST::hasKey(int searchkey){
   return hasKey(searchkey, root);
}

bool BST::hasKey(int searchkey, Node* node){
   if(node == nullptr){
      return false;
   }
   if(searchkey == node->root ){
      return true;
   }
   if(searchkey < node->root){
      hasKey(searchkey, node->leftp);
   }
   else {
      hasKey(searchkey, node->rightp);
   }
   return false;
}

int main(){
   int input,searchkey;
   BST t;

   std::cout << "Enter the numbers to be stored (end with a letter):";
   while(std::cin >> input){
      t.insertKey(input);
   }
   if(std::cin.fail()){
      std::cin.clear();                                                    //clears input stream, else the program will use 0 as the searchkey
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
   }
   std::cout <<std::endl;
   std::cout <<"Which number do you want to look up?";
   std::cin >> searchkey;
   std::cout << std::endl;
   
   if (t.hasKey(searchkey)){
      std::cout << searchkey <<" is in the tree: yes"<<std::endl; 
   }
   else if(!(t.hasKey(searchkey))){
      std::cout << searchkey <<" is in the tree: no"<< std::endl; 
   }
   
   std::cout <<"The numbers in sorted order: ";
   int sizevec = t.inOrder().size();
   for(int i=0; i <sizevec; ++i){
      std::cout << t.inOrder().at(i)<<" ";
   }
   std::cout<<std::endl;

   std::cout <<"Height of the tree: " << t.getHeight()<< std::endl;
   //t.~BST();
   return 0;
   
}
// take median as root normally, in this case the first number is the root. 