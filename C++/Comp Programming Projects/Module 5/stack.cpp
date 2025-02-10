#include <iostream>
#include <limits>
#include <vector>
#include <sstream>
class Stack {
    public:
        Stack(){};
        bool isEmpty() const;
        int top() const;
        int pop();
        void push(int i);
        std::string toString() const;
    private:
        std::vector<int> elements;
};

bool Stack::isEmpty() const {
    if (elements.size() == 0){
        return true;
    }
    return false;
}

void Stack::push(int i) {   //good
    this->elements.push_back(i);
} 

int Stack::top() const {  //good
    if(elements.size() == 0){
        throw std::runtime_error("stack is empty");
    }
    return elements.at(elements.size()-1);
}

int Stack::pop() { //good
    if(elements.size() == 0){
        throw std::runtime_error("stack is empty");
    }
    int j = elements.at(elements.size()-1);
    elements.erase(elements.end()-1);
    return j;
}

std::string Stack::toString() const{
  std::string s;
  if (elements.size() > 0){
   for(int i= elements.size()-1; i >=0; --i){
      std::stringstream ss;  
      ss<< elements.at(i);  
 
      s+= ss.str();
      s.push_back(',');
   }
   s.erase(s.end()-1);
  }
  return "[" + s +"]";
}

int main() {
    Stack stack;
    
    while (true) {
        try {
            std::cout << "stack> ";
            
            std::string command;
            std::cin >> command;
            
            if ( (command.compare("end") == 0) || std::cin.eof() ){
                break;
            } else if (command.compare("top") == 0) {
                std::cout << stack.top() << std::endl;
            } else if (command.compare("pop") == 0) {
                std::cout << stack.pop() << std::endl;
            } else if (command == "push") {
                if ( std::cin.eof() ) break;
                int i;
                std::cin >> i;
                bool failed = std::cin.fail();
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                if ( failed ){
                  throw std::runtime_error("not a number");
                }
                stack.push(i);
            } else if ( command.compare("list") == 0){
                std::cout << stack.toString() << std::endl;;
            } else {
                throw std::runtime_error("invalid command");
            }
            std::cout <<std::endl;
        } catch (std::runtime_error& e) {
            std::cout << "error: " << e.what() << std::endl;
        }
    }
    
    return 0;
}