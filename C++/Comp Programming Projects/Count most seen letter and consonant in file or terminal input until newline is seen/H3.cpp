#include <iostream> 

#include <sstream> 

#include <string> 

#include <fstream> 

#include <stdexcept> 

#include <vector> 

#include <algorithm> 

#include <cctype> 

  

bool alphabetticallyfirst(char& currLetter, char& previousLetter) 

{ 
   if (previousLetter > currLetter) 
   { 
      return true; 
   } 
   else 
   { 
      return false; 
   } 
} 

void consonantcounter(std::vector <char> consonantVector) //this will count and output most seen consonent 
{ 
   int i,amountCons; 
   char consonantMost,consonant; 
 
   consonantMost = consonantVector.at(0); 
   amountCons= std::count(consonantVector.begin(), consonantVector.end(), consonantVector.at(0)); 

   for (i=1; i < consonantVector.size();++i) 
   { 
      if(std::count(consonantVector.begin(), consonantVector.end(), consonantVector.at(i)) > amountCons) 
      { 
         amountCons = std::count(consonantVector.begin(), consonantVector.end(), consonantVector.at(i)); 
         consonantMost = consonantVector.at(i); 
      } 
      else if ((std::count(consonantVector.begin(), consonantVector.end(), consonantVector.at(i))) == amountCons) 
      { 
         consonant = consonantVector.at(i); 
         if (alphabetticallyfirst(consonant, consonantMost)) 
         { 
            consonantMost = consonantVector.at(i); 
         } 
     } 
   } 

 
   if (isupper(consonantMost)) 
   { 
      consonantMost = tolower(consonantMost); 
      std::cout <<"Most frequent consonant: " << consonantMost << " (" << amountCons << " times)" <<std::endl; 
   } 
   else  
   { 
      std::cout <<"Most frequent consonant: " << consonantMost << " (" << amountCons << " times)" <<std::endl; 
   } 
} 

void vowelcounter(std::vector <char> vowelVector)  
{ 
   int i,amountVowel; 
   char vowelMost,vowel; 
    
   vowelMost = vowelVector.at(0); 
   amountVowel= std::count(vowelVector.begin(), vowelVector.end(), vowelVector.at(0)); 

   for (i=1; i < vowelVector.size();++i) 
   { 
      if(std::count(vowelVector.begin(), vowelVector.end(), vowelVector.at(i)) > amountVowel) 
      { 
         amountVowel = std::count(vowelVector.begin(), vowelVector.end(), vowelVector.at(i)); 
         vowelMost = vowelVector.at(i); 
      } 
      else if (std::count(vowelVector.begin(), vowelVector.end(), vowelVector.at(i)) == amountVowel) 
      { 
         vowel = vowelVector.at(i); 
         if (alphabetticallyfirst(vowel, vowelMost)) 
         { 
            vowelMost = vowelVector.at(i); 
         } 
      } 
   } 
 
   if (isupper(vowelMost)) 
   { 
      vowelMost = tolower(vowelMost); 
      std::cout <<"Most frequent vowel: " << vowelMost << " (" << amountVowel << " times)" <<std::endl; 
   } 
   else  
   { 
      std::cout <<"Most frequent vowel: " << vowelMost << " (" << amountVowel << " times)" <<std::endl; 
   } 
} 

void vowelcheck(std::vector <char>& allVector) // this will put a vowel in one vector and consonents in another to make it possible to count them separately 
{  
   int i; 
   char letter; 
   std::vector <char> vowelVector;  
   std::vector <char> consonantVector;  
  
   for(i=0; i <allVector.size(); ++i) 
   { 
      letter = allVector.at(i); 
      if (letter == 'a' || letter == 'e' || letter == 'i' || letter == 'o' || letter == 'u') 
      { 
         vowelVector.push_back(letter); 
      } 
      else  
      { 
         consonantVector.push_back(letter); 
      } 
   } 
   if (vowelVector.size() == 0) 
   { 
      std::cout <<"Most frequent vowel: a (0 times)"<<std::endl; 
      consonantcounter(consonantVector); 
   } 
   else if (consonantVector.size() == 0) 
   { 
      vowelcounter(vowelVector); 
      std::cout <<"Most frequent consonant: b (0 times)"<<std::endl; 
   } 
   else 
   { 
      vowelcounter(vowelVector); 
      consonantcounter(consonantVector); 
   } 
} 

void printletter(char letterMost, int amountLetter) //this function will print the most seen letter in the input overall. 
{ 
   if(isupper(letterMost)) 
   { 
      letterMost = tolower(letterMost); 
      std::cout <<"Most frequent letter, overall: " << letterMost << " (" << amountLetter << " times)"; 
   } 
   else 
   { 
      std::cout <<"Most frequent letter, overall: " << letterMost << " (" << amountLetter << " times)"; 
   } 
} 
  
void totalcounter( std::vector <char>& mainVector) // this function will count most seen letter of all 
{   
   int i,amountLetter; 
   char letterMost,letter; 
   letterMost = mainVector.at(0); 
   amountLetter = std::count(mainVector.begin(), mainVector.end(), mainVector.at(0)); 
  
   for (i=1; i < mainVector.size();++i) 
   { 
      if(std::count(mainVector.begin(), mainVector.end(), mainVector.at(i)) > amountLetter) 
      { 
         amountLetter = std::count(mainVector.begin(), mainVector.end(), mainVector.at(i)); 
         letterMost = mainVector.at(i); 
      } 
      else if (std::count(mainVector.begin(), mainVector.end(), mainVector.at(i)) == amountLetter) 
      { 
         letter = mainVector.at(i); 
         if (alphabetticallyfirst(letter, letterMost)) 
         { 
            letterMost = mainVector.at(i); 
         } 
      } 
   } 
  
   vowelcheck(mainVector); 
   printletter(letterMost,amountLetter); 
} 
  
void allempty(const int& amountVectors)  //this will output the following if no input has been given at all 
{  
   if (amountVectors == 0) 
   { 
      std::cout << "Most frequent vowel: a (0 times)" <<std::endl; 
      std::cout << "Most frequent consonant: b (0 times)" <<std::endl; 
      std::cout << "Most frequent letter, overall: a (0 times)" <<std::endl; 
   } 
} 

void readfile(std::ifstream& fileLetters) //this function takes every character apart, only letters, append these to the main vector 
{   
   char x; 
   std::vector <char> fileVector;  
  
   while ( fileLetters >> x) 
   { 
      if (isalpha(x) && isupper(x)) 
      { 
         x = tolower(x); 
         fileVector.push_back(x); 
      } 
      else if (isalpha(x)) 
      { 
         fileVector.push_back(x); 
      } 
   } 
    
   if(fileVector.size() == 0) 
   { 
      allempty(0); 
   } 
   else 
   { 
      totalcounter(fileVector); 
   } 
} 
  
void readcin(std::string& allLetters) //takes input from cin and puts it in a vector 
{  
   int i; 
   std::vector <char> vectorcin; 
   for (i=0; i< allLetters.size(); ++i) 
   { 
      if (isalpha(allLetters.at(i)) && isupper(allLetters.at(i))) 
      { 
         allLetters.at(i) = tolower(allLetters.at(i)); 
         vectorcin.push_back(allLetters.at(i)); 
      } 
      else if (isalpha(allLetters.at(i))) 
      { 
         vectorcin.push_back(allLetters.at(i)); 
      } 
   } 
   if(vectorcin.size() == 0) 
   { 
      allempty(0); 
   } 
   else 
   { 
   totalcounter(vectorcin); 
   } 
} 

int main(int argc, char* argv[]) 
{ 
   std::ifstream inFS; 
   std::string inputLine; 
   std::string wholeInput; 

 
   try 
   { 
      if (argc == 2) 
      { 
         inFS.open(argv[1]); 
         if (!inFS.is_open()) 
         { 
            throw std::runtime_error("cannot open input file "); 
         } 
         else 
         { 
            readfile(inFS); 
         } 
      } 
      else if (argc == 1)  
      { 
         while(std::getline(std::cin, inputLine)) 
         { 
            if (inputLine.empty()) 
            { 
               break; 
            } 
            else 
            { 
               wholeInput += inputLine; 
            } 
 
         } 
         readcin(wholeInput); 
      } 
      else 
      { 
         std::cout << "Cannot handle parameter list"; 
      } 
   } 
   catch (std::runtime_error &excpt)  
   { 
         std::cout << excpt.what() << argv[1] << std::endl; 
   } 
} 