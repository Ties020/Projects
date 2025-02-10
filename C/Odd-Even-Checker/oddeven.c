#include <stdio.h>
#include <ctype.h>

int main() {
    char myNum;

    printf("Type a number: \n");
    scanf(" %c", &myNum); //Space before %c skips any whitespace or newline characters

    while (!isdigit(myNum)) { 
        printf("That's not a number. Type a number: \n");
        scanf(" %c", &myNum); 
    }

    printf("Your number is: %c\n", myNum);

    if(myNum % 2 == 0){
        printf("Even \n");
    }

    else{
        printf("Odd \n");
    }

    return 0;
}