#include <stdio.h>
#include <ctype.h>
#include <string.h>


int main() {
    char operator;
    double num1;
    double num2;
    double result;

    printf("Enter operator (+, -, *, or / ): ");
    scanf("%c",  &operator);

    printf("Enter num1: ");
    scanf("%lf", &num1);

    printf("Enter num2: ");
    scanf("%lf", &num2);

    switch (operator)
    {
        case '+':
            result = num1 + num2;
            break;
        case '-':
            result = num1 - num2;
            break;
        case '*':
            result = num1 * num2;
            break;
        case '/':
            if (num2 != 0) {
                result = num1 / num2;
            } 
            else {
                printf("Error: Division by zero is not allowed.\n");
                return 1;
            }
            break;
        default:
            printf("%c is not valid \n", operator);
            return 1;
    }
    
    printf("Result: %.2lf\n", result); 

    return 0;
}