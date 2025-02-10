#include "parser/ast.h"
#include "shell.h"
#include <unistd.h>  
#include <sys/types.h>  //for wait functions
#include <sys/wait.h>    // ""
#include <stdlib.h>
#include<string.h>  
#include <stdio.h>  
#include <signal.h>


void interrupt_handler(){
    kill(0,SIGUSR1);  //this will stop the execution of the current process when ctrl C is pressed
}


void handlecmd(node_t *node){

    char *program = node->command.program;
    char **argv = node->command.argv;

    if(strcmp(program,"exit") == 0) exit(atoi(argv[1]));

    if(strcmp(program,"cd") == 0) {
        if(chdir(argv[1]) == -1) perror("Error changing directories");
    }
    
    
    else if(strcmp(program, "set") == 0){
        char * name = strtok(argv[1], "=");
        char * newval = strtok(NULL, "=");
        if(setenv(name,newval,1) == -1) perror("Error assigning variable");
    }

    else if(strcmp(program, "unset") == 0){
        char * name = argv[1];
        unsetenv(name);
    }

    else {
        int status;
        pid_t pid;
        pid = fork();

        if (pid == 0) {
            signal(SIGINT, interrupt_handler);   //would use sigaction instead since it's more specific, but somehow not compatible????
            if(execvp(program, argv) != 0) perror("Error during execution");
            exit(1); 
        } 
        else wait(&status);
    }
}

void initialize(void)
{ 
    /* This code will be called once at startup */
    if (prompt) prompt = "vush$ ";
}

void run_command(node_t *node)
{   
    signal(SIGINT, SIG_IGN);
    
    if (node->type == NODE_COMMAND) {      
        handlecmd(node);
    }

    if(node->type == NODE_DETACH){
        pid_t pid;
        pid = fork();

        if (pid == 0) {
            run_command(node->detach.child);
            exit(1);
        } 

        if(waitpid(-1, NULL, WNOHANG) == -1)    //prevent zombie child
        {
            perror("Error while waiting for child process to terminate");
        }
        //not waiting makes it possible for the parent to keep executing 
        //new commands without being blocked by having to wait for the detached command to finish.
    }

    if(node->type == NODE_SUBSHELL){
        int status;
        pid_t pid;
        pid = fork();

        if (pid == 0) {
            run_command(node->subshell.child);
            exit(1);
        } 
        else wait(&status);
    }

    if (node->type == NODE_SEQUENCE) {  
        run_command(node->sequence.first);
        run_command(node->sequence.second);
    } 

    if(node->type == NODE_PIPE){
        int fd[2];
        if(pipe(fd) == -1) perror("Error creating pipe"); 

        int first_id = fork();
        if (first_id == 0){           
            if(close(fd[0]) != 0) perror("error closing read end of pipe");
            if(close(1) != 0) perror("error closing file");
            dup(fd[1]);
            execvp(node->pipe.parts[0]->command.program, node->pipe.parts[0]->command.argv); 
        }

        int second_id = fork();
        if ( second_id == 0 ) {
            if(close(fd[1]) != 0) perror("error closing write end of pipe");
            if(close(0) != 0) perror("error closing file");
            dup(fd[0]);
            execvp(node->pipe.parts[1]->command.program, node->pipe.parts[1]->command.argv);
        }

        close(fd[0]);
        close(fd[1]);
        waitpid(first_id, NULL, 0);
        waitpid(second_id, NULL, 0);
    }

    if (prompt) {
        prompt = "vush$ ";
    }  
}

