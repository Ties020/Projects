const { createColorConsole } = require('./ColorConsole');
const readline = require('readline');

//Create an interface for input and output
const rl = readline.createInterface(
{
  input: process.stdin,
  output: process.stdout,
});

function askColor() 
{
  rl.question('Please enter a color (red, blue, green): ', (color) =>  //Takes colour input as argument to pass to askMessage method
  {askMessage(color);});
}

function askMessage(color) 
{
  rl.question('Please enter a message: ', (message) => 
  {
    try 
    {
      const colorConsole = createColorConsole(color);
      colorConsole.log(message);
    } 
    catch (error) {console.error(error.message);} 
    
    finally {rl.close();}
    
  });
}

askColor();
