class ColorConsole 
{log() {}}

class RedConsole extends ColorConsole 
{log(message) {console.log(`\x1b[31m${message}\x1b[0m`);} }

class BlueConsole extends ColorConsole 
{log(message) {console.log(`\x1b[34m${message}\x1b[0m`);}}

class GreenConsole extends ColorConsole 
{log(message) {console.log(`\x1b[32m${message}\x1b[0m`);}}

//Factory function to create console color instances
function createColorConsole(color) 
{
  switch (color.toLowerCase()) 
  {
    case 'red':
      return new RedConsole();
    case 'blue':
      return new BlueConsole();
    case 'green':
      return new GreenConsole();
    default:
      throw new Error(`Unknown color: ${color}`);
  }
}

module.exports = { createColorConsole };
