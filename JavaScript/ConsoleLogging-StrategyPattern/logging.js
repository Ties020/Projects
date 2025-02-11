const Logger = require('./Logger');  //Import the classes
const ConsoleStrategy = require('./ConsoleStrategy');

const main = () => {
    const consoleLogger = new Logger(new ConsoleStrategy()); //I instantiate the logging component and the messages are sent to the console
    consoleLogger.debug("This is a debug message"); 
    consoleLogger.info("This is an info message");
    consoleLogger.warn("This is a warning message");
    consoleLogger.error("This is an error message");
    //The same could've been done for a fileStrategy, but then the log method would have added the message to a text file instead of to the console like I did
};

main();
