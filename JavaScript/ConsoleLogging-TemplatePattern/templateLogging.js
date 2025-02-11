const ConsoleLogger = require('./ConsoleLogger');

const logger = new ConsoleLogger();

logger.debug("This is a debug message");  //Debug() is called from the instantiated subclass, the debug() method then calls the log() function which calls the writeLog() function that is overridden by logger
logger.info("This is an info message");
logger.warn("This is a warning message");
logger.error("This is an error message");
