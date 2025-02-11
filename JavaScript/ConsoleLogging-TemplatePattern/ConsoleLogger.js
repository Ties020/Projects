const LoggerTemplate = require('./LoggerTemplate');

class ConsoleLogger extends LoggerTemplate 
{
    writeLog(message) {console.log(message);} //Overrides the abstract writeLog() method to log the message to the console
                                             //Log() in overridden method doesn't call log() from LoggerTemplate again since it uses default log() method related to the console object, not the ConsoleLogger/LoggerTemplate object
}

module.exports = ConsoleLogger;
