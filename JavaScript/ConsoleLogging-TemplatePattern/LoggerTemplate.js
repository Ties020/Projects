class LoggerTemplate 
{
    log(type, message) {this.writeLog(message);} //The overridden method is called and executed by the sublass that implemented the writeLog() method

    writeLog(message) {throw new Error("Subclasses must implement the writeLog method.");}

    debug(message) {this.log('debug', `Debug: ${message}`);}

    info(message) {this.log('info', `Info: ${message}`);}

    warn(message) {this.log('warn', `Warn: ${message}`);}

    error(message) {this.log('error', `Error: ${message}`);}
}

module.exports = LoggerTemplate;
