class Logger 
{
    constructor(strategy) {this.strategy = strategy;}

    debug(message) {this.strategy.log(`Debug: ${message}`);}

    info(message) {this.strategy.log(`Info: ${message}`);}

    warn(message) {this.strategy.log(`Warn: ${message}`);}

    error(message) {this.strategy.log(`Error: ${message}`);}
}

module.exports = Logger;
