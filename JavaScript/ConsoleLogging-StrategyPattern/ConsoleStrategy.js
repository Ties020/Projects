const LogStrategy = require('./LogStrategy');
class ConsoleStrategy extends LogStrategy {
    log(message) {console.log(message);}
}
module.exports = ConsoleStrategy;
