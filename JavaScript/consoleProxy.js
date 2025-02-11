const consoleProxy = new Proxy(console, 
{
    get(target, property) //Console is target and property is the name of the property, like log
    {
        if (['log', 'error', 'debug', 'info'].includes(property)) 
        {
            return function (...args) //Args are all arguments that were originally passed to the console, function is returned and used instead of default function like log
            {
                const timestamp = new Date().toISOString();
                target[property](`[${timestamp}]`, ...args); //Prepend the current timestamp to the message, target[property] is console object with property, like console.log
            };
        }
        return target[property];
    },
});
  

consoleProxy.log('Hello World');   
consoleProxy.error('Error occurred');
consoleProxy.debug('Debugging'); 
consoleProxy.info('Info');   
  