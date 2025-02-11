const http = require('http');
const url = require('url');
const querystring = require('querystring');
const fs = require('fs');
const path = require('path');

let books = [
    "Design Patterns: Elements of Reusable Object-Oriented Software: $59.99",
    "Patterns of Enterprise Application Architecture: $47.99",
    "Node.js Design Patterns: $39.99"
];

let bookPrices = {};
bookPrices["Design Patterns: Elements of Reusable Object-Oriented Software"] = 59.99;
bookPrices["Patterns of Enterprise Application Architecture"] = 47.99;
bookPrices["Node.js Design Patterns"] = 39.99;

let cart = {};

const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const method = req.method;

    //Handle the image file by getting the path and reading the data from that file
    if (parsedUrl.pathname === '/shoppingcart.png') {
        const filePath = path.join(__dirname, 'shoppingcart.png');
        fs.readFile(filePath, (err, data) => 
        {
            if (err) 
            {
                res.writeHead(404, { 'Content-Type': 'text/plain' });
                res.end('Image Not Found');
                return;
            }
            res.writeHead(200, { 'Content-Type': 'image/png' });
            res.end(data); //Send the image/file back, thus displaying it
        });
        return;
    }

    //Create MainMenu screen and handle requests that were made there
    if (parsedUrl.pathname === '/MainMenu' && method === 'GET') 
    {


        res.writeHead(200, { 'Content-Type': 'text/html' });

        let htmlContent = `
        <html><head><title>Book store</title>
        <style>
            body { font-family: Arial, sans-serif; }
            h1, h2 { text-align: center; }
            .cart-button { position: absolute; top: 10px; right: 10px; }
            .book-list { width: 60%; margin: 0 auto; list-style: none; padding: 0; }
            .book-item { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; }
            .book-title { font-size: 1.2em; margin-bottom: 5px; }
            .book-price { color: #007BFF; font-weight: bold; }
        </style>
        </head><body>
        <h1>Welcome to SOEN387 Book Store</h1>
        <h2>Available books:</h2>
        <ul class='book-list'>
        `;

        for (let i = 0; i < books.length; i++) 
        {
            book = books[i];
            const parts = book.split(": ");
            if (i == 0) 
            { 
                bookName = parts[0] + ": " + parts[1];
                bookPrice = parts[2];
            } 
            
            else 
            {
                bookName = parts[0]; 
                bookPrice = parts[1];
            }

            htmlContent += `
            <li class='book-item'>
                <span class='book-title'>${bookName}</span><br/>
                <span class='book-price'>${bookPrice}</span><br/>
                <form action='/MainMenu' method='post'>
                    <input type='number' name='quantity' min='1' value='1' />
                    <input type='hidden' name='bookName' value='${bookName}' />
                    <input type='submit' value='Add to cart' />
                </form>
            </li>`;
        };

        htmlContent += `
        <div class='cart-button'>
            <a href='/CartScreen'>
                <img src='/shoppingcart.png' alt='Cart' width='30' height='30' style='vertical-align: middle;' />
                Items in Cart
            </a>
        </div>
        </ul></body></html>`;

        res.end(htmlContent);
    } 
    
    else if (parsedUrl.pathname === '/MainMenu' && method === 'POST') 
    {
        //Handle adding books to cart

        //Receive all the data from the post request and add it to body string
        let body = '';
        req.on('data', chunk => 
        {
            body += chunk.toString();
        });

        //After all data has been received, add the book to the cart hashmap
        req.on('end', () => 
        {
            const postData = querystring.parse(body);
            const { bookName, quantity } = postData;

            if (!req.cart) req.cart = {};

            const quantityBook = parseInt(quantity, 10);
            cart[bookName] = (cart[bookName] || 0) + quantityBook; 

            //Redirect back to MainMenu
            res.writeHead(302, { Location: '/MainMenu' });
            res.end();
        });
    } 
    
    else if (parsedUrl.pathname === '/CartScreen' && method === 'GET') 
    {

        res.writeHead(200, { 'Content-Type': 'text/html' });

        let cartContent = `
        <html><head><title>Cart</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; }
          table { margin: 0 auto; width: 60%; border-collapse: collapse; }
          .checkout-button { position: absolute; top: 10px; right: 10px; }
        </style></head><body>
        <h1>Your Cart</h1>
        `;

        if (Object.keys(cart).length > 0) 
        {
            cartContent += `
              <form action='/CartScreen' method='post'>
              <table>
                <tr><th>Type</th><th>Price</th><th>Quantity</th><th>Total</th><th>Remove</th></tr>`;
      
                for (const [bookName, quantity] of Object.entries(cart)) 
                {
                    const price = bookPrices[bookName] || 0.0;  
                    const total = price * quantity;
      
              cartContent += `
                <tr>
                  <td>${bookName}</td>
                  <td>$${price.toFixed(2)}</td>
                  <td><input type='number' name='quantity_${bookName}' value='${quantity}' min='1' required></td>
                  <td>$${total}</td>
                  <td><button type='submit' name='remove' value='${bookName}'>Remove</button></td>
                </tr>
                <tr><td colspan='5' style='text-align:center;'>
                  <button type='submit' name='update' value='${bookName}'>Update Quantity</button>
                </td></tr>
                `;
                }
      
            cartContent += `</table></form>`;
        } 
          
        else cartContent += `<p>Your cart is empty!</p>`;
      
        cartContent += `
        <a href='/MainMenu'>Back to Bookstore</a>
        <form action='/CartScreen' method='post'>
            <div class='checkout-button'>
            <input type='submit' name='checkout' value='Checkout' />
            </div>
        </form></body></html>
        `;
    
        res.end(cartContent);
    } 

    else if (parsedUrl.pathname === '/CartScreen' && method === 'POST') 
    {

        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });

        req.on('end', () => 
        {
            const postData = querystring.parse(body);

            //Handle removing  book
            const bookToRemove = postData.remove;  //Extract bookname related to passed remove field
            if (bookToRemove) delete cart[bookToRemove];

            //Handle updating book's quantity
            const updatedBook = postData.update;
            if (updatedBook) 
            {
                const quantityParam = postData[`quantity_${updatedBook}`];
                if (quantityParam) 
                {
                    const newQuantity = parseInt(quantityParam, 10);
                    if (newQuantity > 0) cart[updatedBook] = newQuantity;
                }
            }

            //Handle checkout
            if (postData.checkout) 
            {
                cart = {};
                res.writeHead(302, { Location: '/CheckoutComplete' });
                return res.end();
            }

            res.writeHead(302, { Location: '/CartScreen' });
            res.end();
        });
    } 

    else if (parsedUrl.pathname === '/CheckoutComplete' && method === 'GET') 
    {
        let checkoutContent = `
        <html><head><title>Checkout Complete</title>
        <style>
        body { font-family: Arial, sans-serif; text-align: center; padding-top: 50px; }
        </style></head><body>
            <h1>Checkout Complete</h1>
            <a href='/MainMenu'>Back to Bookstore</a>
        </body></html>
        `;
        res.end(checkoutContent);
    } 

    
    else 
    {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Page Not Found');
    }
});

//Start server on port 3000
server.listen(3000, () => {
    console.log('Server running at port 3000');
});
