import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@WebServlet("/MainMenuServlet") // Place it here
public class MainMenuServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;
    
    /**
     * @see HttpServlet#HttpServlet()
     */
    public MainMenuServlet() {
        super();
    }

    /**
     * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse response)
     */
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/html");
        
        List<String> books = new ArrayList<>();
        books.add("Design Patterns: Elements of Reusable Object-Oriented Software: $59.99");
        books.add("Patterns of Enterprise Application Architecture: $47.99");
        books.add("Node.js Design Patterns: $39.99");

        //Prepare HTML content as a string
        StringBuilder htmlContent = new StringBuilder();
        htmlContent.append("<html><head><title>Book store</title>")
                   .append("<style>")
                   .append("body { font-family: Arial, sans-serif; }")
                   .append("h1, h2 { text-align: center; }")
	               .append(".cart-button { position: absolute; top: 10px; right: 10px; }") 
                   .append(".book-list { width: 60%; margin: 0 auto; list-style: none; padding: 0; }")
                   .append(".book-item { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; }")
                   .append(".book-title { font-size: 1.2em; margin-bottom: 5px; }")
                   .append(".book-price { color: #007BFF; font-weight: bold; }")
                   .append("</style>")
                   .append("</head>")
                   .append("<body>")
                   .append("<h1>Welcome to SOEN387 Book Store</h1>")
                   .append("<h2>Available books:</h2>")
                   .append("<ul class='book-list'>"); 

        //Add books to page with associated buttons
        String book;
        String[] parts;
        String bookPrice;
        for (int i = 0; i < books.size(); i++) {
            book = books.get(i);
            String bookName; 

            //Below is to split the first book name correctly due to : in the title
            parts = book.split(": ");
            if (i == 0) { 
                bookName = parts[0] + ": " + parts[1];
                bookPrice = parts[2];
            } else {
                bookName = parts[0]; 
                bookPrice = parts[1];
            }
      
            htmlContent.append("<li class='book-item'>")
                       .append("<span class='book-title'>").append(bookName).append("</span><br/>")
                       .append("<span class='book-price'>").append(bookPrice).append("</span><br/>")
                       .append("<form action='/A1SOEN387/MainMenuServlet' method='post' target='_self'>") 
                       .append("<input type='number' id='quantity' name='quantity' min='1' />")  
                       .append("<input type='hidden' name='bookName' value='").append(bookName).append("' />")  
                       .append("<input type='submit' value='Add to cart' />")  
                       .append("</form>")
                       .append("</li>");
        }
        
        
        //Go to cart button
        htmlContent.append("<div class='cart-button'>")
        		   .append("<a href='/A1SOEN387/CartScreenServlet'>")
        		   .append("<img src='/shoppingcart.png' alt='Cart' width='30' height='30' style='vertical-align: middle;' />") 
        		   .append("Items in Cart")
        		   .append("</a>")
        		   .append("</div>")
        		   .append("</ul>")
        		   .append("</body></html>");
   
        response.getWriter().write(htmlContent.toString());
    }

    /**
     * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse response)
     */
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    	//This is responsible for updating quantity of books in cart
    	String quantity = request.getParameter("quantity");
        String bookName = request.getParameter("bookName");

        HttpSession session = request.getSession();

        Map<String, Integer> cart = (Map<String, Integer>) session.getAttribute("cart");
        if (cart == null) {
            cart = new HashMap<>();
        }

        //Add the book and quantity to the cart
        int quantityBook = Integer.parseInt(quantity);
        cart.put(bookName, cart.getOrDefault(bookName, 0) + quantityBook);

        //Store updated cart in the session
        session.setAttribute("cart", cart);

        doGet(request, response);  //This reloads the MainMenu page
    }
}