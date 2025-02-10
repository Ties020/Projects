

import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Servlet implementation class CardScreen
 */

@WebServlet("/CartScreenServlet")
public class CartScreenServlet extends HttpServlet {
	private static final long serialVersionUID = 1L;

    public CartScreenServlet() {
        super();
    }

    private static final Map<String, Double> bookPrices = new HashMap<>();
    static {
        bookPrices.put("Design Patterns: Elements of Reusable Object-Oriented Software", 59.99);
        bookPrices.put("Patterns of Enterprise Application Architecture", 47.99);
        bookPrices.put("Node.js Design Patterns", 39.99);
    }

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/html");
        HttpSession session = request.getSession();
        
        Map<String, Integer> cart = (Map<String, Integer>) session.getAttribute("cart");

        StringBuilder htmlContent = new StringBuilder();
        htmlContent.append("<html><head><title>Cart</title>")
			       .append("<style>")
			       .append("body { font-family: Arial, sans-serif; text-align: center; }")  //Center body text
			       .append("table { margin: 0 auto; width: 60%; border-collapse: collapse; }") //Center table horizontally
	               .append(".checkout-button { position: absolute; top: 10px; right: 10px; }") 
			       .append("</style></head><body>")
			       .append("<h1>Your Cart</h1>");
        

        if (cart != null && !cart.isEmpty()) {
            htmlContent.append("<form action='/A1SOEN387/CartScreenServlet' method='post'>")
                       .append("<table>")
                       .append("<tr><th>Type</th><th>Price</th><th>Quantity</th><th>Total</th><th>Remove</th></tr>");


            //Loop through the cart and display items
            for (Map.Entry<String, Integer> entry : cart.entrySet()) {
                String bookName = entry.getKey();
                int quantity = entry.getValue();
                double price = bookPrices.getOrDefault(bookName, 0.0);
                double total = price * quantity;

                htmlContent.append("<tr>")
                           .append("<td>").append(bookName).append("</td>")
                           .append("<td>$").append(String.format("%.2f", price)).append("</td>")
                           .append("<td><input type='number' name='quantity_").append(bookName)
                           .append("' value='").append(quantity).append("' min='1' required></td>")
                           .append("<td>$").append(String.format("%.2f", total)).append("</td>")
                           .append("<td><button type='submit' name='remove' value='").append(bookName).append("'>Remove</button></td>")
                           .append("</tr>");
                
             //Add an update button below the quantity field for each book
                htmlContent.append("<tr><td colspan='5' style='text-align:center;'>")
                .append("<button type='submit' name='update' value='").append(bookName).append("'>Update Quantity</button>")
                .append("</td></tr>");

            }
            
            htmlContent.append("</table></form>");
        } else {
            htmlContent.append("<p>Your cart is empty!</p>");
        }

        htmlContent.append("<a href='/A1SOEN387/MainMenuServlet'>Back to Bookstore</a>")
                   .append("</body></html>");
        
        htmlContent.append("<form action='/A1SOEN387/CartScreenServlet' method='post'>")
        		   .append("<div class='checkout-button'>")
			       .append("<input type='submit'name='checkout' value='Checkout' />")  
			       .append("</form>");
        
        response.getWriter().write(htmlContent.toString());
    }

    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    	 HttpSession session = request.getSession();
         Map<String, Integer> cart = (Map<String, Integer>) session.getAttribute("cart");

         if (cart == null) {
             cart = new HashMap<>();
         }

         //Handle removal of a book
         String bookToRemove = request.getParameter("remove");
         if (bookToRemove != null) {
             cart.remove(bookToRemove);
         }         

         //Handle the individual update button click
         String updatedBook = request.getParameter("update");
         if (updatedBook != null) {
             String quantityParam = request.getParameter("quantity_" + updatedBook);
             if (quantityParam != null) {
                 int newQuantity = Integer.parseInt(quantityParam);
                 if (newQuantity > 0) {
                     cart.put(updatedBook, newQuantity);
                 }
             }
         }
         
         //Handle checkout action
         String checkoutAction = request.getParameter("checkout");
         if (checkoutAction != null) {
             session.removeAttribute("cart");             
             response.sendRedirect("/A1SOEN387/CheckoutCompleteServlet");
             return;
         }

         //Store updated cart in session
         session.setAttribute("cart", cart);

         //Reload the CartScreen
         doGet(request, response);
}
}