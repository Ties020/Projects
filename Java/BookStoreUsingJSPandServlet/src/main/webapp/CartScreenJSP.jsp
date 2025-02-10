<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ page import="java.util.ArrayList, java.util.Map, java.util.List, java.util.HashMap" %>
    
    
<% 
	Map<String, Double> bookPrices = new HashMap<>();
    bookPrices.put("Design Patterns: Elements of Reusable Object-Oriented Software", 59.99);
    bookPrices.put("Patterns of Enterprise Application Architecture", 47.99);
    bookPrices.put("Node.js Design Patterns", 39.99);
    
    session = request.getSession();
    Map<String, Integer> cart = (Map<String, Integer>) session.getAttribute("cart");
    
 	//Handle form submission actions (remove, update, checkout)
    String bookToRemove = request.getParameter("remove");
    if (bookToRemove != null) {
        cart.remove(bookToRemove);
    }

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

    String checkoutAction = request.getParameter("checkout");
    if (checkoutAction != null) {
        session.removeAttribute("cart");
        response.sendRedirect(request.getContextPath() + "/CheckoutCompleteJSP.jsp");
        return;
    }

    // Store the updated cart back in session
    session.setAttribute("cart", cart);
    System.out.println("Cart in session: " + cart);

%>
<html>
<head>
	<title>Cart Screen</title>
	<style>
		body { font-family: Arial, sans-serif; text-align: center; }
		table { margin: 0 auto; width: 60%; border-collapse: collapse; }
		.checkout-button { position: absolute; top: 10px; right: 10px; }
	</style>
</head>
<body>
	<h1>Your Cart</h1>
	 	<% if (cart != null && !cart.isEmpty()) { %>

         <form action='CartScreenJSP.jsp' method='post'>
         <table>
		 <tr><th>Type</th><th>Price</th><th>Quantity</th><th>Total</th><th>Remove</th></tr>

           <% 
           for (Map.Entry<String, Integer> entry : cart.entrySet()) {
	           String bookName = entry.getKey();
	           int quantity = entry.getValue();
	           double price = bookPrices.getOrDefault(bookName, 0.0);
	           double total = price * quantity;
	           System.out.println("bookPrices contains " + bookName + ": " + bookPrices.containsKey(bookName));	           
			   System.out.println(bookPrices);
           %>

 		<tr>
	    	<td><%= bookName %></td>
	    	<td>$<%= String.format("%.2f", price) %></td>
	    	<td>
	        	<input type="number" name="quantity_<%= bookName %>" value="<%= quantity %>" min="1" required>
	   		</td>
	    	<td>$<%= String.format("%.2f", total) %></td>
	    	<td>
	        	<button type="submit" name="remove" value="<%= bookName %>">Remove</button>
	    	</td>
		</tr>
				
     	<tr>
	     	<td colspan='5' style='text-align:center;'>
	        <button type='submit' name='update' value="<%= bookName %>"> Update Quantity </button>
	        </td>
        </tr>

        <%  } %>    
		</table></form>
        
        <% } else { %>
            <p>Your cart is empty!</p>
        <% } %>
        
        <a href="<%= request.getContextPath() %>/MainMenuJSP.jsp">Back to Bookstore</a>

    	<form action="<%= request.getContextPath() %>/CartScreenJSP.jsp" method="post">
	        <div class="checkout-button">
	            <input type="submit" name="checkout" value="Checkout" />
	        </div>
    	</form>
</body>
</html>