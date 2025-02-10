<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
<%@ page import="java.util.ArrayList, java.util.Map, java.util.List, java.util.HashMap" %>

<%
    List<String> books = new ArrayList<>();
    books.add("Design Patterns: Elements of Reusable Object-Oriented Software: $59.99");
    books.add("Patterns of Enterprise Application Architecture: $47.99");
    books.add("Node.js Design Patterns: $39.99");

    session = request.getSession();
    Map<String, Integer> cart = (Map<String, Integer>) session.getAttribute("cart");
    if (cart == null) {
        cart = new HashMap<>();
    }
    
 	//Handle POST request for adding books to the cart
    if ("POST".equalsIgnoreCase(request.getMethod())) {
        String bookName = request.getParameter("bookName");
        int quantity = Integer.parseInt(request.getParameter("quantity"));

        cart.put(bookName, cart.getOrDefault(bookName, 0) + quantity);
        session.setAttribute("cart", cart);
    }
%>

<html>
<head>
    <title>Book Store</title>
    <style>
        body { font-family: Arial, sans-serif; }
        h1, h2 { text-align: center; }
        .cart-button { position: absolute; top: 10px; right: 10px; }
        .book-list { width: 60%; margin: 0 auto; list-style: none; padding: 0; }
        .book-item { border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; }
        .book-title { font-size: 1.2em; margin-bottom: 5px; }
        .book-price { color: #007BFF; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Welcome to SOEN387 Book Store</h1>
    <h2>Available books:</h2>
    <ul class="book-list">
        <%
	        String book;
	        String[] parts;
	        String bookPrice;
	        for (int i = 0; i < books.size(); i++) {
	            book = books.get(i);
	            String bookName; 
	
	            parts = book.split(": ");
	            if (i == 0) { 
	                bookName = parts[0] + ": " + parts[1];
	                bookPrice = parts[2];
	            } else {
	                bookName = parts[0]; 
	                bookPrice = parts[1];
	            }
        %>
            <li class="book-item">
                <span class="book-title"><%= bookName %></span><br/>
                <span class="book-price"><%= bookPrice %></span><br/>
                <form action="MainMenuJSP.jsp" method="post">
                    <input type="number" id="quantity" name="quantity" min="1" />
                    <input type="hidden" name="bookName" value="<%= bookName %>" />
                    <input type="submit" value="Add to cart" />
                </form>
            </li>
        <% } %>
    </ul>

    <!-- Go to cart button -->
    <div class="cart-button">
        <a href="<%= request.getContextPath() %>/CartScreenJSP.jsp">
            <img src="<%= request.getContextPath() %>/shoppingcart.png" alt="Cart" width="30" height="30" style="vertical-align: middle;" />
            Items in Cart
        </a>
    </div>
</body>
</html>
