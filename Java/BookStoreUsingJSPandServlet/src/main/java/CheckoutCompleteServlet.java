import jakarta.servlet.ServletException;
import jakarta.servlet.annotation.WebServlet;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import java.io.IOException;

@WebServlet("/CheckoutCompleteServlet")
public class CheckoutCompleteServlet extends HttpServlet {
    private static final long serialVersionUID = 1L;

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/html");

        StringBuilder htmlContent = new StringBuilder();
        htmlContent.append("<html><head><title>Checkout Complete</title>")
                   .append("<style>")
                   .append("body { font-family: Arial, sans-serif; text-align: center; padding-top: 50px; }")
                   .append("</style></head><body>")
                   .append("<h1>Checkout Complete</h1>")
                   .append("<a href='/A1SOEN387/MainMenuServlet'>Back to Bookstore</a>")
                   .append("</body></html>");
        
        response.getWriter().write(htmlContent.toString());
    }
}