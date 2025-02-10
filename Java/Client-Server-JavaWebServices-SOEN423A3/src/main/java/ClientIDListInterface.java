import javax.jws.WebMethod;
import javax.jws.WebService;

@WebService(targetNamespace = "http://example.com/ClientIDList")
public interface ClientIDListInterface {
    @WebMethod
    String[] getClientIDs();

    @WebMethod
    void addClientID(String clientId);

    @WebMethod
    void removeClientID(String clientId);
}
