import javax.jws.WebMethod;
import javax.jws.WebService;
import java.io.IOException;
import java.rmi.RemoteException;

@WebService(targetNamespace = "http://example.com/Server")
public interface ServerInterface {
    @WebMethod
    String addResource(String resourceId, String resourceName, int duration) throws IOException;

    @WebMethod
    String removeResource(String resourceId, int duration) throws IOException;

    @WebMethod
    String listResourceAvailability(String resourceName) throws IOException;

    @WebMethod
    String requestResource(String clientId, String resourceId, int duration) throws IOException;

    @WebMethod
    String findResource(String clientId, String resourceName) throws IOException;

    @WebMethod
    String returnResource(String clientId, String resourceId) throws IOException;

    @WebMethod
    String swapResource(String clientId, String oldResourceId, String resourceName, String newResourceId, String newResourceName) throws IOException;

    @WebMethod
    public void addToQueue(String clientId, String resourceId) throws RemoteException;

}


