import javax.jws.WebService;
import javax.xml.ws.Endpoint;
import java.util.ArrayList;
import java.util.List;

@WebService(endpointInterface = "ClientIDListInterface", targetNamespace = "http://example.com/ClientIDList")
public class ClientIDListServer implements ClientIDListInterface {
    private List<String> clientIDs;

    public ClientIDListServer() {
        this.clientIDs = new ArrayList<>();
    }

    @Override
    public synchronized String[] getClientIDs() {return clientIDs.toArray(new String[0]);}

    @Override
    public synchronized void addClientID(String clientID) {clientIDs.add(clientID);}

    @Override
    public synchronized void removeClientID(String clientID) {clientIDs.remove(clientID);}

    public static void main(String[] args) {
        try {
            System.out.println("Initializing the service...");
            ClientIDListServer server = new ClientIDListServer();
            System.out.println("Service initialized, publishing...");
            Endpoint.publish("http://localhost:8080/ClientIDListServer", server);
            System.out.println("ClientIDListServer Service is published!");

        } catch (Exception e) {
            System.err.println("Error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
