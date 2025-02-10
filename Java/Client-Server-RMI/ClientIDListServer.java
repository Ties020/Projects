import java.rmi.*;
import java.rmi.registry.LocateRegistry;
import java.rmi.server.UnicastRemoteObject;
import java.util.ArrayList;
import java.util.List;

public class ClientIDListServer extends UnicastRemoteObject implements ClientIDListInterface {

    private List<String> clientIDs;

    protected ClientIDListServer() throws RemoteException {
        super();
        this.clientIDs = new ArrayList<>();
    }
    @Override
    public synchronized List<String> getClientIDs() throws RemoteException {return clientIDs;}

    @Override
    public synchronized void addClientID(String clientID) throws RemoteException {clientIDs.add(clientID);}

    @Override
    public synchronized void removeClientID(String clientID) throws RemoteException {clientIDs.remove(clientID);}

    public static void main(String[] args) {
        try {
            LocateRegistry.createRegistry(1099);
            System.out.println("RMI registry started.");
            ClientIDListServer clientIDList = new ClientIDListServer();
            Naming.rebind("ClientIDList", clientIDList);
            System.out.println("ClientIDList bound to registry.");

        } catch (Exception e) {
            System.err.println("Exception occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}