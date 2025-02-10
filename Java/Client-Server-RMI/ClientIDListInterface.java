import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.List;

public interface ClientIDListInterface extends Remote {
    void addClientID(String clientID) throws RemoteException;
    void removeClientID(String clientID) throws RemoteException;
    List<String> getClientIDs() throws RemoteException;
}