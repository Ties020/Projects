import java.io.IOException;
import java.rmi.Remote;
import java.rmi.RemoteException;
import java.rmi.server.ServerNotActiveException;
import java.util.List;

public interface ServerInterface extends Remote {
	//Responder methods
	public String addResource(String resourceID, String resourceName, Integer duration) throws IOException, ServerNotActiveException;

	public String removeResource(String resourceID, Integer duration) throws IOException;

	public String listResourceAvailability(String resourceName) throws IOException;

	//Coordinator methods

	public String requestResource(String coordinatorID, String resourceID, Integer duration) throws IOException;

	public String findResource(String coordinatorID, String resourceName) throws IOException;

	public String returnResource(String coordinatorID, String resourceID) throws IOException;

	public void addToQueue(String clientId, String resourceId) throws RemoteException;
}
