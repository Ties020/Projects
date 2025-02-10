import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.ArrayList;
import java.util.List;

public class ConcurrencyTest {

    private static final int NUMBER_OF_CLIENTS = 10; // Number of concurrent clients

    public static void main(String[] args) {
        List<Thread> clientThreads = new ArrayList<>();

        for (int i = 0; i < NUMBER_OF_CLIENTS; i++) {
            final String clientId = "MTLR" + (1000 + i); // Create unique client IDs
            final String resourceID = "MTL" + (1000);
            final String toberemoved = "MTL" + (1000);

            Thread clientThread = new Thread(() -> {
                try {
                    Registry registry = LocateRegistry.getRegistry("localhost", getPort(clientId));
                    ServerInterface stub = (ServerInterface) registry.lookup(getServerName(clientId));

                    // Simulate operations
                    String resourceName = "AMBULANCE";
                    int duration = 5;

                    String response2 = stub.removeResource(toberemoved,null);
                    String response = stub.addResource(resourceID, resourceName, duration);
                    String response3 = stub.requestResource(clientId,resourceID,duration);

                    System.out.println("Client " + clientId + ": " + response);

                    System.out.println("Client " + clientId + "did: " + response2);

                    System.out.println("Client " + clientId + "post request: " + response3);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
            clientThreads.add(clientThread);
            clientThread.start();
        }

        // Wait for all client threads to finish
        for (Thread thread : clientThreads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        System.out.println("All clients finished.");
    }

    private static int getPort(String clientId) {
        switch (clientId.substring(0, 3).toUpperCase()) {
            case "MTL":
                return 7777;
            case "QUE":
                return 7778;
            case "SHE":
                return 7779;
            default:
                throw new IllegalArgumentException("Unknown client ID prefix");
        }
    }

    private static String getServerName(String clientId) {
        return clientId.substring(0, 3);
    }
}