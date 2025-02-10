import org.omg.CORBA.*;
import org.omg.CosNaming.*;
import java.util.ArrayList;
import java.util.List;

public class ConcurrencyTest {

    private static final int NUMBER_OF_CLIENTS = 10; //Number of concurrent clients

    public static void main(String[] args) {
        List<Thread> clientThreads = new ArrayList<>();

        try {
            String[] orbArgs = {"-ORBInitialHost", "localhost", "-ORBInitialPort", "1050"};
            ORB orb = ORB.init(orbArgs, null);

            org.omg.CORBA.Object objRef = orb.resolve_initial_references("NameService");
            NamingContextExt ncRef = NamingContextExtHelper.narrow(objRef);

            for (int i = 0; i < NUMBER_OF_CLIENTS; i++) {
                final String clientId = "MTLR" + (1000 + i); //Create unique client IDs
                final String resourceID = "MTL" + (1000);
                final String newResource = "MTL" + (2000);
                final String toberemoved = "MTL" + (1000);

                Thread clientThread = new Thread(() -> {
                    try {
                        String serverName = getServerName(clientId);
                        ServerInterface serverStub = ServerInterfaceHelper.narrow(ncRef.resolve_str(serverName));

                        String resourceName = "AMBULANCE";
                        int duration = 5;

                        String response1 = serverStub.removeResource(toberemoved, 0);
                        String response2 = serverStub.addResource(resourceID, resourceName, duration);
                        String response3 = serverStub.addResource(newResource, resourceName, duration);
                        String response4 = serverStub.requestResource(clientId, resourceID, duration);
                        String response5 = serverStub.swapResource(clientId, resourceID,resourceName,newResource,resourceName);

                        System.out.println("Client " + clientId + "remove: " + response1);
                        System.out.println("Client " + clientId + " add1: " + response2);
                        System.out.println("Client " + clientId + " add2: " + response3);
                        System.out.println("Client " + clientId + " req: " + response4);
                        System.out.println("Client " + clientId + " swap: " + response5);

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                });
                clientThreads.add(clientThread);
                clientThread.start();
            }

            //Wait for all client threads to finish
            for (Thread thread : clientThreads) {
                try {
                    thread.join();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            System.out.println("All clients finished.");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String getServerName(String clientId) {
        return clientId.substring(0, 3); //Return the appropriate server name based on clientId
    }
}
