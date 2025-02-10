import org.omg.CORBA.ORB;
import org.omg.CosNaming.NamingContextExt;
import org.omg.CosNaming.NamingContextExtHelper;
import org.omg.PortableServer.POA;
import org.omg.PortableServer.POAHelper;
import java.util.ArrayList;
import java.util.List;

public class ClientIDListServer extends ClientIDListInterfacePOA {
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
            String[] orbArgs = {"-ORBInitialHost", "localhost", "-ORBInitialPort", "1050"};
            ORB orb = ORB.init(orbArgs, null);

            POA rootPOA = POAHelper.narrow(orb.resolve_initial_references("RootPOA"));
            rootPOA.the_POAManager().activate();

            ClientIDListServer clientIDListServer = new ClientIDListServer();

            org.omg.CORBA.Object ref = rootPOA.servant_to_reference(clientIDListServer);
            ClientIDListInterface href = ClientIDListInterfaceHelper.narrow(ref);

            org.omg.CORBA.Object objRef = orb.resolve_initial_references("NameService");
            NamingContextExt ncRef = NamingContextExtHelper.narrow(objRef);

            //Bind the servant object in the Naming Service under the name "ClientIDList"
            String name = "ClientIDList";
            ncRef.rebind(ncRef.to_name(name), href);

            System.out.println("ClientIDListServer ready and waiting...");

            orb.run();

        } catch (Exception e) {
            System.err.println("Error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
