
import java.io.Serializable;
import java.net.URL;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import javax.xml.namespace.QName;
import javax.xml.ws.Service;


public class Client implements Serializable {
    public String clientID;
    private static List<List<String>> logfilesClient = new ArrayList<>();

    public static void addLogfileClient(LocalDateTime currentTime, String resourceName, List<String> requestParameters, String responseServer) {
        String successServer = responseServer.contains("couldn't") || responseServer.contains("No") ? "failure" : "success";

        List<String> logfile = new ArrayList<>();
        logfile.add(currentTime.toString());
        logfile.add(resourceName);
        logfile.addAll(requestParameters);
        logfile.add(successServer);
        logfile.add(responseServer);

        logfilesClient.add(logfile);
    }

    public Client(String clientID) {
        super();
        this.clientID = clientID;
    }

    public static boolean isValidId(String id, boolean usedForClientID) {
        String digits = id.substring(3);

        if (usedForClientID) {
            if (id.length() != 8) return false;

            char role = id.charAt(3);
            if (role != 'R' && role != 'C' && role != 'r' && role != 'c') return false;

            digits = id.substring(4);
        }

        if (!digits.matches("\\d{4}")) return false;

        String prefix = id.substring(0, 3);
        return prefix.equalsIgnoreCase("MTL") || prefix.equalsIgnoreCase("SHE") || prefix.equalsIgnoreCase("QUE");
    }

    public static boolean isValidResName(String name) {
        return name.equalsIgnoreCase("AMBULANCE") || name.equalsIgnoreCase("FIRETRUCK") || name.equalsIgnoreCase("PERSONNEL");
    }

    //Get the appropriate server URL based on the city prefix
    private static String getServerURL(String cityPrefix) {
        switch (cityPrefix) {
            case "MTL":
                return "http://localhost:7777/ServerService?wsdl";
            case "QUE":
                return "http://localhost:7778/ServerService?wsdl";
            case "SHE":
                return "http://localhost:7779/ServerService?wsdl";
            default:
                return null;
        }
    }

    public static void main(String... strings) {
        Scanner scan = new Scanner(System.in);
        try {
            URL clientIDListURL = new URL("http://localhost:8080/ClientIDListServer?wsdl");
            QName clientIDListQName = new QName("http://example.com/ClientIDList", "ClientIDListServerService"); //specify the unique identifier for the ClientIDListServiceImplService service to be referenced by the Service object
            Service clientIDListService = Service.create(clientIDListURL, clientIDListQName); //here the Service object acts as a factory for creating service stubs that can communicate with the actual web service
            ClientIDListInterface clientIDListStub = clientIDListService.getPort(ClientIDListInterface.class);

            System.out.println("Enter your unique ID with 3 prefix letters MTL, SHE, or QUE followed by R or C meaning responder or coordinator, followed by 4 digits:");
            String clientId = scan.next();

            String[] clientIDs = clientIDListStub.getClientIDs();

            boolean idExists = false;
            for (String id : clientIDs) {
                if (id.equals(clientId)) {
                    idExists = true;
                    break;
                }
            }

            while (!isValidId(clientId, true) || idExists) {  //
                System.out.println("ID is not valid, try again");
                clientId = scan.next();
                idExists = false;  //Same as clientIDs.contains(clientID), but contains() isnt supported in java 8
                for (String id : clientIDs) {
                    if (id.equals(clientId)) {
                        idExists = true;
                        break;
                    }
                }
            }

            clientIDListStub.addClientID(clientId);
            String cityPrefix = clientId.substring(0, 3).toUpperCase();
            String serverURL = getServerURL(cityPrefix);

            //Connect to the appropriate server
            QName serverQName = new QName("http://example.com/Server", "ServerService");

            Service serverService = Service.create(new URL(serverURL), serverQName);
            System.out.println("3");

            ServerInterface serverStub = serverService.getPort(ServerInterface.class);
            System.out.println("4");


            if (clientId.charAt(3) == 'R' || clientId.charAt(3) == 'r') {
                while (true) {
                    System.out.println("Enter 'e' to disconnect or other key to continue with operations:");
                    String command = scan.next();
                    if (command.equalsIgnoreCase("e")) {
                        System.out.println("Disconnecting client...");
                        clientIDListStub.removeClientID(clientId);
                        break;
                    }

                    System.out.println("What operation (add, rem, or list)?");
                    String operation = scan.next();

                    String resourceId = "";
                    String resourceName = "";
                    Integer duration = null;
                    String responseServer = "";

                    switch (operation) {
                        case "add":
                            System.out.println("ID (like MTL1111 or SHE1384):");
                            resourceId = scan.next();
                            while (!isValidId(resourceId, false)) {
                                System.out.println("Invalid id.");
                                resourceId = scan.next();
                            }

                            System.out.println("Name (ambulance, firetruck, or personnel):");
                            resourceName = scan.next();
                            while (!isValidResName(resourceName)) {
                                System.out.println("Invalid name.");
                                resourceName = scan.next();
                            }

                            System.out.println("Duration:");
                            while (!scan.hasNextInt()) {
                                System.out.println("Input is not a valid integer.");
                                scan.next();
                            }
                            duration = scan.nextInt();
                            while (duration <= 0) {
                                System.out.println("Duration must be greater than 0. Please enter a valid duration:");
                                while (!scan.hasNextInt()) {
                                    System.out.println("Input is not a valid integer.");
                                    scan.next();
                                }
                                duration = scan.nextInt();
                            }
                            responseServer = serverStub.addResource(resourceId, resourceName, duration);
                            break;
                        case "rem":
                            System.out.println("ID:");
                            resourceId = scan.next();
                            System.out.println("remove res or decrease duration (R or D)");
                            String choice = scan.next();
                            if (choice.equals("r")) responseServer = serverStub.removeResource(resourceId, 0);
                            else {
                                System.out.println("Decrease duration by:");
                                while (!scan.hasNextInt()) {
                                    System.out.println("Input is not a valid integer.");
                                    scan.next();
                                }
                                duration = scan.nextInt(); // read the input as an integer
                                while (duration <= 0) {
                                    System.out.println("Duration must be greater than 0. Please enter a valid duration:");
                                    while (!scan.hasNextInt()) {
                                        System.out.println("Input is not a valid integer.");
                                        scan.next();
                                    }
                                    duration = scan.nextInt();
                                }
                                responseServer = serverStub.removeResource(resourceId, duration);
                            }
                            break;
                        case "list":
                            System.out.println("Name (ambulance, firetruck, or personnel):");
                            resourceName = scan.next();
                            while (!isValidResName(resourceName)) {
                                System.out.println("Invalid name.");
                                resourceName = scan.next();
                            }
                            responseServer = serverStub.listResourceAvailability(resourceName);
                            break;
                        default:
                            break;
                    }

                    // Now we add the logfile that the server has to the client's own array
                    List<String> requestParameters = new ArrayList<>();
                    requestParameters.add(resourceId);
                    requestParameters.add(resourceName);
                    if (duration != null) requestParameters.add(duration.toString());
                    else requestParameters.add(null);

                    addLogfileClient(LocalDateTime.now(), resourceName, requestParameters, responseServer);
                    System.out.println(responseServer);
                }
            } else if (clientId.charAt(3) == 'C' || clientId.charAt(3) == 'c') {
                while (true) {
                    System.out.println("Enter 'e' to disconnect or other key to continue with operations:");
                    String command = scan.next();
                    if (command.equalsIgnoreCase("e")) {
                        System.out.println("Disconnecting client...");
                        clientIDListStub.removeClientID(clientId);
                        break;
                    }

                    System.out.println("What operation (req, find, return, or swap)?");
                    String operation = scan.next();

                    String resourceId = "";
                    String resourceName = "";
                    Integer duration = null;
                    String responseServer = "";
                    String newResourceId = "";

                    switch (operation) {
                        case "req":
                            System.out.println("ID (like MTL1111 or SHE1384):");
                            resourceId = scan.next();
                            while (!isValidId(resourceId, false)) {
                                System.out.println("Invalid id.");
                                resourceId = scan.next();
                            }

                            System.out.println("Duration:");
                            while (!scan.hasNextInt()) {
                                System.out.println("Input is not a valid integer.");
                                scan.next();
                            }
                            duration = scan.nextInt(); // read the input as an integer
                            responseServer = serverStub.requestResource(clientId, resourceId, duration);


                            if(responseServer.contains("Resource not available. Would you like to be added to the queue? (yes/no)")) {
                                System.out.println("Enter 'yes' to be added to the queue, 'no' to skip:");
                                String queueResponse = scan.next();
                                if (queueResponse.equalsIgnoreCase("yes")) {
                                    serverStub.addToQueue(clientId, resourceId);
                                    responseServer = "You have been added to the queue.";
                                } else {
                                    responseServer = "You chose not to join the queue.";
                                }
                            }
                            break;
                        case "find":
                            System.out.println("Name (ambulance, firetruck, or personnel):");
                            resourceName = scan.next();
                            while (!isValidResName(resourceName)) {
                                System.out.println("Invalid name.");
                                resourceName = scan.next();
                            }
                            responseServer = serverStub.findResource(clientId, resourceName);
                            break;
                        case "return":
                            System.out.println("ID (like MTL1111 or SHE1384):");
                            resourceId = scan.next();
                            while (!isValidId(resourceId, false)) {
                                System.out.println("Invalid id.");
                                resourceId = scan.next();
                            }
                            responseServer = serverStub.returnResource(clientId, resourceId);
                            break;
                        case "swap":
                            System.out.println("Old resourceId (like MTL1111 or SHE1384):");
                            resourceId = scan.next();
                            while (!isValidId(resourceId, false)) {
                                System.out.println("Invalid id.");
                                resourceId = scan.next();
                            }

                            System.out.println("New resourceId (like MTL1111 or SHE1384):");
                            newResourceId = scan.next();
                            while (!isValidId(newResourceId, false)) {
                                System.out.println("Invalid id.");
                                newResourceId = scan.next();
                            }

                            responseServer = serverStub.swapResource(clientId, resourceId, resourceName, newResourceId, resourceName);
                            break;
                        default:
                            break;
                    }

                    // Now we add the logfile that the server has to the client's own array
                    List<String> requestParameters = new ArrayList<>();
                    requestParameters.add(resourceId);
                    requestParameters.add(resourceName);
                    if (duration != null) requestParameters.add(duration.toString());
                    else requestParameters.add(null);

                    addLogfileClient(LocalDateTime.now(), resourceName, requestParameters, responseServer);
                    System.out.println(responseServer);
                }
            }
            scan.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}