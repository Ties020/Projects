import java.io.IOException;
import java.net.*;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.net.DatagramSocket;

import javax.jws.WebService;
import javax.xml.ws.Endpoint;

@WebService(endpointInterface = "ServerInterface", targetNamespace = "http://example.com/Server")
public class Server implements ServerInterface {
    private static ConcurrentHashMap<String, Map<String, Map<String, Integer>>> clientResources = new ConcurrentHashMap();
    private Map<String, Map<String, Integer>> resourcesServer = new HashMap();
    private List<List<String>> logfilesServer = new ArrayList();
    private Map<String, Queue<String>> resourceWaitQueue = new ConcurrentHashMap();
    private int udpPort;
    private String serverPrefix;
    private DatagramSocket udpSocket;
    private boolean udpListenerRunning;
    private final Lock resourcesServerLock = new ReentrantLock();
    private final Lock clientResourcesLock = new ReentrantLock();
    private final Lock resourceWaitQueueLock = new ReentrantLock();


    public Server(String serverPrefix, int udpPort) {
        this.serverPrefix = serverPrefix;
        this.udpPort = udpPort;
    }


    public void addLogfileToServer(LocalDateTime currentTime, String requestName, List<String> requestParameters, String requestStatus, String serverResponse){
        List<String> logfile = new ArrayList<>();

        logfile.add(currentTime.toString());
        logfile.add(requestName);
        logfile.addAll(requestParameters);
        logfile.add(requestStatus);
        logfile.add(serverResponse);

        logfilesServer.add(logfile);
    }


    private String getIdResource(String resourceID){
        return resourceID.substring(0,3);
    }

    private int findServerPort(String serverPrefix){
        switch (serverPrefix.toUpperCase()){
            case "MTL":
                return 7877;
            case "QUE":
                return 7878;
            case "SHE":
                return	7879;
        }
        return 0;
    }

    @Override
    public String addResource(String resourceID, String resourceName, int duration) throws IOException {
        String resourcePrefix = getIdResource(resourceID);
        System.out.println("Currently in server:" + resourcesServer);
        System.out.println("name" + serverPrefix);
        if(!resourcePrefix.equalsIgnoreCase(serverPrefix)){
            System.out.println("Going to other server");
            int targetPort = findServerPort(resourcePrefix);
            String message = "addResource:" + resourceID + ":" + resourceName + ":" + duration;
            return sendUDPMessage(message, "localhost", targetPort);


        }
        resourcesServerLock.lock();

        try {
            //Check if client entered existing resourceID but with another resource name
            for (Map.Entry<String, Map<String, Integer>> entry : resourcesServer.entrySet()) {
                Map<String, Integer> subMap = entry.getValue();
                if (subMap.containsKey(resourceID)) {
                    if (!entry.getKey().equalsIgnoreCase(resourceName)) {

                        List<String> requestParameters = new ArrayList<>();
                        requestParameters.add(resourceID);
                        requestParameters.add(resourceName);
                        requestParameters.add(String.valueOf(duration));
                        addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "failure", "Resource couldn't be added since id already belongs to another resource with another name");

                        return "Resource couldn't be added since id already belongs to another resource with another name";
                    }
                }
            }

            //Below means no name in list yet, or if the name is already present, but the given id to the name doesn't exist yet
            if (!resourcesServer.containsKey(resourceName) || !resourcesServer.get(resourceName).containsKey(resourceID)) {
                resourcesServer.putIfAbsent(resourceName, new HashMap<>());
                resourcesServer.get(resourceName).put(resourceID, duration);

                List<String> requestParameters = new ArrayList<>();
                requestParameters.add(resourceID);
                requestParameters.add(resourceName);
                requestParameters.add(String.valueOf(duration));
                addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "complete", "Resource added successfully");

                return "Resource added successfully";
            } else if (duration >= resourcesServer.get(resourceName).get(resourceID)) { //Increase duration of already existing resource
                resourcesServer.get(resourceName).put(resourceID, duration + resourcesServer.get(resourceName).get(resourceID));
                List<String> requestParameters = new ArrayList<>();
                requestParameters.add(resourceID);
                requestParameters.add(resourceName);
                requestParameters.add(String.valueOf(duration));
                addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "success", "Resource duration increased successfully");

                return "Duration increased successfully";
            }

            List<String> requestParameters = new ArrayList<>();
            requestParameters.add(resourceID);
            requestParameters.add(resourceName);
            requestParameters.add(String.valueOf(duration));
            addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "failure", "Resource couldn't be added since given duration was lower than duration for already present resource");

            return "Resource couldn't be added since given duration was lower than duration for already present resource";

        }
        finally {
            resourcesServerLock.unlock();
        }
    }


    public String removeResource(String resourceID, int duration) throws IOException {
        String resourcePrefix = getIdResource(resourceID);
        if(!resourcePrefix.equalsIgnoreCase(serverPrefix)){
            int targetPort = findServerPort(resourcePrefix);
            String message;
            if(duration != 0) message = "removeResource:" + resourceID + ":" +duration;
            else message = "removeResource:" + resourceID;
            return sendUDPMessage(message, "localhost", targetPort);
        }

        resourcesServerLock.lock();

        try {
            for (Map.Entry<String, Map<String, Integer>> entry : resourcesServer.entrySet()) {
                Map<String, Integer> subMap = entry.getValue();  //Iterate through the hashmap and create map with resourcename mapped to resourceid

                if (subMap.containsKey(resourceID)) {
                    if (duration == 0) {
                        resourcesServer.remove(entry.getKey());
                        List<String> requestParameters = new ArrayList<>();
                        requestParameters.add(resourceID);
                        addLogfileToServer(LocalDateTime.now(), entry.getKey(), requestParameters, "success", "Resource removed");

                        return "Resource removed";
                    } else if (resourcesServer.get(entry.getKey()).get(resourceID) >= duration) {
                        //Decrease duration
                        resourcesServer.get(entry.getKey()).put(resourceID, resourcesServer.get(entry.getKey()).get(resourceID) - duration);
                        List<String> requestParameters = new ArrayList<>();
                        requestParameters.add(resourceID);
                        requestParameters.add(String.valueOf(duration));

                        addLogfileToServer(LocalDateTime.now(), entry.getKey(), requestParameters, "success", "Adjusted duration");
                        return "Adjusted duration";
                    }
                }
            }

            List<String> requestParameters = new ArrayList<>();
            requestParameters.add(resourceID);
            if (duration != 0) requestParameters.add(String.valueOf(duration));
            addLogfileToServer(LocalDateTime.now(), null, requestParameters, "failure", "Resource couldn't be removed");

            return "Resource couldn't be removed";
        }
        finally {
            resourcesServerLock.unlock();
        }
    }

    private String returnResourceDetails(String resourceName) {
        String resourceInfo = "";

        resourcesServerLock.lock();
        try {
            if (resourcesServer.containsKey(resourceName)) {
                Map<String, Integer> resourceDetails = resourcesServer.get(resourceName);
                for (Map.Entry<String, Integer> entry : resourceDetails.entrySet()) {
                    resourceInfo += " " + entry.getKey() + " " + entry.getValue() + ",";
                }
            }
        }
        finally {
            resourcesServerLock.unlock();
        }
        if (!resourceInfo.isEmpty()) return resourceInfo;

        else return "";
    }

    @Override
    public String listResourceAvailability(String resourceName) throws IOException {
        String resourceAvailability = "";
        resourceAvailability += (resourceName + " -");
        String currentServerResources = returnResourceDetails(resourceName);
        if (!currentServerResources.isEmpty()) resourceAvailability += currentServerResources;

        // Get resources from other servers
        Integer currentServerUdpPort = udpPort;
        ArrayList<Integer> UDPPorts = new ArrayList<>(Arrays.asList(7877, 7878, 7879));

        for (Integer port : UDPPorts) {
            if (!port.equals(currentServerUdpPort)) {
                String response = sendUDPMessage("listResources:" + resourceName, "localhost", port);
                if (!response.isEmpty()) resourceAvailability += response;
            }
        }

        List<String> requestParameters = new ArrayList<>();
        requestParameters.add(resourceName);

        if (resourceAvailability.equals(resourceName + " -")){
            resourceAvailability += ("No resources found");
            addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "failure", "No resources found given the resource name");

        }

        else {
            resourceAvailability = resourceAvailability.substring(0, resourceAvailability.length() - 1); //Remove last comma
            addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "success", resourceAvailability);
        }

        return resourceAvailability;
    }



    @Override
    public String requestResource(String coordinatorID, String resourceID, int duration) throws IOException {
        String resourcePrefix = getIdResource(resourceID);
        if(!resourcePrefix.equalsIgnoreCase(serverPrefix)){
            int targetPort = findServerPort(resourcePrefix);
            String message = "requestResource:" + coordinatorID + ":" + resourceID  + ":" + duration;
            return sendUDPMessage(message, "localhost", targetPort);
        }

        else{
            resourcesServerLock.lock();
            try {
                for (Map.Entry<String, Map<String, Integer>> entry : resourcesServer.entrySet()) {
                    Map<String, Integer> subMap = entry.getValue();
                    if (subMap.containsKey(resourceID)) {
                        if (resourcesServer.get(entry.getKey()).get(resourceID) >= duration) {

                            resourcesServer.get(entry.getKey()).put(resourceID, resourcesServer.get(entry.getKey()).get(resourceID) - duration);
                            //Check whether client already holds the requested resource
                            clientResourcesLock.lock();
                            try {
                                if (!clientResources.isEmpty() && clientResources.get(coordinatorID) != null && clientResources.get(coordinatorID).get(entry.getKey()).containsKey(resourceID)) {
                                    int clientHeldDuration = clientResources.get(coordinatorID).get(entry.getKey()).get(resourceID);
                                    //Increase duration if held is lower/equal to requested, if duration is higher than requested, decrease duration
                                    if (clientResources.get(coordinatorID).get(entry.getKey()).get(resourceID) <= duration) {
                                        clientResources.get(coordinatorID).get(entry.getKey()).put(resourceID, clientHeldDuration + duration);
                                    } else {
                                        clientResources.get(coordinatorID).get(entry.getKey()).put(resourceID, clientHeldDuration - duration);
                                    }
                                } else {
                                    clientResources.putIfAbsent(coordinatorID, new HashMap<>());
                                    clientResources.get(coordinatorID).putIfAbsent(entry.getKey(), new HashMap<>());
                                    clientResources.get(coordinatorID).get(entry.getKey()).put(resourceID, duration);
                                }
                            }
                            finally {
                                clientResourcesLock.unlock();
                            }

                            List<String> requestParameters = new ArrayList<>();
                            requestParameters.add((coordinatorID));
                            requestParameters.add(resourceID);
                            requestParameters.add(String.valueOf(duration));

                            addLogfileToServer(LocalDateTime.now(), entry.getKey(), requestParameters, "success", "Resource was given");

                            return "Resource was given";
                        }
                    }
                }


                List<String> requestParameters = new ArrayList<>();
                requestParameters.add((coordinatorID));
                requestParameters.add(resourceID);
                requestParameters.add(String.valueOf(duration));

                addLogfileToServer(LocalDateTime.now(), null, requestParameters, "success", "Resource not available. Would you like to be added to the queue? (yes/no)");

                return "Resource not available. Would you like to be added to the queue? (yes/no)";
            }
            finally {
                resourcesServerLock.unlock();
            }

        }
    }

    @Override
    public void addToQueue(String clientId, String resourceId){
        resourceWaitQueueLock.lock();
        try{
            resourceWaitQueue.putIfAbsent(resourceId, new LinkedList<>());
            resourceWaitQueue.get(resourceId).add(clientId);
        }
        finally {
            resourceWaitQueueLock.unlock();
        }

    }


    private String returnHeldResourceDetails(String resourceName, String clientID) {
        String resourceInfo = "";

        //first get the inner map based on the id
        clientResourcesLock.lock();
        try {
            if (clientResources.containsKey(clientID)) {
                Map<String, Map<String, Integer>> resourceDetails = clientResources.get(clientID);

                for (Map.Entry<String, Map<String, Integer>> resourceNameMapping : resourceDetails.entrySet()) {
                    if (resourceNameMapping.getKey().equals(resourceName)) {
                        Map<String, Integer> resourceIDMap = resourceNameMapping.getValue();
                        for (Map.Entry<String, Integer> idAndDuration : resourceIDMap.entrySet()) {
                            resourceInfo += " " + idAndDuration.getKey() + " " + idAndDuration.getValue() + ",";
                        }
                    }
                }
            }
        }
        finally {
            clientResourcesLock.unlock();
        }
        if (!resourceInfo.isEmpty()) return resourceInfo.substring(0, resourceInfo.length() - 1); //Remove last comma

        else return "";
    }

    @Override
    public String findResource(String coordinatorID, String resourceName) throws IOException {
        //First add ids and durations to the string, these are details from resourcename only held by coordinator. So check array coordinator resources
        //If not prefix same as current server -> send udp request to other server to return details
        String resourceHeldDetails = "";
        resourceHeldDetails += (resourceName + " -");
        String currentServerResources = returnHeldResourceDetails(resourceName, coordinatorID);
        if (!currentServerResources.isEmpty()) resourceHeldDetails += currentServerResources;

        // Get resources from other servers
        Integer currentServerUdpPort = udpPort;
        ArrayList<Integer> UDPPorts = new ArrayList<>(Arrays.asList(7877, 7878, 7879));

        for (Integer port : UDPPorts) {
            if (!port.equals(currentServerUdpPort)) {
                String response = sendUDPMessage("findResource:" + coordinatorID + ":" + resourceName, "localhost", port);
                if (!response.isEmpty()) resourceHeldDetails += response;
            }
        }

        List<String> requestParameters = new ArrayList<>();
        requestParameters.add((coordinatorID));
        requestParameters.add(resourceName);

        if (resourceHeldDetails.equals(resourceName + " -")){
            resourceHeldDetails += ("No resources found");
            addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "failure", resourceHeldDetails);

        }

        else addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "success", resourceHeldDetails);

        return resourceHeldDetails;
    }

    private String returnResourceToServer(String coordinatorID, String resourceID) {
        clientResourcesLock.lock();
        try {

            if (clientResources.containsKey(coordinatorID)) {
                Map<String, Map<String, Integer>> resourceDetails = clientResources.get(coordinatorID);
                for (Map.Entry<String, Map<String, Integer>> entry : resourceDetails.entrySet()) {
                    Map<String, Integer> subMap = entry.getValue();
                    if (subMap.containsKey(resourceID)) {
                        resourcesServerLock.lock();

                        try {

                            //Increase duration if resource already owned by server
                            if (resourcesServer.get(entry.getKey()).containsKey(resourceID)) {
                                resourcesServer.get(entry.getKey()).put(resourceID, resourcesServer.get(entry.getKey()).get(resourceID) + subMap.get(resourceID));
                            }
                            //Server already has resources of the same resource type, so just add the resource back
                            else if (resourcesServer.containsKey(entry.getKey())) {
                                resourcesServer.get(entry.getKey()).put(resourceID, subMap.get(resourceID));
                            }
                            //Add resource with name to server if not present there
                            else resourcesServer.put(entry.getKey(), subMap);
                        }
                        finally {
                            resourcesServerLock.unlock();
                        }


                        // Check if there are any clients waiting for this resource
                        resourceWaitQueueLock.lock();
                        try {
                            if (resourceWaitQueue.containsKey(resourceID) && !resourceWaitQueue.get(resourceID).isEmpty()) {
                                String nextClient = resourceWaitQueue.get(resourceID).poll(); //Retrieves and removes the head of the waiting queue and adds resource to it
                                if (nextClient != null) {
                                    clientResources.putIfAbsent(nextClient, new HashMap<>());
                                    clientResources.get(nextClient).putIfAbsent(entry.getKey(), new HashMap<>());
                                    clientResources.get(nextClient).get(entry.getKey()).put(resourceID, subMap.get(resourceID));
                                }
                            }
                        }
                        finally{
                            resourceWaitQueueLock.unlock();
                        }

                        subMap.remove(resourceID);
                        // Clean up if the subMap becomes empty
                        if (subMap.isEmpty()) {
                            resourceDetails.remove(entry.getKey());
                        }
                        // Clean up if the resourceDetails becomes empty
                        if (resourceDetails.isEmpty()) {
                            clientResources.remove(coordinatorID);
                        }

                        return "Successfully returned resource to server";
                    }
                }
            }
        }
        finally {
            clientResourcesLock.unlock();
        }
        return "";
    }

    @Override
    public String returnResource(String coordinatorID, String resourceID) throws IOException {
        String resourcePrefix = getIdResource(resourceID);
        if(!resourcePrefix.equalsIgnoreCase(serverPrefix)){
            int targetPort = findServerPort(resourcePrefix);
            String message = "returnResource:" + coordinatorID + ":" + resourceID;
            return sendUDPMessage(message, "localhost", targetPort);

        }

        else{
            returnResourceToServer(coordinatorID, resourceID);
            List<String> requestParameters = new ArrayList<>();
            requestParameters.add((coordinatorID));
            requestParameters.add(resourceID);
            addLogfileToServer(LocalDateTime.now(), null, requestParameters, "success", "Successfully returned resource to server");
            return "Successfully returned resource to server";
        }
    }

    private int getResourceDuration(String coordinatorID, String resourceID) {
        clientResourcesLock.lock();
        try {
            if (clientResources.containsKey(coordinatorID)) {
                Map<String, Map<String, Integer>> resourceDetails = clientResources.get(coordinatorID);

                for (Map.Entry<String, Map<String, Integer>> entry : resourceDetails.entrySet()) {
                    Map<String, Integer> subMap = entry.getValue();
                    if (subMap.containsKey(resourceID)) {
                        return subMap.get(resourceID); //Return duration of the resource
                    }
                }
            }
            return 0; // Resource not found
        } finally {
            clientResourcesLock.unlock();
        }
    }


    @Override
    public String swapResource(String coordinatorID, String oldResourceID, String oldResourceType, String newResourceID, String newResourceType) throws IOException {
        int duration = getResourceDuration(coordinatorID,oldResourceID);
        String responseRequest = "";
        String responseReturn = "";
        if(duration != 0){  //Means client has resource
            responseRequest = requestResource(coordinatorID,newResourceID,1); //Check whether resource is available

            if(responseRequest.equals("Resource was given")){
                responseReturn = returnResource(coordinatorID,oldResourceID);

                if(responseReturn.equals("Successfully returned resource to server")){   //Swap was successful
                    List<String> requestParameters = new ArrayList<>();
                    requestParameters.add((coordinatorID));
                    requestParameters.add(oldResourceID);
                    requestParameters.add(newResourceID);
                    addLogfileToServer(LocalDateTime.now(), null, requestParameters, "success", "Successfully swapped resources");
                    return "Successfully swapped resources";
                }
            }
        }

        List<String> requestParameters = new ArrayList<>();
        requestParameters.add((coordinatorID));
        requestParameters.add(oldResourceID);
        requestParameters.add(newResourceID);
        addLogfileToServer(LocalDateTime.now(), null, requestParameters, "success", "No success in swapping resources");
        return "No success in swapping resources";

    }


    public void startUDPListener() throws IOException {
        System.out.println("Started at" + udpPort);

        udpListenerRunning = true;
        udpSocket = new DatagramSocket(udpPort); //Listening on server's UDP port

        Thread udpListenerThread = new Thread(() -> {
            byte[] buffer = new byte[256];
            while (udpListenerRunning) {
                try {
                    DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                    udpSocket.receive(packet);  // Wait for incoming messages
                    String received = new String(packet.getData(), 0, packet.getLength());
                    System.out.println("Server " + serverPrefix + " received: " + received);
                    processUDPMessage(received,packet);
                    System.out.println("Processed message");


                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            udpSocket.close();
        });

        udpListenerThread.start();  //Start UDP listener in a separate thread
    }

    private String processUDPMessage(String message, DatagramPacket packet) throws IOException {
        System.out.println("processing udpmessage");
        String[] parts = message.split(":");
        String method = parts[0];
        String response = "";
        String resourceID;
        Integer duration;


        switch (method) {
            case "addResource":
                //Extract the parameters
                resourceID = parts[1];
                String resourceName = parts[2];
                duration = Integer.parseInt(parts[3]);
                System.out.println(serverPrefix + " added resource from UDP: " + resourceName + " (ID: " + resourceID + ", Duration: " + duration + ")");
                response = addResource(resourceID, resourceName, duration);
                break;


            case "removeResource":
                resourceID = parts[1];
                if (parts.length > 2){
                    duration = Integer.parseInt(parts[2]);
                    System.out.println(serverPrefix + " removed resource from UDP: (ID: " + resourceID + ", Duration: " + duration + ")");
                }
                else{
                    duration = 0;
                    System.out.println(serverPrefix + " removed resource from UDP: (ID: " + resourceID + ")");
                }

                response = removeResource(resourceID,  duration);
                System.out.println(serverPrefix + " is removing resource: " + parts[1]);
                break;

            case "listResources":
                System.out.println(serverPrefix + " is listing resources for: " + parts[1]);
                resourceName = parts[1];
                response = returnResourceDetails(resourceName);  // Get resources for the requested resourceName
                break;


            case "requestResource":
                String userID = parts[1];
                resourceID = parts[2];
                duration = Integer.parseInt(parts[3]);
                response = requestResource(userID,resourceID,duration);
                break;



            case "findResource":
                System.out.println(serverPrefix + " is listing the held resources of name:" + parts[1] + " for coordinator: " + parts[2]);
                String coordinatorID = parts[1];
                resourceName = parts[2];
                response = returnHeldResourceDetails(coordinatorID,resourceName);
                break;

            case "returnResource":
                System.out.println(serverPrefix + " is returning:" + parts[2] + " for coordinator: " + parts[1]);
                coordinatorID = parts[1];
                resourceID = parts[2];
                response = returnResourceToServer(coordinatorID,resourceID);
                break;

            default:
                System.out.println("Unknown method: " + method);
                return "";
        }
        // Send response back
        InetAddress returnAddress = packet.getAddress();
        int returnPort = packet.getPort();
        byte[] responseData = response.getBytes();
        DatagramPacket responsePacket = new DatagramPacket(responseData, responseData.length, returnAddress, returnPort);
        udpSocket.send(responsePacket);  // Send the response
        return response;
    }

    public String sendUDPMessage(String message, String targetHost, int targetPort) throws IOException {
        DatagramSocket socket = new DatagramSocket();
        byte[] buffer = message.getBytes();
        InetAddress targetAddress = InetAddress.getByName(targetHost);
        System.out.println("Send to:" + targetAddress + "and" + targetPort);

        DatagramPacket packet = new DatagramPacket(buffer, buffer.length, targetAddress, targetPort);
        socket.send(packet);

        //Wait for response
        byte[] responseBuffer = new byte[256];
        DatagramPacket responsePacket = new DatagramPacket(responseBuffer, responseBuffer.length);

        socket.receive(responsePacket);  // Wait for the response from the target server
        String response = new String(responsePacket.getData(), 0, responsePacket.getLength());

        return response;

    }

    public static class ServerThread implements Runnable {
        private int port;
        private int udpPort;
        private String serverName;

        public ServerThread(int port, String serverName, int udpPort) {
            this.port = port;
            this.serverName = serverName;
            this.udpPort = udpPort;
        }

        public void run() {
            try {
                Server server = new Server(this.serverName,this.udpPort);

                String url = "http://localhost:" + port + "/ServerService";
                Endpoint.publish(url, server);
                System.out.println("Server: " + this.serverName + " is running on port: " + this.port + " and UDP port: " + this.udpPort);

                //Start the UDP listener in a separate thread
                server.startUDPListener();

            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }


    public static void main(String[] args) {

        List<Thread> serverThreads = new ArrayList<>();

        Thread serverMTL = new Thread(new ServerThread(7777, "MTL",7877));
        serverThreads.add(serverMTL);
        serverMTL.start();

        Thread serverQUE = new Thread(new ServerThread(7778, "QUE", 7878));
        serverThreads.add(serverQUE);
        serverQUE.start();

        Thread serverSHE = new Thread(new ServerThread(7779, "SHE", 7879));
        serverThreads.add(serverSHE);
        serverSHE.start();

        for (Thread t : serverThreads) {
            try {
                t.join(); //Wait for all server threads to finish execution
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}



