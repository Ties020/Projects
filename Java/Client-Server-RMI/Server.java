import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.rmi.AlreadyBoundException;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Server extends UnicastRemoteObject implements ServerInterface {
	private static ThreadLocal<String> currentClientId = new ThreadLocal<>(); //Makes it only accessible to thread that creates id
	private static ConcurrentHashMap<String, Map<String, Map<String, Integer>>> clientResources = new ConcurrentHashMap<>();
	private Map<String, Map<String, Integer>> resourcesServer;
	private List<List<String>> logfilesServer;
	private Map<String, Queue<String>> resourceWaitQueue = new ConcurrentHashMap<>();
	private int udpPort;
	private String serverPrefix;
	private DatagramSocket udpSocket;
	private boolean udpListenerRunning;
	private final Lock resourcesServerLock = new ReentrantLock();
	private final Lock clientResourcesLock = new ReentrantLock();
	private final Lock resourceWaitQueueLock = new ReentrantLock();

	public Server(int udpPort, String serverPrefix) throws RemoteException {
		super();
		resourcesServer = new HashMap<>();
		logfilesServer = new ArrayList<>();
		this.udpPort = udpPort;
		this.serverPrefix = serverPrefix;
	}

	public void setCurrentClientId(String clientId) throws RemoteException {
		currentClientId.set(clientId);
	}

	// Start the UDP listener inside the server
	public void startUDPListener() throws IOException {
		udpListenerRunning = true;
		udpSocket = new DatagramSocket(udpPort); // Listening on server's UDP port

		Thread udpListenerThread = new Thread(() -> {
			byte[] buffer = new byte[256];
			while (udpListenerRunning) {
				try {
					DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
					udpSocket.receive(packet);  // Wait for incoming messages
					String received = new String(packet.getData(), 0, packet.getLength());
					System.out.println("Server " + serverPrefix + " received: " + received);
					processUDPMessage(received,packet);

				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			udpSocket.close();
		});

		udpListenerThread.start();  // Start UDP listener in a separate thread
	}

	// Process incoming UDP messages
	private String processUDPMessage(String message, DatagramPacket packet) throws IOException {
		String[] parts = message.split(":");
		String method = parts[0];
		String response = "";
		String resourceID;
		Integer duration;


		switch (method) {
			case "addResource":
				// Extract the parameters
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
					duration = null;
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

	//Used for inter-server communication
	public String sendUDPMessage(String message, String targetHost, int targetPort) throws IOException {
		DatagramSocket socket = new DatagramSocket();
		byte[] buffer = message.getBytes();
		InetAddress targetAddress = InetAddress.getByName(targetHost);
		DatagramPacket packet = new DatagramPacket(buffer, buffer.length, targetAddress, targetPort);
		socket.send(packet);

		// Wait for response
		byte[] responseBuffer = new byte[256];
		DatagramPacket responsePacket = new DatagramPacket(responseBuffer, responseBuffer.length);
		socket.receive(responsePacket);  // Wait for the response from the target server
		String response = new String(responsePacket.getData(), 0, responsePacket.getLength());

		return response;

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

	//Responder methods
	@Override
	public String addResource(String resourceID, String resourceName, Integer duration) throws IOException {
		String resourcePrefix = getIdResource(resourceID);
		System.out.println("Currently in server:" + resourcesServer);
		if(!resourcePrefix.equalsIgnoreCase(serverPrefix)){
			int targetPort = findServerPort(resourcePrefix);
			String message = "addResource:" + resourceID + ":" + resourceName + ":" + duration.toString();
			return sendUDPMessage(message, "localhost", targetPort);
		}
		resourcesServerLock.lock();

		//Check if client entered existing resourceID but with another resource name
		for (Map.Entry<String, Map<String, Integer>> entry : resourcesServer.entrySet()) {
			Map<String, Integer> subMap = entry.getValue();
			if (subMap.containsKey(resourceID)) {
				if (!entry.getKey().equalsIgnoreCase(resourceName)) {
					resourcesServerLock.unlock();

					List<String> requestParameters = new ArrayList<>();
					requestParameters.add(resourceID);
					requestParameters.add(resourceName);
					requestParameters.add(duration.toString());
					addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "failure", "Resource couldn't be added since id already belongs to another resource with another name");

					return "Resource couldn't be added since id already belongs to another resource with another name";
				}
			}
		}

		//Below means no name in list yet, or if the name is already present, but the given id to the name doesn't exist yet
		if (!resourcesServer.containsKey(resourceName) || !resourcesServer.get(resourceName).containsKey(resourceID)) {
			resourcesServer.putIfAbsent(resourceName, new HashMap<>());
			resourcesServer.get(resourceName).put(resourceID, duration);
			resourcesServerLock.unlock();

			List<String> requestParameters = new ArrayList<>();
			requestParameters.add(resourceID);
			requestParameters.add(resourceName);
			requestParameters.add(duration.toString());
			addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "complete", "Resource added successfully");

			return "Resource added successfully";
		}
		else if(duration >= resourcesServer.get(resourceName).get(resourceID)){ //Increase duration of already existing resource
			resourcesServer.get(resourceName).put(resourceID, duration + resourcesServer.get(resourceName).get(resourceID));
			resourcesServerLock.unlock();
			List<String> requestParameters = new ArrayList<>();
			requestParameters.add(resourceID);
			requestParameters.add(resourceName);
			requestParameters.add(duration.toString());
			addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "success", "Resource duration increased successfully");

			return "Duration increased successfully";
		}

		resourcesServerLock.unlock();

		List<String> requestParameters = new ArrayList<>();
		requestParameters.add(resourceID);
		requestParameters.add(resourceName);
		requestParameters.add(duration.toString());
		addLogfileToServer(LocalDateTime.now(), resourceName, requestParameters, "failure", "Resource couldn't be added since given duration was lower than duration for already present resource");

		return "Resource couldn't be added since given duration was lower than duration for already present resource";
	}

	@Override
	public String removeResource(String resourceID, Integer duration) throws IOException {
		String resourcePrefix = getIdResource(resourceID);
		if(!resourcePrefix.equalsIgnoreCase(serverPrefix)){
			int targetPort = findServerPort(resourcePrefix);
			String message;
			if(duration != null) message = "removeResource:" + resourceID + ":" + duration.toString();
			else message = "removeResource:" + resourceID;

			return sendUDPMessage(message, "localhost", targetPort);
		}

		resourcesServerLock.lock();
		for (Map.Entry<String, Map<String, Integer>> entry : resourcesServer.entrySet()) {
			Map<String, Integer> subMap = entry.getValue();  //Iterate through the hashmap and create map with resourcename mapped to resourceid

			if (subMap.containsKey(resourceID)) {
				if (duration == null) {
					resourcesServer.remove(entry.getKey());
					resourcesServerLock.unlock();
					List<String> requestParameters = new ArrayList<>();
					requestParameters.add(resourceID);
					addLogfileToServer(LocalDateTime.now(), entry.getKey(), requestParameters, "success", "Resource removed");

					return "Resource removed";
				}
				else if(resourcesServer.get(entry.getKey()).get(resourceID) >= duration){
					//Decrease duration
					resourcesServer.get(entry.getKey()).put(resourceID, resourcesServer.get(entry.getKey()).get(resourceID) - duration  );
					resourcesServerLock.unlock();
					List<String> requestParameters = new ArrayList<>();
					requestParameters.add(resourceID);
					requestParameters.add(duration.toString());

					addLogfileToServer(LocalDateTime.now(), entry.getKey(), requestParameters, "success", "Adjusted duration");
					return "Adjusted duration";
				}
			}
		}

		resourcesServerLock.unlock();

		List<String> requestParameters = new ArrayList<>();
		requestParameters.add(resourceID);
		if(duration != null) requestParameters.add(duration.toString());
		addLogfileToServer(LocalDateTime.now(), null, requestParameters, "failure", "Resource couldn't be removed");

		return "Resource couldn't be removed";
	}

	private String returnResourceDetails(String resourceName) {
		String resourceInfo = "";

		resourcesServerLock.lock();
		if (resourcesServer.containsKey(resourceName)) {
			Map<String, Integer> resourceDetails = resourcesServer.get(resourceName);
			resourcesServerLock.unlock();
			for (Map.Entry<String, Integer> entry : resourceDetails.entrySet()) {
				resourceInfo += " " + entry.getKey() + " " + entry.getValue() + ",";
			}
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

	//Coordinator methods
	@Override
	public String requestResource(String coordinatorID, String resourceID, Integer duration) throws IOException {
		String resourcePrefix = getIdResource(resourceID);
		if(!resourcePrefix.equalsIgnoreCase(serverPrefix)){
			int targetPort = findServerPort(resourcePrefix);
			String message = "requestResource:" + coordinatorID + ":" + resourceID  + ":" + duration.toString();
			return sendUDPMessage(message, "localhost", targetPort);
		}

		else{
			resourcesServerLock.lock();
			for (Map.Entry<String, Map<String, Integer>> entry : resourcesServer.entrySet()) {
				Map<String, Integer> subMap = entry.getValue();
				if (subMap.containsKey(resourceID)) {
					if(resourcesServer.get(entry.getKey()).get(resourceID) >= duration){

						resourcesServer.get(entry.getKey()).put(resourceID, resourcesServer.get(entry.getKey()).get(resourceID) - duration);
						resourcesServerLock.unlock();
						//Check whether client already holds the requested resource
						clientResourcesLock.lock();
						if(!clientResources.isEmpty() && clientResources.get(coordinatorID) != null && clientResources.get(coordinatorID).get(entry.getKey()).containsKey(resourceID)){
							int clientHeldDuration = clientResources.get(coordinatorID).get(entry.getKey()).get(resourceID);
							//Increase duration if held is lower/equal to requested, if duration is higher than requested, decrease duration
							if(clientResources.get(coordinatorID).get(entry.getKey()).get(resourceID) <= duration){
								clientResources.get(coordinatorID).get(entry.getKey()).put(resourceID, clientHeldDuration + duration);
							}
							else{
								clientResources.get(coordinatorID).get(entry.getKey()).put(resourceID, clientHeldDuration - duration);
							}
						}
						else{
							clientResources.putIfAbsent(coordinatorID, new HashMap<>());
							clientResources.get(coordinatorID).putIfAbsent(entry.getKey(), new HashMap<>());
							clientResources.get(coordinatorID).get(entry.getKey()).put(resourceID, duration);
						}
						clientResourcesLock.unlock();

						List<String> requestParameters = new ArrayList<>();
						requestParameters.add((coordinatorID));
						requestParameters.add(resourceID);
						requestParameters.add(duration.toString());

						addLogfileToServer(LocalDateTime.now(), entry.getKey(), requestParameters, "success", "Resource was given");

						return "Resource was given";
					}
				}
			}

			resourcesServerLock.unlock();

			List<String> requestParameters = new ArrayList<>();
			requestParameters.add((coordinatorID));
			requestParameters.add(resourceID);
			requestParameters.add(duration.toString());

			addLogfileToServer(LocalDateTime.now(), null, requestParameters, "success", "Resource not available. Would you like to be added to the queue? (yes/no)");

			return "Resource not available. Would you like to be added to the queue? (yes/no)";

		}
	}

	@Override
	public void addToQueue(String clientId, String resourceId) throws RemoteException{
		resourceWaitQueueLock.lock();
		resourceWaitQueue.putIfAbsent(resourceId, new LinkedList<>());
		resourceWaitQueue.get(resourceId).add(clientId);
		resourceWaitQueueLock.unlock();
	}

	private String returnHeldResourceDetails(String resourceName, String clientID) {
		String resourceInfo = "";

		//first get the inner map based on the id
		clientResourcesLock.lock();
		if (clientResources.containsKey(clientID)) {
			Map<String,Map<String, Integer>> resourceDetails = clientResources.get(clientID);
			clientResourcesLock.unlock();

			for (Map.Entry<String,Map<String, Integer>> resourceNameMapping : resourceDetails.entrySet()) {
				if(resourceNameMapping.getKey().equals(resourceName)) {
					Map<String, Integer> resourceIDMap = resourceNameMapping.getValue();
					for (Map.Entry<String, Integer> idAndDuration : resourceIDMap.entrySet()) {
						resourceInfo += " " + idAndDuration.getKey() + " " + idAndDuration.getValue() + ",";
					}
				}
			}
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
		//first get the inner map based on the id
		clientResourcesLock.lock();

		if (clientResources.containsKey(coordinatorID)) {
			Map<String, Map<String, Integer>> resourceDetails = clientResources.get(coordinatorID);
			for (Map.Entry<String, Map<String, Integer>> entry : resourceDetails.entrySet()) {
				Map<String, Integer> subMap = entry.getValue();
				if (subMap.containsKey(resourceID)) {
					resourcesServerLock.lock();

					//Increase duration if resource already owned by server
					if(resourcesServer.get(entry.getKey()).containsKey(resourceID)){
						resourcesServer.get(entry.getKey()).put(resourceID, resourcesServer.get(entry.getKey()).get(resourceID) + subMap.get(resourceID));
					}
					//Server already has resources of the same resource type, so just add the resource back
					else if(resourcesServer.containsKey(entry.getKey())){
						resourcesServer.get(entry.getKey()).put(resourceID,subMap.get(resourceID));
					}
					//Add resource with name to server if not present there
					else resourcesServer.put(entry.getKey(),subMap);
					resourcesServerLock.unlock();


					// Check if there are any clients waiting for this resource
					resourceWaitQueueLock.lock();
					if (resourceWaitQueue.containsKey(resourceID) && !resourceWaitQueue.get(resourceID).isEmpty()) {
						String nextClient = resourceWaitQueue.get(resourceID).poll(); //Retrieves and removes the head of the waiting queue and adds resource to it
						if (nextClient != null) {
							clientResources.putIfAbsent(nextClient, new HashMap<>());
							clientResources.get(nextClient).putIfAbsent(entry.getKey(), new HashMap<>());
							clientResources.get(nextClient).get(entry.getKey()).put(resourceID, subMap.get(resourceID));
						}
					}
					resourceWaitQueueLock.unlock();

					subMap.remove(resourceID);
					// Clean up if the subMap becomes empty
					if (subMap.isEmpty()) {
						resourceDetails.remove(entry.getKey());
					}
					// Clean up if the resourceDetails becomes empty
					if (resourceDetails.isEmpty()) {
						clientResources.remove(coordinatorID);
					}
					clientResourcesLock.unlock();

					return "Successfully returned resource to server";
				}
			}
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

	public static class ServerThread implements Runnable {
		private int port;
		private int udpPort;
		private String serverName;

		public ServerThread(int port, String serverName, int udpPort) {
			this.port = port;
			this.serverName = serverName;
			this.udpPort = udpPort;

		}

		@Override
		public void run() {
			try {
				Registry registry = LocateRegistry.createRegistry(port);
				Server server = new Server(udpPort, serverName.substring(0, 3));
				registry.bind(serverName, server);
				System.out.println("Server:" + serverName + "running on ports:" + port + "and" + udpPort);
				server.startUDPListener();

			} catch (RemoteException | AlreadyBoundException e) { //This prints what error happened, either during remote method call or registry already bound
				e.printStackTrace();
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}
	}

	public static void main(String[] args) throws RemoteException {
		List<Thread> serverThreads = new ArrayList<>();

		// Start 3 servers for the 3 cities
		Thread serverMTL = new Thread(new ServerThread(7777,"MTL", 7877));
		serverThreads.add(serverMTL);
		serverMTL.start();

		Thread serverQUE = new Thread(new ServerThread(7778,"QUE", 7878));
		serverThreads.add(serverQUE);
		serverQUE.start();

		Thread serverSHE = new Thread(new ServerThread(7779,"SHE", 7879));
		serverThreads.add(serverSHE);
		serverSHE.start();

		for (Thread t : serverThreads) {
			try {
				t.join(); // Wait for all server threads to finish execution
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
}