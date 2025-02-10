import java.io.Serializable;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Client implements Serializable {
	public String clientID;
	private static List<List<String>> logfilesClient = new ArrayList<>();


	public static void addLogfileClient(LocalDateTime currentTime, String resourceName, List<String> requestParameters, String responseServer) {
		String successServer;
		if(responseServer.contains("couldn't") || responseServer.contains("No")){
			successServer = "failure";
		}
		else{
			successServer = "success";
		}

		List<String> logfile = new ArrayList<>();
		logfile.add(currentTime.toString());
		logfile.add(resourceName);
		logfile.addAll(requestParameters);
		logfile.add(successServer);
		logfile.add(responseServer);

		logfilesClient.add(logfile);
	}

	public Client(String clientID){
		super();
		this.clientID = clientID;
	}

	public static boolean isValidId(String Id, boolean usedForClientID) {
		String digits = Id.substring(3);

		if(usedForClientID){
			if (Id.length() != 8) return false;

			char role = Id.charAt(3);
			if (role != 'R' && role != 'C' && role != 'r' && role != 'c') return false;

			digits = Id.substring(4);
		}

		if (!digits.matches("\\d{4}")) return false;

		String prefix = Id.substring(0, 3);
		if (!prefix.equalsIgnoreCase("MTL") && !prefix.equalsIgnoreCase("SHE") && !prefix.equalsIgnoreCase("QUE")) {
			return false;
		}

        return true;
    }

	public static boolean isValidResName(String name){
        return name.equalsIgnoreCase("AMBULANCE") || name.equalsIgnoreCase("FIRETRUCK") || name.equalsIgnoreCase("PERSONNEL");
    }

	public static void main(String... strings) {
		Scanner scan = new Scanner(System.in);

		try {
			Registry registry = LocateRegistry.getRegistry("localhost", 1099);
			ClientIDListInterface clientIDList = (ClientIDListInterface) registry.lookup("ClientIDList");
			System.out.println("Enter your unique ID with 3 prefix letters MTL, SHE, or QUE followed by R or C meaning responder or coordinator, followed by 4 digits:");
			String clientId = scan.next();
			while(!isValidId(clientId,true) || clientIDList.getClientIDs().contains(clientId)){
				System.out.println("ID is not valid, try again");
				clientId = scan.next();
			}

			clientIDList.addClientID(clientId);

			String serverName;
			int port;
			// Match the client to the corresponding server and port
			switch (clientId.substring(0,3).toUpperCase()) {
				case "MTL" :
					serverName = "MTL";
					port = 7777;
					break;
				case "QUE":
					serverName = "QUE";
					port = 7778;
					break;
				case "SHE":
					serverName = "SHE";
					port = 7779;
					break;
				default:
					return;
			}
			Registry registryMain = LocateRegistry.getRegistry("localhost", port);
			ServerInterface stub = (ServerInterface) registryMain.lookup(serverName);

			if(clientId.charAt(3) == 'R' || clientId.charAt(3) == 'r') {
				while(true){
					System.out.println("Enter 'e' to disconnect or other key to continue with operations:");
					String command = scan.next();
					if (command.equalsIgnoreCase("e")) {
						System.out.println("Disconnecting client...");
						clientIDList.removeClientID(clientId);
						break;
					}

					System.out.println("What operation (add, rem, or list)?");
					String operation = scan.next();

					String resourceId = "";
					String resourceName = "";
					Integer duration = null;
					String responseServer = "";

					switch (operation){
						case "add":
							System.out.println("ID (like MTL1111 or SHE1384):");
							resourceId = scan.next();
							while(!isValidId(resourceId,false)){
								System.out.println("Invalid id.");
								resourceId = scan.next();
							}

							System.out.println("Name (ambulance, firetruck, or personnel):");
							resourceName = scan.next();
							while(!isValidResName(resourceName)){
								System.out.println("Invalid name.");
								resourceName = scan.next();
							}

							System.out.println("Duration:");
							while (!scan.hasNextInt()) {
								System.out.println("Input is not a valid integer.");
								scan.next();
							}
							duration = scan.nextInt(); // read the input as an integer
							responseServer = stub.addResource(resourceId,resourceName,duration);
							break;
						case "rem":
							System.out.println("ID:");
							resourceId = scan.next();
							System.out.println("remove res or decrease duration (R or D)");
							String choice = scan.next();
							if(choice.equals("r")) responseServer = stub.removeResource(resourceId,null);

							else{
								System.out.println("Decrease duration by:");
								duration = scan.nextInt();
								responseServer = stub.removeResource(resourceId,duration);
                            }
                            break;
                        case "list":
							System.out.println("Name (ambulance, firetruck, or personnel):");
							resourceName = scan.next();
							while(!isValidResName(resourceName)){
								System.out.println("Invalid name.");
								resourceName = scan.next();
							}
							responseServer = stub.listResourceAvailability(resourceName);
							break;
						default:
							break;
					}


					//Now we add the logfile that the server has to the client's own array
					List<String> requestParameters = new ArrayList<>();
					requestParameters.add(resourceId);
					requestParameters.add(resourceName);
					if(duration != null) requestParameters.add(duration.toString());
					else requestParameters.add(null);


					addLogfileClient(LocalDateTime.now(),resourceName,requestParameters, responseServer);
					System.out.println(responseServer);

				}
			}

			else if (clientId.charAt(3) == 'C' || clientId.charAt(3) == 'c') {
				while(true){
					System.out.println("Enter 'e' to disconnect or other key to continue with operations:");
					String command = scan.next();
					if (command.equalsIgnoreCase("e")) {
						System.out.println("Disconnecting client...");
						clientIDList.removeClientID(clientId);
						break;
					}

					System.out.println("What operation (req, find, or return)?");
					String operation = scan.next();

					String resourceId = "";
					String resourceName = "";
					Integer duration = null;
					String responseServer = "";

					switch (operation){
						case "req":
							System.out.println("ID (like MTL1111 or SHE1384):");
							resourceId = scan.next();
							while(!isValidId(resourceId,false)){
								System.out.println("Invalid id.");
								resourceId = scan.next();
							}

							System.out.println("Duration:");
							while (!scan.hasNextInt()) {
								System.out.println("Input is not a valid integer.");
								scan.next();
							}
							duration = scan.nextInt();
							responseServer = stub.requestResource(clientId,resourceId,duration);

							if(responseServer.contains("Would you like to be added to the queue")) {
								System.out.println("Enter 'yes' to be added to the queue, 'no' to skip:");
								String queueResponse = scan.next();
								if (queueResponse.equalsIgnoreCase("yes")) {
									stub.addToQueue(clientId, resourceId);
									responseServer = "You have been added to the queue.";
								} else {
									responseServer = "You chose not to join the queue.";
								}
							}

							break;
						case "find":
							System.out.println("Name (ambulance, firetruck, or personnel):");
							resourceName = scan.next();
							while(!isValidResName(resourceName)){
								System.out.println("Invalid name.");
								resourceName = scan.next();
							}
							responseServer = stub.findResource(clientId,resourceName);
							break;
						case "return":
							System.out.println("ID (like MTL1111 or SHE1384):");
							resourceId = scan.next();
							while(!isValidId(resourceId,false)){
								System.out.println("Invalid id.");
								resourceId = scan.next();
							}
							responseServer = stub.returnResource(clientId,resourceId);
							break;
						default:
							break;
					}

					List<String> requestParameters = new ArrayList<>();
					requestParameters.add(resourceId);
					requestParameters.add(resourceName);
					if(duration != null) requestParameters.add(duration.toString());
					else requestParameters.add(null);

					addLogfileClient(LocalDateTime.now(),resourceName,requestParameters, responseServer);
					System.out.println(responseServer);


				}
			}
		} catch (Exception e) {
			System.out.println(e.getMessage());

		}

		scan.close();


	}


}
