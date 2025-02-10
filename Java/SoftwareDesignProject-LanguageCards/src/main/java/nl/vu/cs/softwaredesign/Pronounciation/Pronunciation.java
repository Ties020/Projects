package nl.vu.cs.softwaredesign.Pronounciation;

import java.io.IOException;
import java.io.BufferedInputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URL;
import javazoom.jl.player.Player;
import java.io.ByteArrayInputStream;
import java.util.concurrent.ConcurrentHashMap;
import java.io.ByteArrayOutputStream;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;
import java.util.Map;

public class Pronunciation {
    private static final Map<String, byte[]> preloadedAudioData = new ConcurrentHashMap<>();

    private static final String API_KEY = "ccdc2706-0319-4715-836e-76acd6a131b7";
    private static final String API_URL = "https://www.dictionaryapi.com/api/v3/references/spanish/json/";
    private static final String AUDIO_BASE_URL = "https://media.merriam-webster.com/audio/prons/es/me/mp3/";

    public static void preloadAudioFiles(List<String> words) {
        ExecutorService executor = Executors.newCachedThreadPool();
        for (String word : words) {
            executor.submit(() -> {
                try {
                    String audioUrl = fetchAudioFileName(word);
                    if (audioUrl != null) {
                        URL url = new URL(audioUrl);
                        ByteArrayOutputStream out = new ByteArrayOutputStream();
                        try (BufferedInputStream in = new BufferedInputStream(url.openStream())) {
                            byte[] buffer = new byte[1024];
                            int bytesRead;
                            while ((bytesRead = in.read(buffer)) != -1) {
                                out.write(buffer, 0, bytesRead);
                            }
                            preloadedAudioData.put(word, out.toByteArray());
                        }
                    }
                } catch (Exception e) {
                    System.err.println("Failed to preload audio for word: " + word);
                }
            });
        }
        executor.shutdown();
    }

    private static String fetchAudioFileName(String word) {
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(API_URL + word + "?key=" + API_KEY))
                .build();

        try {
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            String responseBody = response.body();

            String audioKey = "\"audio\":\"";
            int audioStartIndex = responseBody.indexOf(audioKey);
            if (audioStartIndex != -1) {
                audioStartIndex += audioKey.length();
                int audioEndIndex = responseBody.indexOf("\"", audioStartIndex);
                String audioFileName = responseBody.substring(audioStartIndex, audioEndIndex);

                return buildAudioURL(audioFileName);
            }
        } catch (IOException | InterruptedException e) {
            System.err.println("Error fetching the audio file name: " + e.getMessage());
        }
        return null;
    }

    private static String buildAudioURL(String audioFileName) {
        String subdirectory = audioFileName.startsWith("bix") ? "bix" :
                audioFileName.startsWith("gg") ? "gg" :
                        audioFileName.matches("^[0-9].*") ? "number" :
                                audioFileName.substring(0, 1);
        return AUDIO_BASE_URL + subdirectory + "/" + audioFileName + ".mp3";
    }

    public static boolean playWordSound(String word) {
        try {
            byte[] audioData = preloadedAudioData.get(word);
            if (audioData != null) {
                // Play audio from preloaded data
                try (ByteArrayInputStream bis = new ByteArrayInputStream(audioData)) {
                    Player player = new Player(bis);
                    player.play();
                    return true; // Successfully played the audio from preloaded data
                }
            } else {
                String audioUrl = fetchAudioFileName(word);
                if (audioUrl != null) {
                    URL url = new URL(audioUrl);
                    try (BufferedInputStream bis = new BufferedInputStream(url.openStream())) {
                        Player player = new Player(bis);
                        player.play();
                        return true; // Successfully played the audio after fetching it
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error playing the audio file for word '" + word + "': " + e.getMessage());
        }
        return false; // An error occurred or the audio data was not found, thus not played
    }
}
