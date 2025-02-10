package nl.vu.cs.softwaredesign.Persistence;


import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.File;
import java.io.IOException;
import java.io.FileWriter;
import javax.swing.*;
import java.nio.file.Files;
import java.nio.file.Paths;

import nl.vu.cs.softwaredesign.Levels.Level;
import nl.vu.cs.softwaredesign.Levels.LevelManager;
import org.json.JSONObject;

public class PersistanceManager extends JFrame {
    private static PersistanceManager instance;
    static {
        try {
            instance = new PersistanceManager();
        } catch (Exception e) {
            throw new RuntimeException("Exception occurred in creating singleton instance");
        }
    }
    public static PersistanceManager getInstance() {
        return instance;
    }

    public void saveData(Level level, boolean isSavingLevel, String word, String translation) {
        //create a file chooser
        JFileChooser fileChooser;
        if (isSavingLevel) {
            fileChooser = new JFileChooser("./src/main/resources/levels");
        }
        else {
            fileChooser = new JFileChooser("./src/main/resources/flashcards");
        }
        //set file selection to files only
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
        fileChooser.setAcceptAllFileFilterUsed(false);
        int result = fileChooser.showSaveDialog(null);
        //if user has chosen a file or created a new one
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            if (isSavingLevel) {
                serialize(level, selectedFile);
            }
            else {
                JSONObject flashcard = new JSONObject();
                flashcard.put("word", word);
                flashcard.put("translation", translation);

                // Write the JSONObject to the file
                try (FileWriter file = new FileWriter(selectedFile)) {
                    file.write(flashcard.toString(4)); // 4 is for pretty print
                    file.flush();
                } catch (IOException e) {
                    JOptionPane.showMessageDialog(null, "Saving data failed!");
                }
            }
        }
    }

    public void loadData(boolean isLoadingLevel, Level currLevel) {
        //create a file chooser
        JFileChooser fileChooser;
        if (isLoadingLevel) {
            fileChooser = new JFileChooser("./src/main/resources/levels");
        }
        else {
            fileChooser = new JFileChooser("./src/main/resources/flashcards");
        }
        fileChooser.setDialogTitle("Select a JSON file!");
        int result = fileChooser.showOpenDialog(this);
        //if user has chosen some file
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            //read from the file
            try {
                //deserialize the file and add it into levels
                if (isLoadingLevel) {
                    LevelManager.getInstance().addLevel(deserialize(selectedFile));
                }
                //if loading flashcard just add it to the level selected
                else {
                    String data = new String(Files.readAllBytes(Paths.get(selectedFile.getAbsolutePath())));
                    JSONObject jsonObject = new JSONObject(data);
                    String word = jsonObject.getString("word");
                    String translation = jsonObject.getString("translation");
                    if (!currLevel.flashcards.containsKey(word)) {
                        currLevel.addFlashcard(word, translation);
                    }
                }
            } catch(Exception e) {
                JOptionPane.showMessageDialog(null, "Loading data failed!");
            }
        }
    }

    private void serialize(Level level, File selectedFile) {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        try {
            //Write the JSON string to the file
            mapper.writeValue(selectedFile, level);
        } catch (IOException exception) {
            System.out.println("serializing level failed!");
        }
    }

    private Level deserialize(File selectedFile) {
        Level level = new Level();  
        try {
            String data = new String(Files.readAllBytes(Paths.get(selectedFile.getAbsolutePath())));
            JSONObject jsonObject = new JSONObject(data);
            level.setLevelName(jsonObject.getString("levelName"));
            level.difficulty = jsonObject.getInt("difficulty");
            level.passingScore = jsonObject.getInt("passingScore");
            JSONObject flashcards = jsonObject.getJSONObject("flashcards");
            //parse flaschcards into dict
            for (String key : flashcards.keySet()) {
                String value = flashcards.getString(key);
                System.out.println(key+ " "+value);
                level.addFlashcard(key, value);
            }
        } catch (IOException exception) {
            System.out.println(exception);
        }
        return level;
    }
}
