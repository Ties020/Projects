package nl.vu.cs.softwaredesign.Levels;

import java.util.*;

public class Level {
    public String levelName;
    public Integer difficulty;
    public LinkedHashMap<String,String> flashcards; //Do we need a class for flashcards? Where each flashcard has an id, to edit the right one.
    public Integer passingScore;

    public Level() {
        flashcards = new LinkedHashMap<>();
    }

    public static Level getInstance(String levelNameToFind){
        for(Level levelInstance: LevelManager.getInstance().levels) {
            if (levelInstance != null && levelInstance.levelName.equals(levelNameToFind)) {
                return levelInstance;
            }
        }
        System.out.print("Couldn't find level");
        return null;
    }
    public void addFlashcard(String word, String translation) {
        flashcards.put(word, translation);
    }
    public void setLevelName(String levelName){ this.levelName = levelName;}
    public void setPassingScore(Integer newPassingScore){this.passingScore = newPassingScore;}
    public void setDifficulty(Integer newDifficulty){this.difficulty = newDifficulty;}
    public String getPassingScore() {
        if (this.passingScore == null || this.passingScore == 0) {
            return "None";
        } else {
            return this.passingScore.toString();
        }
    }
}
