package nl.vu.cs.softwaredesign.Levels;
//Singleton pattern is applied here since we only need 1 instance of LevelManager

import java.util.ArrayList;

public class LevelManager {
    private static LevelManager instance;
    public Level[] levels;
    public int maxLevels = 50;
    public int currNumLevels = 0;

    //private constructor to prevent instantiation from outside
    private LevelManager() {
        levels = new Level[maxLevels];  //50 is max number of levels, need to put conditions later
    }

    public static LevelManager getInstance() {
        if (instance == null) {
            instance = new LevelManager();
        }
        return instance;
    }

    public String[] getLevelNames(){
        ArrayList<String> levelNames = new ArrayList<>();
        for(int i = 0; i < maxLevels; i++){
            if (levels[i] != null) {
                levelNames.add(levels[i].levelName);
            }
        }
        return levelNames.toArray(new String[levelNames.size()]);

    }

    public void addLevel(Level newLevel) {
        for (int i = 0; i < levels.length; i++) {
            if (levels[i] == null) {
                levels[i] = newLevel;
                currNumLevels ++;
                return;
            }
        }
        throw new IllegalStateException("Maximum number of levels reached.");
    }

    public void deleteLevel(Level levelToDelete) {
        for (int i=0; i<levels.length; i++){
            if(levels[i] != null && levels[i].levelName.equals(levelToDelete.levelName )){//need different comparator to compare all fields instead of only the name, ez solution: use id per level
                    levels[i] = null;
                    levelToDelete = null;   //delete both the instance of the level, and its entry in the levels array
                    currNumLevels --;
                    return;
            }
        }
        System.out.print("Level not found");
    }
}
