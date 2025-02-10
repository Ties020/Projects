package nl.vu.cs.softwaredesign.UserManagement;
import nl.vu.cs.softwaredesign.Observer;
import nl.vu.cs.softwaredesign.Subject;

import java.util.ArrayList;

//use Singleton pattern for this since there will ever only be 1 instance
public class User implements Subject {
    private static User instance;
    public String name;
    public ArrayList<String>achievements;
    private ArrayList<Observer> observers = new ArrayList<>();
    public int xpLevel;
    public int points;
    static {
        try {
            instance = new User();
        } catch (Exception e) {
            throw new RuntimeException("Exception occurred in creating singleton instance");
        }
    }
    public static User getInstance() {
        return instance;
    }
    public void initializeUser(String name, ArrayList<String> achievements, int xpLevel, int points) {
        this.name = name;
        this.achievements = achievements;
        this.xpLevel = xpLevel;
        this.points = points;
    }
    public void earnAchievement(){
        /*this is checked everytime the user opens achievements screen, so add all the achievements the user has achieved,
          it is also modified everytime the user finishes a level, so check if there is only the userOnboarding achievement
          if there are more achievements, this means that the user has not played a level in-between checking the achievements
          screen last time. In this case don't do anything.
         */
        if (achievements.size() == 1) {
            if (points >= 10) {
                achievements.add("Achieved 10 points!");
            }
            if (points >= 50) {
                achievements.add("Achieved 50 points!");
            }
            if (points >= 100) {
                achievements.add("Achieved 100 points!");
            }
            if (xpLevel >= 10) {
                achievements.add("Achieved xpLevel 10!");
            }
            if (xpLevel >= 30) {
                achievements.add("Achieved xpLevel 30!");
            }
        }
        notifyObservers();
    }
    @Override
    public void attach(Observer observer) {
        observers.add(observer);
    }
    @Override
    public void notifyObservers() {
        for (Observer observer : observers) {
            observer.update(achievements  );
        }
    }
}
