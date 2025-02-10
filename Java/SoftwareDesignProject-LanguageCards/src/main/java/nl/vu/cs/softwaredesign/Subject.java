package nl.vu.cs.softwaredesign;

public interface Subject {
        void attach(Observer observer);
        void notifyObservers();
}

