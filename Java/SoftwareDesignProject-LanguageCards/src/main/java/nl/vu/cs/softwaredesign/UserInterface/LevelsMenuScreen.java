package nl.vu.cs.softwaredesign.UserInterface;

import nl.vu.cs.softwaredesign.Levels.Level;
import nl.vu.cs.softwaredesign.Levels.LevelManager;
import nl.vu.cs.softwaredesign.Persistence.PersistanceManager;
import nl.vu.cs.softwaredesign.UserManagement.User;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class LevelsMenuScreen {
    private final PersistanceManager persistanceManager = PersistanceManager.getInstance();
    private final JFrame menuFrame;

    public  LevelsMenuScreen() {
        menuFrame = new JFrame("Levels Menu");
        menuFrame.getContentPane().setBackground(Color.BLUE);

        //Make buttons describing the features that can be used by the user
        JButton returnButton = new JButton("\u2190");
        returnButton.setBounds(0 , 0, 50, 50);
        returnButton.addActionListener(new ReturnListener());

        JButton playLevelButton = new JButton("Play level");
        playLevelButton.setBounds(150, 0, 100, 30);
        playLevelButton.addActionListener(new PlayLevelListener());

        JButton addLevelButton = new JButton("Add level");
        addLevelButton.setBounds(150, 50, 100, 30);
        addLevelButton.addActionListener(new AddLevelListener());

        JButton deleteLevelButton = new JButton("Delete level");
        deleteLevelButton.setBounds(150, 100, 100, 30);
        deleteLevelButton.addActionListener(new DeleteLevelListener());

        JButton editLevelButton = new JButton("Edit level");
        editLevelButton.setBounds(150, 150, 100, 30);
        editLevelButton.addActionListener(new EditLevelListener());

        JButton loadLevelButton = new JButton("Load level");
        loadLevelButton.setBounds(150, 200, 100, 30);
        loadLevelButton.addActionListener(new LoadLevelListener());

        JButton saveLevelButton = new JButton("Save level");
        saveLevelButton.setBounds(150, 250, 100, 30);
        saveLevelButton.addActionListener(new SaveLevelListener());

        menuFrame.setLayout(null); //Set layout to null for absolute positioning
        menuFrame.add(returnButton);
        menuFrame.add(playLevelButton);
        menuFrame.add(addLevelButton);
        menuFrame.add(deleteLevelButton);
        menuFrame.add(editLevelButton);
        menuFrame.add(loadLevelButton);
        menuFrame.add(saveLevelButton);


        menuFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        menuFrame.setSize(400, 400);
        menuFrame.setLocationRelativeTo(null);
        menuFrame.setVisible(true);
        menuFrame.setResizable(false);
    }

    private class PlayLevelListener implements ActionListener {
        public void actionPerformed(ActionEvent e) {
            String[] names = LevelManager.getInstance().getLevelNames();
            if (names.length == 0) {
                JOptionPane.showMessageDialog(null, "There isn't any level to be played!");
                return;
            }
            JComboBox<String> comboBox = new JComboBox<>(names);
            comboBox.setSelectedIndex(-1);  //No level is selected initially

            JOptionPane.showMessageDialog(null, comboBox, "Select level to play", JOptionPane.PLAIN_MESSAGE);

            String selectedLevelName = (String) comboBox.getSelectedItem();

            if (selectedLevelName != null) {
                Level selectedLevel = Level.getInstance(selectedLevelName); //get the level that was selected in the dropbox
                assert selectedLevel != null;
                if (selectedLevel.flashcards.isEmpty()) {
                    JOptionPane.showMessageDialog(null, "Selected level doesn't have any flashcards!");
                }
                //Check if user has the xpLevel to play the level
                else if (selectedLevel.difficulty > User.getInstance().xpLevel) {
                    JOptionPane.showMessageDialog(null, "Your xpLevel is lower than the difficulty of the selected Level!");
                }
                else {
                    new LevelScreen(false, selectedLevel, User.getInstance().name);
                    menuFrame.setVisible(false);
                }
            }
        }
    }

    private class ReturnListener implements ActionListener{
        @Override
        public void actionPerformed(ActionEvent e) {
            menuFrame.setVisible(false);
            new MenuScreen(User.getInstance().name);
        }
    }

    private class AddLevelListener implements ActionListener {
        public void actionPerformed(ActionEvent e) {
            menuFrame.setVisible(false);
            new TemplateLevelScreen();  //Make new instance of template to get unpopulated fields
            new AddLevelScreen(menuFrame);
        }
    }


    private class EditLevelListener implements ActionListener {
        public void actionPerformed(ActionEvent e) {
            String[] names = LevelManager.getInstance().getLevelNames();
            if (names.length == 0) {
                JOptionPane.showMessageDialog(null, "There isn't any level to edit!");
                return;
            }
            JComboBox<String> comboBox = new JComboBox<>(names);
            comboBox.setSelectedIndex(-1);  //no level is selected initially

            JOptionPane.showMessageDialog(null, comboBox, "Select level to edit", JOptionPane.PLAIN_MESSAGE);

            String selectedLevelName = (String) comboBox.getSelectedItem();

            if (selectedLevelName != null) {
                Level selectedLevel = Level.getInstance(selectedLevelName); //get the level that was selected in the dropbox
                new TemplateLevelScreen();
                new EditLevelScreen(menuFrame, selectedLevel);
                menuFrame.setVisible(false);
            }
        }
    }

    private static class DeleteLevelListener implements ActionListener {
        public void actionPerformed(ActionEvent e) {
            String[] names = LevelManager.getInstance().getLevelNames();
            if (names.length == 0) {
                JOptionPane.showMessageDialog(null, "There isn't any level to delete!");
                return;
            }
            JComboBox<String> comboBox = new JComboBox<>(names);
            comboBox.setSelectedIndex(-1);  //No level is selected in the menu initially
            JOptionPane.showMessageDialog(null, comboBox, "Select level to delete", JOptionPane.PLAIN_MESSAGE);

            String selectedLevelName = (String) comboBox.getSelectedItem();
            Level selectedLevel = Level.getInstance(selectedLevelName); //Get the level that was selected in the dropbox

            if (selectedLevelName != null) {
                int confirm = JOptionPane.showConfirmDialog(null, "Are you sure you want to delete level: " + selectedLevelName + "?", "Confirmation", JOptionPane.YES_NO_OPTION);
                if (confirm == JOptionPane.YES_OPTION) {
                    LevelManager.getInstance().deleteLevel(selectedLevel);
                    JOptionPane.showMessageDialog(null, "Level deleted: " + selectedLevelName);
                }
            }
        }
    }

    private class LoadLevelListener implements ActionListener {
        public void actionPerformed(ActionEvent e) {
            persistanceManager.loadData(true, null);
        }
    }

    private class SaveLevelListener implements ActionListener {
        public void actionPerformed(ActionEvent e) {
            String[] names = LevelManager.getInstance().getLevelNames();
            if (names.length == 0) {
                JOptionPane.showMessageDialog(null, "There isn't any level to be saved!");
                return;
            }
            JComboBox<String> comboBox = new JComboBox<>(names);
            comboBox.setSelectedIndex(-1);
            JOptionPane.showMessageDialog(null, comboBox, "Select level to save", JOptionPane.PLAIN_MESSAGE);
            String selectedLevelName = (String) comboBox.getSelectedItem();
            Level selectedLevel = Level.getInstance(selectedLevelName);
            persistanceManager.saveData(selectedLevel, true, null, null);
        }
    }
}

