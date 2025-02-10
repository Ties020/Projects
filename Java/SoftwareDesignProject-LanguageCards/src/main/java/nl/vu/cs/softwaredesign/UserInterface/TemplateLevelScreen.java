package nl.vu.cs.softwaredesign.UserInterface;
import nl.vu.cs.softwaredesign.Levels.Level;
import nl.vu.cs.softwaredesign.Levels.LevelManager;
import nl.vu.cs.softwaredesign.Persistence.PersistanceManager;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.util.Map;
import java.util.Objects;

public class TemplateLevelScreen extends JFrame {
    protected final JFrame previousFrame;
    protected JPanel panel = new JPanel();
    protected transient Level currLevelToEdit = new Level();
    private static final String ENTER_LVLNAME_TEXT = "Enter level name";
    private static final String ENTER_WORD_TEXT = "Enter word";
    private static final String ENTER_TRANS_TEXT = "Enter translation";
    private static final String ENTER_DIFF_TEXT ="Enter difficulty from 1-10";
    private static final String ENTER_PASS_TEXT = "Enter passing score";

    protected JTextField enterLevelName = new JTextField(ENTER_LVLNAME_TEXT);
    protected JTextField difficultyField = new JTextField(ENTER_DIFF_TEXT);
    protected JTextField passingScoreField = new JTextField(ENTER_PASS_TEXT);
    protected JTextField wordField;
    protected JTextField translationField;
    protected JButton saveFlashcardButton = new JButton("Save");
    private final JButton addFlashcardButton = new JButton("Add flashcard");
    protected int yCoordAddFlashcardButton = 100;
    public TemplateLevelScreen(){this.previousFrame = null;}

    public TemplateLevelScreen(JFrame previousFrame, Level levelToEdit) {  //Common initialization steps for adding/editing a level

        if(levelToEdit != null) currLevelToEdit = levelToEdit;
        this.previousFrame = previousFrame;
        saveFlashcardButton.addActionListener(new SaveFlashcardListener());

        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        this.setSize(450, 400);
        this.setLocationRelativeTo(null);
        this.setVisible(true);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        this.getContentPane().setLayout(new BoxLayout(getContentPane(), BoxLayout.Y_AXIS));  //Boxlayout is needed for scrolling
        this.setResizable(false);

        panel.setPreferredSize(new Dimension(450,400)); //same size as window frame to fit all GUI components in 
        JScrollPane scrollPane = new JScrollPane(panel, ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED, ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
        this.getContentPane().add(scrollPane);

        enterLevelName.setForeground(Color.BLACK);
        enterLevelName.addFocusListener(new LevelNameFocusListener());
        enterLevelName.setBounds(0,0,100,50);
        if(currLevelToEdit != null  && currLevelToEdit.levelName != null) enterLevelName.setText(currLevelToEdit.levelName);

        if (currLevelToEdit != null && currLevelToEdit.passingScore != null) passingScoreField.setText(currLevelToEdit.passingScore.toString());
        passingScoreField.setBounds(100,0,200,50);
        passingScoreField.addFocusListener(new PassingScoreFocusListener());

        difficultyField.setBounds(100,50,200,50);
        if(currLevelToEdit != null && currLevelToEdit.difficulty != null) difficultyField.setText(currLevelToEdit.difficulty.toString());

        difficultyField.addFocusListener(new DifficultyFieldFocusListener());

        ImageIcon originalIcon = new ImageIcon(Objects.requireNonNull(EditLevelScreen.class.getResource("/images/done.jpg")));
        Image scaledImage = originalIcon.getImage().getScaledInstance(100, 70, Image.SCALE_SMOOTH);
        ImageIcon scaledIcon = new ImageIcon(scaledImage);

        //Button used for saving level and returning to levels menu screen
        JButton doneAddingButton = new JButton();
        doneAddingButton.addActionListener(new DoneButtonListener());
        doneAddingButton.setBounds(300,0,100,50);
        doneAddingButton.setBorderPainted(false);
        doneAddingButton.setIcon(scaledIcon);

        JButton loadFlashcardButton = new JButton("Load flashcard");
        loadFlashcardButton.setBounds(0, 50, 100, 50);
        loadFlashcardButton.setFont(new Font("Arial", Font.PLAIN, 10));
        loadFlashcardButton.addActionListener(new LoadFlashcardListener());

        JButton saveFlashcardButtonPersistence = new JButton("Save flashcard");
        saveFlashcardButtonPersistence.setBounds(0, 100, 100, 50);
        saveFlashcardButtonPersistence.setFont(new Font("Arial", Font.PLAIN, 10));
        saveFlashcardButtonPersistence.addActionListener(new SaveFlashcardListenerPersistence());

        if(levelToEdit != null) createEditableFlashcards(); //This is only done when editing a level, since there are no flashcards present when adding a new level
        wordField = new JTextField(ENTER_WORD_TEXT);
        translationField = new JTextField(ENTER_TRANS_TEXT);

        addFlashcardButton.setBounds(100,yCoordAddFlashcardButton,200,50);
        addFlashcardButton.addActionListener(new AddFlashcardListener());
        
        panel.setLayout(null);
        panel.add(difficultyField);
        panel.add(passingScoreField);
        panel.add(enterLevelName);
        panel.add(addFlashcardButton);
        panel.add(doneAddingButton);
        panel.add(loadFlashcardButton);
        panel.add(saveFlashcardButton);
        panel.add(saveFlashcardButtonPersistence);
        panel.revalidate();
        panel.repaint();
    }

    public void createEditableFlashcards(){
        //to be implemented by EditLevelScreen
    }

    public class LevelNameFocusListener implements FocusListener {

        @Override
        public void focusGained(FocusEvent e) {  //Remove default text when user tries to type in the field
            if (enterLevelName.getText().equals(ENTER_LVLNAME_TEXT)) {
                enterLevelName.setText("");
            }
        }
        @Override
        public void focusLost(FocusEvent e) {
            if (enterLevelName.getText().trim().isEmpty()) {  //Put default text back
                enterLevelName.setText(ENTER_LVLNAME_TEXT);
            } else {
                currLevelToEdit.setLevelName(enterLevelName.getText());
            }
        }
    }

    public class DifficultyFieldFocusListener implements FocusListener {
        @Override
        public void focusGained(FocusEvent e) {
            if (difficultyField.getText().equals(ENTER_DIFF_TEXT)) {
                difficultyField.setText("");
            }
        }
        @Override
        public void focusLost(FocusEvent e) {
            if (difficultyField.getText().trim().isEmpty()) difficultyField.setText(ENTER_DIFF_TEXT);
            
            //Check whether given difficulty is a number from 1-10 and set that as the new difficulty
            else if (isNumeric(difficultyField.getText().trim())) {
                int difficulty = Integer.parseInt(difficultyField.getText());
                if (difficulty >= 1 && difficulty <= 10) currLevelToEdit.setDifficulty(difficulty);
                else
                    JOptionPane.showMessageDialog(null, "Difficulty level must be from 1-10", "Change difficulty", JOptionPane.PLAIN_MESSAGE);
            } else
                JOptionPane.showMessageDialog(null, "Difficulty must be an integer", "Change difficulty", JOptionPane.PLAIN_MESSAGE);
        }
    }

    public class PassingScoreFocusListener implements FocusListener {

        @Override
        public void focusGained(FocusEvent e) {
            if (passingScoreField.getText().equals(ENTER_PASS_TEXT)) passingScoreField.setText("");
        }

        @Override
        public void focusLost(FocusEvent e) {
            if (passingScoreField.getText().trim().isEmpty()) passingScoreField.setText(ENTER_PASS_TEXT);

            else if (isNumeric(passingScoreField.getText().trim())) {
                currLevelToEdit.setPassingScore(Integer.parseInt(passingScoreField.getText()));
            } else
                JOptionPane.showMessageDialog(null, "Passing score must be an integer", "Change passing score", JOptionPane.PLAIN_MESSAGE);
        }
    }

    private class WordfieldFocus implements FocusListener {
        @Override
        public void focusGained(FocusEvent e) {  //Remove text when user tries to type in the field
            if (wordField.getText().equals(ENTER_WORD_TEXT)) {
                wordField.setText("");
            }
        }

        @Override
        public void focusLost(FocusEvent e) { //Put default text back
            if (wordField.getText().trim().isEmpty()) {
                wordField.setText(ENTER_WORD_TEXT);
            }
        }
    }

    private class TranslationfieldFocus implements FocusListener {
        @Override
        public void focusGained(FocusEvent e) {
            if (translationField.getText().equals(ENTER_TRANS_TEXT)) {
                translationField.setText("");
            }
        }

        @Override
        public void focusLost(FocusEvent e) {
            if (translationField.getText().trim().isEmpty()) {
                translationField.setText(ENTER_TRANS_TEXT);
            }
        }
    }


    // Concrete subclass within TemplateLevelScreen
    class DoneButtonListener implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            handleDoneButton();
        }
    }

    public void handleDoneButton() {  //This implementation is for adding a new level, editing overrides this method accordingly
        if (currLevelToEdit.levelName != null) {
            currLevelToEdit.setLevelName(enterLevelName.getText());
            LevelManager.getInstance().addLevel(currLevelToEdit);
            setVisible(false);
            previousFrame.setVisible(true);
        }
    }

    private void createNewAddFlashcardButton(){
        yCoordAddFlashcardButton += 50;
        addFlashcardButton.setBounds(100,yCoordAddFlashcardButton,200,50);
        panel.add(addFlashcardButton);
        panel.revalidate();
        panel.repaint();
    }

    public class AddFlashcardListener implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent actionEvent) {
            panel.remove(addFlashcardButton);

            saveFlashcardButton.setBounds(300, yCoordAddFlashcardButton, 100, 50);

            wordField.setBounds(100, yCoordAddFlashcardButton, 100, 50);
            wordField.addFocusListener(new WordfieldFocus());

            translationField.addFocusListener(new TranslationfieldFocus());
            translationField.setBounds(200, yCoordAddFlashcardButton, 100, 50);

            panel.add(saveFlashcardButton);
            panel.add(wordField);
            panel.add(translationField);

            Dimension preferredSize = panel.getPreferredSize();
            preferredSize.height += 50; // increase height for newly added components
            panel.setPreferredSize(preferredSize);
            panel.revalidate();
            panel.repaint();

            //Scroll to the bottom, line below is to get the scrollbar since Scrollpane is in another class
            JScrollBar verticalScrollBar = ((JScrollPane) SwingUtilities.getAncestorOfClass(JScrollPane.class, panel)).getVerticalScrollBar();
            verticalScrollBar.setValue(verticalScrollBar.getMaximum());
        }
    }

    class DeleteCardListener implements ActionListener{
        JTextField keyField;
        JTextField translationField;
        JButton removebutton;
        String oldKey;

        public DeleteCardListener(JTextField wordfield, JTextField transfield, String oldkey, JButton deletebutton) {
            this.keyField = wordfield;
            this.translationField = transfield;
            this.oldKey = oldkey;
            this.removebutton = deletebutton;
        }

        @Override
        public void actionPerformed(ActionEvent actionEvent) {
            currLevelToEdit.flashcards.remove(oldKey);

            //move all the components below the to be deleted one up
            Component[] components = panel.getComponents();
            for (Component component : components) {
                int newY = component.getY();
                if (component.getY() > removebutton.getY()) {
                    newY -= 50;
                    component.setLocation(component.getX(), newY);
                }
            }

            panel.remove(keyField);
            panel.remove(translationField);
            panel.remove(removebutton);

            //Adjust the y-coordinate for the next flashcard addition
            yCoordAddFlashcardButton -= 50;

            Dimension preferredSize = panel.getPreferredSize();
            preferredSize.height -= 50; // Decrease height due to deleted components
            panel.setPreferredSize(preferredSize);
            panel.revalidate();
            panel.repaint();

            //Scroll up, code below is to fetch the scrollbar since Scrollpane is in another class
            JScrollBar verticalScrollBar = ((JScrollPane)SwingUtilities.getAncestorOfClass(JScrollPane.class, panel)).getVerticalScrollBar();
            verticalScrollBar.setValue(verticalScrollBar.getMaximum());
        }
    }

    public void handleSaveFlashcard(){ //Used for editing, when adding new level, override this method
        //Make newly added flashcard editable
        JButton deleteCardButton = new JButton("Delete");
        deleteCardButton.setBounds(300,yCoordAddFlashcardButton,100,50);
        deleteCardButton.addActionListener( new DeleteCardListener( wordField, translationField, wordField.getText(), deleteCardButton));

        wordField = new JTextField(ENTER_WORD_TEXT);
        translationField = new JTextField(ENTER_TRANS_TEXT);

        wordField.setBounds(100, yCoordAddFlashcardButton, 100, 50);
        wordField.addFocusListener(new WordfieldFocus());

        translationField.setBounds(200, yCoordAddFlashcardButton, 100, 50);
        translationField.addFocusListener(new TranslationfieldFocus());

        wordField.setText(ENTER_WORD_TEXT);
        translationField.setText(ENTER_TRANS_TEXT);

        panel.add(deleteCardButton);
    }

    private class SaveFlashcardListener implements ActionListener{
        //Register flashcard to the level
        @Override
        public void actionPerformed(ActionEvent actionEvent) {
            if(!wordField.getText().trim().isEmpty() && !wordField.getText().equals(ENTER_WORD_TEXT) && !translationField.getText().trim().isEmpty() && !translationField.getText().equals(ENTER_TRANS_TEXT)) {
                if (!currLevelToEdit.flashcards.containsKey(wordField.getText())) {
                    currLevelToEdit.flashcards.put(wordField.getText(), translationField.getText());

                    handleSaveFlashcard(); // make new flashcard editable when editing, turn into a label when adding new level
                    panel.remove(saveFlashcardButton);
                    panel.revalidate();
                    panel.repaint();
                    createNewAddFlashcardButton();
                }
                else{
                    JOptionPane.showMessageDialog(null, "Word already exists", "Word must be changed", JOptionPane.PLAIN_MESSAGE);
                }
            }

        }
    }


    private class LoadFlashcardListener implements ActionListener{
        @Override
        public void actionPerformed(ActionEvent e) {
            int nOfElements = currLevelToEdit.flashcards.size();
            PersistanceManager.getInstance().loadData(false, currLevelToEdit);
            if (nOfElements == currLevelToEdit.flashcards.size()) {return;}

            Map.Entry<String,String> lastEntry = null;
            for (Map.Entry<String, String> entry : currLevelToEdit.flashcards.entrySet()) {
                lastEntry = entry;
            }

            assert lastEntry != null;
            JLabel wordLabel = new JLabel(lastEntry.getKey());
            JLabel translationLabel = new JLabel(lastEntry.getValue());

            wordLabel.setBounds(100, yCoordAddFlashcardButton, 100, 50);
            translationLabel.setBounds(200, yCoordAddFlashcardButton, 100, 50);

            panel.add(wordLabel);
            panel.add(translationLabel);

            wordField.setText(ENTER_WORD_TEXT);
            translationField.setText(ENTER_TRANS_TEXT);

            panel.remove(wordField);
            panel.remove(translationField);
            panel.remove(saveFlashcardButton);
            panel.revalidate();
            panel.repaint();
            createNewAddFlashcardButton();
        }
    }

    private class SaveFlashcardListenerPersistence implements ActionListener {
        //This is used for exporting the flashcard to files
        @Override
        public void actionPerformed(ActionEvent e) {
            String[] names = currLevelToEdit.flashcards.keySet().toArray(new String[0]);
            JComboBox<String> comboBox = new JComboBox<>(names);
            comboBox.setSelectedIndex(-1);  //no level is selected initially
            JOptionPane.showMessageDialog(null, comboBox, "Select flashcard to save", JOptionPane.PLAIN_MESSAGE);
            String selectedFlashcard = (String) comboBox.getSelectedItem();
            if (selectedFlashcard != null) {
                PersistanceManager.getInstance().saveData(null, false, selectedFlashcard, currLevelToEdit.flashcards.get(selectedFlashcard));
            }
        }
    }

    private static boolean isNumeric(String str) {
        if (str == null || str.isEmpty()) {
            return false;
        }
        for (char c : str.toCharArray()) {
            if (!Character.isDigit(c)) {
                return false;
            }
        }
        return true;
    }

}
