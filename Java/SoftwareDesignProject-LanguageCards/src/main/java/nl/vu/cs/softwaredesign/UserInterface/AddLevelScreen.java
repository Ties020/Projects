package nl.vu.cs.softwaredesign.UserInterface;

import javax.swing.*;

public class AddLevelScreen extends TemplateLevelScreen {
    public AddLevelScreen(JFrame previousFrame) {
        super(previousFrame,null);
        super.setTitle("Add level");
    }

    @Override
    public void handleSaveFlashcard(){
        //In contrast to the EditLevelScreen, here new flashcard can't be edited after adding them. This can be strictly done by editing the level
        JLabel wordLabel = new JLabel(wordField.getText());
        JLabel translationLabel = new JLabel(translationField.getText());

        wordLabel.setBounds(100, yCoordAddFlashcardButton, 100, 50);
        translationLabel.setBounds(200, yCoordAddFlashcardButton, 100, 50);

        panel.add(wordLabel);
        panel.add(translationLabel);

        wordField.setText("Enter word");
        translationField.setText("Enter translation");

        panel.remove(wordField);
        panel.remove(translationField);
    }
}
