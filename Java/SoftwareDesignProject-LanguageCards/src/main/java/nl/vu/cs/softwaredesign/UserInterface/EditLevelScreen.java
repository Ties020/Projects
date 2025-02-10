package nl.vu.cs.softwaredesign.UserInterface;

import nl.vu.cs.softwaredesign.Levels.Level;
import nl.vu.cs.softwaredesign.UserInterface.TemplateLevelScreen;

import javax.swing.*;
import java.awt.*;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.util.Map;

public class EditLevelScreen extends TemplateLevelScreen {
    public EditLevelScreen(JFrame previousframe, Level levelToEdit) {
        super(previousframe, levelToEdit);
        super.setTitle("Editing level");
    }

    @Override
    public void createEditableFlashcards() {
        //Go through existing flashcards in the level and output them to the screen for the user to edit
        for (Map.Entry<String, String> entry : currLevelToEdit.flashcards.entrySet()) {
            String translation = entry.getValue();
            String key = entry.getKey();

            wordField = new JTextField(key);
            translationField = new JTextField(translation);
            JButton deleteCardButton = new JButton("Delete");

            wordField.addFocusListener(new FlashcardFocusListener(wordField, key, translation, true));
            translationField.addFocusListener(new FlashcardFocusListener(translationField, key, translation, false));
            deleteCardButton.addActionListener(new DeleteCardListener(wordField, translationField, key, deleteCardButton));

            wordField.setBounds(100, yCoordAddFlashcardButton, 100, 50);
            translationField.setBounds(200, yCoordAddFlashcardButton, 100, 50);
            deleteCardButton.setBounds(300, yCoordAddFlashcardButton, 100, 50);

            Dimension preferredSize = panel.getPreferredSize();
            preferredSize.height += 50; //Increase height for newly added components
            panel.setPreferredSize(preferredSize);

            panel.add(deleteCardButton);
            panel.add(wordField);
            panel.add(translationField);

            yCoordAddFlashcardButton += 50;
        }

    }
     class FlashcardFocusListener implements FocusListener {
         JTextField textField;
         boolean isKey;
         private final String oldkey;
         private final String oldtranslation;
        public FlashcardFocusListener(JTextField textField, String oldKey, String oldTranslation, boolean isKeyField) {
            this.textField = textField;
            this.oldkey = oldKey;
            this.oldtranslation = oldTranslation;
            this.isKey = isKeyField;
        }

        @Override
        public void focusGained(FocusEvent e) {
            //don't do anything
        }

        @Override
        public void focusLost(FocusEvent e) {
            String newText = textField.getText();
            if (isKey) { //change flashcard's key
                currLevelToEdit.flashcards.remove(oldkey);
                currLevelToEdit.flashcards.put(newText,oldtranslation);
            } else { //change flashcard's translation
                currLevelToEdit.flashcards.put(oldkey,newText);
            }
        }
    }

    @Override
    public void handleDoneButton(){
        currLevelToEdit.setLevelName(enterLevelName.getText());
        setVisible(false);
        previousFrame.setVisible(true);
    }
}
