package nl.vu.cs.softwaredesign.UserInterface;

import nl.vu.cs.softwaredesign.Levels.Level;
import nl.vu.cs.softwaredesign.Pronounciation.Pronunciation;
import nl.vu.cs.softwaredesign.UserManagement.User;

import javax.swing.*;
import java.awt.*;
import javax.swing.BorderFactory;
import javax.swing.border.Border;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.*;

public class LevelScreen {
    private JFrame frame = new JFrame();
    //dict for testing
    private int score = User.getInstance().points;
    private int startingScore = User.getInstance().points;
    private int currLevelScore = 0;
    private String flashcardText;
    private String flashcardTranslationText = "Press show to see translation!";
    private ArrayList<String> flashcardKeys = new ArrayList<String>();
    private int flashcardIndex = 0;
    private JLabel flashcardLabel = new JLabel(flashcardText);
    private JLabel flashcardTranslationLabel = new JLabel(flashcardTranslationText);
    public LevelScreen(boolean userOnboarding, Level selectedLevel, String userName) {
        Map<String, String> dict = new HashMap<>(); //first parse Level, to create this Map!


        //if userOnboarding, then put these hardcoded flashcards inside the dictionary
        if (userOnboarding) {
            // Basic vocabulary
            dict.put("water", "agua");
            dict.put("fire", "fuego");
            dict.put("earth", "tierra");
            dict.put("air", "aire");
            dict.put("sun", "sol");
            dict.put("moon", "luna");
            dict.put("star", "estrella");
            dict.put("sky", "cielo");
            dict.put("rain", "lluvia");

//            // Intermediate vocabulary
            dict.put("freedom", "libertad");
            dict.put("strength", "fuerza");
            dict.put("knowledge", "conocimiento");
            dict.put("happiness", "felicidad");
            dict.put("nature", "naturaleza");
            dict.put("music", "música");
            dict.put("mountain", "montaña");
            dict.put("ocean", "océano");

//            // Advanced vocabulary
            dict.put("wisdom", "sabiduría");
            dict.put("achievement", "logro");
            dict.put("courage", "valentía");
            dict.put("peace", "paz");
            dict.put("justice", "justicia");
            dict.put("honor", "honor");
            dict.put("loyalty", "lealtad");
            dict.put("liberty", "libertad");
            dict.put("tradition", "tradición");
            dict.put("success", "éxito");
            dict.put("adventure", "aventura");
            dict.put("discovery", "descubrimiento");
            dict.put("inspiration", "inspiración");
        }
        //if normal level, selectedLevel.flashcards is the dictionary!
        else {
            dict.putAll(selectedLevel.flashcards);
            //read through the flashcards and parse them into the dict
        }
        //create arrayList from keys
        for (Map.Entry<String, String> me :
                dict.entrySet()) {
            // Printing keys
            System.out.print(me.getKey() + ":");
            System.out.println(me.getValue());
            flashcardKeys.add(me.getKey());
        }
        flashcardText = flashcardKeys.get(flashcardIndex);

        // Now, initiate preloading of audio for the Spanish words
        ArrayList<String> spanishWordsToPreload = new ArrayList<>(dict.values());
        Pronunciation.preloadAudioFiles(spanishWordsToPreload);

        System.out.println("Showing levelScreen!");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setTitle("LevelScreen");
        frame.pack();
        frame.setSize(400,400);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        frame.requestFocusInWindow(); //focus on nothing initially

        JPanel panel = new JPanel(null);
        panel.setPreferredSize(new Dimension(400, 400));

        //panel for title and score, check if user onboarding or normal level
        JPanel titlePanel = new JPanel(null);
        titlePanel.setBounds(0, 0, 400, 50);

        //title
        String title;
        if (userOnboarding) {
            title = "User Onboarding!";
        }
        else {
            title = "Level: "+ selectedLevel.levelName; //when calling this method, caller has to know which level will be played!
        }
        JLabel titleText = new JLabel(title);
        titleText.setBounds(25, 0, 100, 50);

        JLabel scoreText = new JLabel("Score: "+currLevelScore);
        scoreText.setBounds(325, 0, 75, 50);
        if (!userOnboarding) {
            JLabel scoreReqText = new JLabel("Passing Score: " + selectedLevel.getPassingScore());
            scoreReqText.setBounds(150, 0, 150, 50);
            titlePanel.add(scoreReqText);
        }

        titlePanel.add(scoreText);
        titlePanel.add(titleText);

        //flashcard and translation panel
        JPanel flashcardPanel = new JPanel(null);
        flashcardPanel.setBounds(0, 50, 400, 250);

        //flashcard
        flashcardLabel.setBounds(50, 0, 300, 100);
        flashcardLabel.setVerticalAlignment(JLabel.CENTER);
        flashcardLabel.setHorizontalAlignment(JLabel.CENTER);
        flashcardText = flashcardKeys.get(flashcardIndex);
        flashcardLabel.setText(flashcardText);

        //change size of the flashcard text
        Font labelFont = flashcardLabel.getFont();
        flashcardLabel.setFont(new Font(labelFont.getName(), Font.PLAIN, 20));

        //add border
        Border border = BorderFactory.createLineBorder(Color.BLACK);
        flashcardLabel.setBorder(border);

        //translation
        flashcardTranslationLabel.setBounds(50, 125, 300, 100);
        flashcardTranslationLabel.setVerticalAlignment(JLabel.CENTER);
        flashcardTranslationLabel.setHorizontalAlignment(JLabel.CENTER);

        //change size of the flashcard text
        labelFont = flashcardTranslationLabel.getFont();
        flashcardTranslationLabel.setFont(new Font(labelFont.getName(), Font.PLAIN, 20));

        //add border
        border = BorderFactory.createLineBorder(Color.BLACK);
        flashcardTranslationLabel.setBorder(border);

        flashcardPanel.add(flashcardLabel);
        flashcardPanel.add(flashcardTranslationLabel);


        // Buttons Panel where all the buttons will be placed
        JPanel buttonsPanel = new JPanel(null);
        buttonsPanel.setBounds(0, 300, 400, 130);

        JButton wrongButton = new JButton("Wrong");
        wrongButton.setBounds(50 , 10, 80, 30);
        wrongButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                showTranslationText();
                if (flashcardIndex == dict.size()-1) {
                    //create user object
                    finishOnboarding(userName, selectedLevel);
                }
                else {
                    nextFlashcard();
                }
            }
        });

        JButton showButton = new JButton("Show");
        showButton.setBounds(160, 10, 80, 30);
        showButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                System.out.println(dict.get(flashcardText));
                flashcardTranslationText = dict.get(flashcardText);
                flashcardTranslationLabel.setText(flashcardTranslationText);
                flashcardTranslationLabel.revalidate();
                flashcardTranslationLabel.repaint();
            }
        });

        JButton correctButton = new JButton("Correct");
        correctButton.setBounds(270, 10, 80, 30);
        correctButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //change score
                score++;
                currLevelScore++;
                scoreText.setText("Score: "+currLevelScore);
                scoreText.revalidate();
                scoreText.repaint();

                //change the flashcardTranslation back to default
                showTranslationText();

                //increase flashcardIndex and show the nextFlashcard, if last proceed to homeScreen;
                if (flashcardIndex == dict.size()-1) {
                    //create user object
                    finishOnboarding(userName, selectedLevel);
                }
                else {
                    nextFlashcard();
                }
            }
        });

        JButton pronunciationButton = new JButton("Pronunciation");
        pronunciationButton.setBounds(100, 50, 200, 30);
        pronunciationButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                // Check if the translation text is not the placeholder text
                if (!"Press show to see translation!".equals(flashcardTranslationText)) {
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            // Fetch the Spanish translation from the map using the English word
                            String spanishWord = dict.get(flashcardText);
                            // Play the sound for the Spanish word, if it exists
                            if (spanishWord != null && !spanishWord.isEmpty()) {
                                boolean played = Pronunciation.playWordSound(spanishWord);
                                if (!played) {
                                    // Show a popup message if the pronunciation is not available
                                    SwingUtilities.invokeLater(new Runnable() {
                                        @Override
                                        public void run() {
                                            JOptionPane.showMessageDialog(frame, "Translation not recognized, pronunciation not available!", "Pronunciation", JOptionPane.INFORMATION_MESSAGE);
                                        }
                                    });
                                }
                            }
                        }
                    }).start();
                }
            }
        });

        buttonsPanel.add(wrongButton);
        buttonsPanel.add(showButton);
        buttonsPanel.add(correctButton);
        buttonsPanel.add(pronunciationButton);

        // Add buttonsPanel to the main panel
        panel.add(buttonsPanel);
        panel.add(flashcardPanel);
        panel.add(titlePanel);

        // Add the main panel to the frame
        frame.add(panel);

        // Display the frame
        frame.pack(); // This will respect the preferred sizes of the panels and components
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
    private void nextFlashcard(){
        flashcardIndex++;
        flashcardText = flashcardKeys.get(flashcardIndex);
        flashcardLabel.setText(flashcardText);
        flashcardLabel.revalidate();
        flashcardLabel.repaint();
    }
    private void showTranslationText() {
        flashcardTranslationText = "Press show to see translation!";
        flashcardTranslationLabel.setText(flashcardTranslationText);
        flashcardTranslationLabel.revalidate();
        flashcardTranslationLabel.repaint();
    }
    private void finishOnboarding(String userName, Level currLevel) {
        User user = User.getInstance();
        user.initializeUser(userName, new ArrayList<String>(Arrays.asList("Completed User Onboarding!")), (int)score/10, score);
        user.earnAchievement();
        //user hasn't passed level
        if (currLevel != null && currLevel.passingScore != null && currLevelScore < currLevel.passingScore) {
            user.points = startingScore;
            user.xpLevel = (int)user.points/10;
            JOptionPane.showMessageDialog(null, "Your score was lower than the passing score of this Level!");
        }
        //user onboarding
        else if (currLevel == null) {
            JOptionPane.showMessageDialog(null, "You finished the user onboarding! Your score is: "+score+", your xpLevel: "+ user.xpLevel);
            frame.setVisible(false);
            new MenuScreen(userName);
            return;
        }
        //user passed
        else {
            JOptionPane.showMessageDialog(null, "You passed this level!");
        }
        frame.setVisible(false);
        new LevelsMenuScreen();
    }
}