package nl.vu.cs.softwaredesign.UserInterface;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;


public class StartScreen {
    private JFrame frame = new JFrame(); //the window
    public StartScreen() {
        try {
            // Set to Nimbus Look and Feel
            for (UIManager.LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (Exception e) {
            // If Nimbus is not available, fall back to the default look and feel.
            try {
                UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName());
            } catch (Exception ex) {
                // Handle exception
            }
        }
        //title panel
        JPanel panel = new JPanel(null);
        panel.setPreferredSize(new Dimension(400, 400));

        JPanel titlePanel = new JPanel(null);
        titlePanel.setBounds(0, 0, 400, 100);

        //background icon
        ImageIcon backgroundIcon = new ImageIcon(StartScreen.class.getResource("/images/appIcon.jpeg"));
        JLabel backgroundLabel = new JLabel(backgroundIcon);
        backgroundLabel.setBounds(0, 0, 400, 400);
        backgroundLabel.setLayout(new BorderLayout());

        JLabel titleText = new JLabel("<html><center>" + "FlashLingua: English & Spanish" + "</center></html>", SwingConstants.CENTER);

        //create app title
        Font labelFont = titleText.getFont();
        titleText.setFont(new Font(labelFont.getName(), Font.BOLD, 20));
        titleText.setBorder(BorderFactory.createEmptyBorder(10, 0, 0, 0));
        backgroundLabel.add(titleText, BorderLayout.NORTH);
        titleText.setOpaque(true);
        titleText.setBackground(new Color(0, 0, 0, 123));
        titleText.setForeground(Color.WHITE);

        JTextField nameField = new JTextField("Enter name to create new profile");
        nameField.setBounds(50, 250, 300, 30);
        nameField.setForeground(Color.BLACK);
        nameField.addFocusListener(new FocusListener() {
            @Override
            public void focusGained(FocusEvent e) {  //remove text when user tries to type in the field
                if (nameField.getText().equals("Enter name to create new profile") || nameField.getText().equals("Enter a valid name under 10 characters!")) {
                    nameField.setText("");
                }
            }

            @Override
            public void focusLost(FocusEvent e) { //put default text back
                if (nameField.getText().trim().isEmpty()) {
                    nameField.setText("Enter name to create new profile");
                }
            }
        });
        JButton startButton = new JButton("Start user onboarding!");
        startButton.setBounds(50, 300, 300, 50);

        startButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String userName = nameField.getText();
                if (!userName.equals("") && (userName.length() < 10)) {
                    frame.setVisible(false);
                    new LevelScreen(true, null, nameField.getText());
                }
                else {
                    nameField.setText("Enter a valid name under 10 characters!");
                }
            }
        });

        panel.add(nameField, BorderLayout.NORTH);
        panel.add(startButton, BorderLayout.SOUTH);
        panel.add(backgroundLabel);


        // set up the frame and display it
        frame.add(panel, BorderLayout.CENTER);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setTitle("GUI");
        frame.pack();
        frame.setSize(400,400);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        frame.requestFocusInWindow(); //focus on nothing initially
        frame.setResizable(false);
    }
    // create one Frame
    public static void main(String[] args) {
        new StartScreen();
    }
}
