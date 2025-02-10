package nl.vu.cs.softwaredesign.UserInterface;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;


public class MenuScreen {
    private JFrame frame = new JFrame(); //the window

    public MenuScreen(String nameUser) {
        //configure frame
        frame.setTitle("MenuScreen");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setSize(400,400);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        frame.requestFocusInWindow(); //focus on nothing initially

        //title panel
        JPanel panel = new JPanel(null);
        panel.setPreferredSize(new Dimension(400, 400));

        JPanel titlePanel = new JPanel(null);
        titlePanel.setBounds(0, 0, 400, 100);

        JLabel titleText = new JLabel("Menu");
        titleText.setBounds(100, 0, 200, 100);
        titleText.setVerticalAlignment(JLabel.CENTER);
        titleText.setHorizontalAlignment(JLabel.CENTER);
        Font labelFont = titleText.getFont();
        titleText.setFont(new Font(labelFont.getName(), Font.PLAIN, 40));
        titlePanel.add(titleText);

        //buttons panel
        JPanel buttonsPanel = new JPanel(null);
        buttonsPanel.setBounds(0, 100, 400, 300);

        JButton profileButton = new JButton("Profile");
        profileButton.setBounds(100 , 0, 200, 50);
        profileButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                System.out.println(nameUser);
                frame.setVisible(false);
                new ProfileScreen(nameUser);
            }
        });

        JButton LevelsButton = new JButton("Levels");
        LevelsButton.setBounds(100 , 100, 200, 50);
        LevelsButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.setVisible(false);
                new LevelsMenuScreen();
            }
        });

        JButton QuitButton = new JButton("Quit");
        QuitButton.setBounds(100 , 200, 200, 50);
        QuitButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.setVisible(false);
                System.exit(0);
            }
        });

        //add buttons to the button panel
        buttonsPanel.add(profileButton);
        buttonsPanel.add(LevelsButton);
        buttonsPanel.add(QuitButton);

        //add panels to the frame
        panel.add(titlePanel);
        panel.add(buttonsPanel);

        frame.add(panel);

        // Display the frame
        frame.pack(); // This will respect the preferred sizes of the panels and components
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
