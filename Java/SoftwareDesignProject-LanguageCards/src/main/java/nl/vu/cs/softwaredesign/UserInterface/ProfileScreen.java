package nl.vu.cs.softwaredesign.UserInterface;

import nl.vu.cs.softwaredesign.UserManagement.User;

import javax.swing.*;
import javax.swing.border.Border;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class ProfileScreen {
    private JFrame frame = new JFrame();
    public ProfileScreen(String userName) {
        //configure frame
        frame.setTitle("Profile");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setSize(400,400);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        frame.requestFocusInWindow(); //focus on nothing initially

        JPanel panel = new JPanel(null);
        panel.setPreferredSize(new Dimension(400, 400));

        JPanel titlePanel = new JPanel(null);
        titlePanel.setBounds(0, 0, 400, 100);

        JLabel titleText = new JLabel("Profile");
        titleText.setBounds(100, 0, 200, 100);
        titleText.setVerticalAlignment(JLabel.CENTER);
        titleText.setHorizontalAlignment(JLabel.CENTER);
        Font labelFont = titleText.getFont();
        titleText.setFont(new Font(labelFont.getName(), Font.PLAIN, 40));
        JButton returnButton = new JButton("\u2190");
        returnButton.setBounds(0 , 0, 50, 50);
        returnButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.setVisible(false);
                new MenuScreen(userName);
            }
        });
        titlePanel.add(returnButton);
        titlePanel.add(titleText);

        //stats panel
        JPanel statsPanel = new JPanel(null);
        statsPanel.setBounds(50, 100, 300, 175);
        Border border = BorderFactory.createLineBorder(Color.BLACK);
        statsPanel.setBorder(border);

        JLabel userNameText = new JLabel("Username: "+ User.getInstance().name);
        userNameText.setBounds(5, 25, 250, 25);
        userNameText.setVerticalAlignment(JLabel.CENTER);
        userNameText.setHorizontalAlignment(JLabel.LEFT);
        labelFont = userNameText.getFont();
        userNameText.setFont(new Font(labelFont.getName(), Font.PLAIN, 20));
        statsPanel.add(userNameText);

        JLabel xpText = new JLabel("xpLevel: "+User.getInstance().xpLevel);
        xpText.setBounds(5, 75, 195, 25);
        xpText.setVerticalAlignment(JLabel.CENTER);
        xpText.setHorizontalAlignment(JLabel.LEFT);
        labelFont = xpText.getFont();
        xpText.setFont(new Font(labelFont.getName(), Font.PLAIN, 20));
        statsPanel.add(xpText);

        JLabel pointsText = new JLabel("Points: "+User.getInstance().points);
        pointsText.setBounds(5, 125, 195, 25);
        pointsText.setVerticalAlignment(JLabel.CENTER);
        pointsText.setHorizontalAlignment(JLabel.LEFT);
        labelFont = pointsText.getFont();
        pointsText.setFont(new Font(labelFont.getName(), Font.PLAIN, 20));
        statsPanel.add(pointsText);

        //button panel
        JPanel buttonsPanel = new JPanel(null);
        buttonsPanel.setBounds(0, 275, 400, 125);

        JButton achievementsButton = new JButton("Achievements");
        achievementsButton.setBounds(100 , 25, 200, 50);
        achievementsButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.setVisible(false);
                //lazily check achievements
                User.getInstance().earnAchievement();
                AchievementsScreen.getInstance().showScreen();
            }
        });
        buttonsPanel.add(achievementsButton);

        //add panels to the frame
        panel.add(titlePanel);
        panel.add(statsPanel);
        panel.add(buttonsPanel);

        frame.add(panel);
        // Display the frame
        frame.pack(); // This will respect the preferred sizes of the panels and components
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
