package nl.vu.cs.softwaredesign.UserInterface;
import nl.vu.cs.softwaredesign.Observer;
import nl.vu.cs.softwaredesign.UserManagement.User;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

//singleton
public class AchievementsScreen implements Observer {
    private JFrame frame = new JFrame();
    private Font labelFont;
    private JPanel panel = new JPanel(null);
    private JPanel achievementsPanel = new JPanel(null);
    private static AchievementsScreen instance;
    public static AchievementsScreen getInstance() {
        if (instance == null) {
            instance = new AchievementsScreen();
        }
        return instance;
    }
    public void showScreen() {
        refreshAchievementScreen(); // Ensure the content is up to date
        frame.pack();               // Re-layout the components
        frame.setVisible(true);     // Make the frame visible again
    }
    private AchievementsScreen() {
        frame.setTitle("AchievementsScreen");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setSize(400,400);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        frame.requestFocusInWindow(); //focus on nothing initially
        panel.setPreferredSize(new Dimension(400, 400));

        JPanel titlePanel = new JPanel(null);
        titlePanel.setBounds(0, 0, 400, 100);

        JLabel titleText = new JLabel("Achievements");
        titleText.setBounds(100, 0, 200, 100);
        titleText.setVerticalAlignment(JLabel.CENTER);
        titleText.setHorizontalAlignment(JLabel.CENTER);
        labelFont = titleText.getFont();
        titleText.setFont(new Font(labelFont.getName(), Font.PLAIN, 30));
        JButton returnButton = new JButton("\u2190");
        returnButton.setBounds(0 , 0, 50, 50);
        returnButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.setVisible(false);
                new ProfileScreen(User.getInstance().name);
            }
        });
        titlePanel.add(returnButton);
        titlePanel.add(titleText);
        panel.add(titlePanel);
        frame.add(panel);
        frame.pack(); // This will respect the preferred sizes of the panels and components
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
    private void refreshAchievementScreen() {
        achievementsPanel.removeAll();
        achievementsPanel.revalidate();
        achievementsPanel.repaint();
        achievementsPanel.setBounds(0, 100, 400, 300);
        //create a for loop, for every achievment create an icon with text from User.achievements array
        ImageIcon trophyIcon = null;
        try {
            trophyIcon = new ImageIcon(AchievementsScreen.class.getResource("/images/trophyIcon.jpeg"));
        } catch (Exception e) {
            System.out.println("Couldn't load image from resources!");
        }
        int xAxis = 25;
        int yAxis = 0;
        int iconWidth = 75;
        int iconHeight = 75;
        int maxIconsPerRow = 3;
        int gap = (achievementsPanel.getWidth() - (maxIconsPerRow * iconWidth)) / (maxIconsPerRow + 1);
        for (int i = 0; i < User.getInstance().achievements.size(); i++) {
            System.out.println("Showing icon!");
            JLabel achievementLabel = new JLabel();
            String achievement = User.getInstance().achievements.get(i);
            JLabel achievementText = new JLabel("<html><center>" + achievement + "</center></html>");
            achievementLabel.setBounds(xAxis, yAxis, iconWidth, iconHeight);
            achievementText.setBounds(xAxis, yAxis+75, iconWidth, iconHeight);
            achievementText.setVerticalAlignment(JLabel.NORTH);
            achievementText.setHorizontalAlignment(JLabel.CENTER);
            labelFont = achievementText.getFont();
            achievementText.setFont(new Font(labelFont.getName(), Font.PLAIN, 10));
            achievementsPanel.add(achievementLabel);
            achievementsPanel.add(achievementText);
            achievementLabel.setIcon(trophyIcon);

            xAxis += iconWidth + gap;
            if ((i + 1) % maxIconsPerRow == 0) {
                yAxis = yAxis + 150;
                xAxis = 25;
            }
        }
        panel.add(achievementsPanel);
        if (!panel.isAncestorOf(achievementsPanel)) {
            panel.add(achievementsPanel);
        }
        achievementsPanel.revalidate();
        achievementsPanel.repaint();
    }
    @Override
    public void update (Object arg){
        refreshAchievementScreen();
    }
}
