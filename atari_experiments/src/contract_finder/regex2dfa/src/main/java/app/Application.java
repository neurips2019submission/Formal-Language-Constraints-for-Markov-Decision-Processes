package app;

import service.RegexService;

import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Created by hd on 04/14/18.
 */
public class Application {

    public static void main(String[] args) {

        Properties properties = new Properties();
        InputStream input = null;

        try {
            input = new FileInputStream("config.properties");
            properties.load(input);

            String letter = properties.getProperty("letter");
            boolean totalize = Boolean.parseBoolean(properties.getProperty("total"));
            if (args.length == 0) {
                System.err.println("Usage: java -jar regex2dfa.jar input_regex.file output_dfa.dot");
                System.exit(1);
            }
            String output = new RegexService(totalize, args[0]).toDot(letter);
            try {
                FileWriter writer = new FileWriter(args[1]);
                writer.write(output);
                writer.close();

            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}