package app;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import rest.RegexService;
import spark.Spark;

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

            Spark.port(Integer.parseInt(properties.getProperty("port")));
            Gson gson = new GsonBuilder().setPrettyPrinting().create();

            boolean totalize = Boolean.parseBoolean(properties.getProperty("total"));
            if (args.length == 0) {
                Spark.get("/regex2dfa2dot/:letter/:regex/",
                        (request, response) -> new RegexService(request.params(":letter"), request.params(":regex"), totalize),
                        gson::toJson);
            }
            else {
                String output = gson.toJson(new RegexService(args[0], args[1], totalize));
                try {
                    FileWriter writer = new FileWriter(args[2]);
                    writer.write(output);
                    writer.close();

                } catch (IOException e) {
                    e.printStackTrace();
                }
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