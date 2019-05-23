package util;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;

public class FileUtil {

    public static String readAll(String fileName) throws IOException {
        List<String> lines = Files.readAllLines(new File(fileName).toPath());
        String content = String.join("", lines.toArray(new String[lines.size()]));

        return content;
    }

}
