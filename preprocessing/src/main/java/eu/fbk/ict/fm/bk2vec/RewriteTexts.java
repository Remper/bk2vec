package eu.fbk.ict.fm.bk2vec;

import com.google.common.collect.HashMultimap;
import eu.fbk.dkm.utils.CommandLine;
import eu.fbk.rdfpro.util.IO;
import org.apache.log4j.Logger;
import org.fbk.cit.hlt.thewikimachine.analysis.HardTokenizer;
import org.fbk.cit.hlt.thewikimachine.util.FrequencyHashSet;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by alessio on 10/04/16.
 */

public class RewriteTexts {

    private static Logger logger = Logger.getLogger(RewriteTexts.class);
    private static Integer STOPWORDS_NUM = 30;
    private static Integer MIN_CAT_TOKEN_LEN = 3;
    private static boolean KEEP_ALL_NUMBERS = true;
    private static Pattern numericPattern = Pattern.compile("^[0-9]+$");

    public static void main(String[] args) {
        try {
            final CommandLine cmd = CommandLine
                    .parser()
                    .withName("rewrite-texts")
                    .withHeader("Rewrite TWM text index")
                    .withOption("i", "input", "category file", "FILE",
                            CommandLine.Type.FILE_EXISTING, true, false, true)
                    .withOption("o", "output", "category output file", "FILE",
                            CommandLine.Type.FILE, true, false, true)
                    .withLogger(LoggerFactory.getLogger("eu.fbk.fssa")).parse(args);

            final File inputPath = cmd.getOptionValue("i", File.class);
            final File outputPath = cmd.getOptionValue("o", File.class);

            BufferedReader in;
            String line;
            InputStream stream;

            OutputStream outputStream;
            BufferedWriter writer;

            outputStream = IO.write(outputPath.getAbsolutePath());
            writer = new BufferedWriter(new OutputStreamWriter(outputStream));

            HardTokenizer tokenizer = HardTokenizer.getInstance();

            try {
                stream = IO.read(inputPath.getAbsolutePath());
                in = new BufferedReader(new InputStreamReader(stream));
                while ((line = in.readLine()) != null) {
                    if (line.trim().length() == 0) {
                        continue;
                    }

                    String[] parts = line.split("\\s+");
                    if (parts.length < 2) {
                        continue;
                    }

                    String page = parts[0].trim();
                    String tokenizedPage = tokenizer.tokenizedString(page.replaceAll("_", " "));

                    try {
                        String text = line.substring(page.length() + 1 + tokenizedPage.length() + 1).trim();
                        writer.append(page).append("\t").append(text).append("\n");
                    } catch (Exception e) {
                        // ignored
                    }
                }

                outputStream.close();

            } catch (Exception e) {
                e.printStackTrace();
            }

        } catch (Exception e) {
            CommandLine.fail(e);
        }
    }
}
