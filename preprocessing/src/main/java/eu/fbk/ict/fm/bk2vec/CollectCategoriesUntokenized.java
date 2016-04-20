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

public class CollectCategoriesUntokenized {

    private static Logger logger = Logger.getLogger(CollectCategoriesUntokenized.class);
    private static Integer MIN_CAT_TOKEN_LEN = 3;
    private static boolean KEEP_ALL_NUMBERS = true;
    private static Pattern numericPattern = Pattern.compile("^[0-9]+$");

    public static void main(String[] args) {
        try {
            final CommandLine cmd = CommandLine
                    .parser()
                    .withName("collect-category-tokens")
                    .withHeader("Collect Wikipedia category tokens")
                    .withOption("i", "input", "category file", "FILE",
                            CommandLine.Type.FILE_EXISTING, true, false, true)
                    .withOption("o", "output", "category output file", "FILE",
                            CommandLine.Type.FILE, true, false, true)
                    .withOption("f", "output-frequencies", "output frequencies file", "FILE",
                            CommandLine.Type.FILE, true, false, true)
                    .withLogger(LoggerFactory.getLogger("eu.fbk.fssa")).parse(args);

            final File inputPath = cmd.getOptionValue("i", File.class);
            final File outputPath = cmd.getOptionValue("o", File.class);
            final File freqPath = cmd.getOptionValue("f", File.class);

            BufferedReader in;
            String line;
            InputStream stream;
            HardTokenizer tokenizer = HardTokenizer.getInstance();

            HashMultimap<String, String> categories = HashMultimap.create();
            FrequencyHashSet<String> catFreq = new FrequencyHashSet<>();

            try {

                stream = IO.read(inputPath.getAbsolutePath());
                in = new BufferedReader(new InputStreamReader(stream));
                while ((line = in.readLine()) != null) {
                    if (line.trim().length() == 0) {
                        continue;
                    }

                    String[] parts = line.split("\t");
                    if (parts.length < 2) {
                        continue;
                    }

                    String page = parts[0].trim();
                    String category = parts[1].trim();

                    if (category.length() >= MIN_CAT_TOKEN_LEN) {
                        categories.put(page, category);
                    }
                    if (KEEP_ALL_NUMBERS) {
                        Matcher matcher = numericPattern.matcher(category);
                        if (matcher.find()) {
                            categories.put(page, category);
                        }
                    }
                }

                OutputStream outputStream;
                BufferedWriter writer;

                outputStream = IO.write(outputPath.getAbsolutePath());
                writer = new BufferedWriter(new OutputStreamWriter(outputStream));

                for (String page : categories.keySet()) {
                    writer.append(page);
                    for (String cat : categories.get(page)) {
                        writer.append("\t").append(cat);
                        catFreq.add(cat);
                    }

                    writer.append("\n");
                }

                outputStream.close();

                outputStream = IO.write(freqPath.getAbsolutePath());
                writer = new BufferedWriter(new OutputStreamWriter(outputStream));
                for (String cat : catFreq.keySet()) {
                    writer.append(cat).append("\t").append(catFreq.get(cat).toString()).append("\n");
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
