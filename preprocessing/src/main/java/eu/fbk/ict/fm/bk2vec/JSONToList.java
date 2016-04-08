package eu.fbk.ict.fm.bk2vec;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.tools.javac.util.Pair;
import org.apache.commons.cli.*;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import java.io.*;
import java.util.*;

/**
 * Converts relation jsons to a single csv list
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class JSONToList {
    private static Logger logger = Logger.getLogger(JSONToList.class);
    public final static double SPLIT = 0.1;

    private Writer listHandler;
    private Writer listHandlerTest;

    public JSONToList(String outputFile, String testFile, Set<Pair<String, String>> list) throws IOException {
        listHandler = openHandler(outputFile);
        listHandlerTest = null;
        if (testFile != null) {
            listHandlerTest = openHandler(testFile);
        }
        writeList(list);
        listHandler.close();
        listHandlerTest.close();
    }

    private Writer openHandler(String file) throws IOException {
        return new BufferedWriter(new FileWriter(new File(file)));
    }

    private void writeList(Set<Pair<String, String>> list) throws IOException {
        Random rnd = new Random();
        for (Pair<String, String> element : list) {
            Writer currentWriter = listHandler;
            double value = rnd.nextDouble();
            if (listHandlerTest != null && value <= SPLIT) {
                currentWriter = listHandlerTest;
            }
            currentWriter.write(element.fst);
            currentWriter.write('\t');
            currentWriter.write(element.snd);
            currentWriter.write('\n');
        }
    }

    public static void main(String[] args) {
        String logConfig = System.getProperty("log-config");
        if (logConfig == null) {
            logConfig = "log-config.properties";
        }

        PropertyConfigurator.configure(logConfig);

        Options options = new Options();
        try {
            options.addOption(Option.builder("l").desc("Location of jsons").required().hasArg().argName("list").longOpt("list").build());
            options.addOption(Option.builder("o").desc("Output file").required().hasArg().argName("file").longOpt("output").build());
            options.addOption(Option.builder("t").desc("Output test file").hasArg().argName("file").longOpt("output-test").build());
            options.addOption(Option.builder().desc("Only unigrams").longOpt("unigrams").build());

            options.addOption("h", "help", false, "print this message");
            options.addOption("v", "version", false, "output version information and exit");

            CommandLine line = new DefaultParser().parse(options, args);
            logger.debug(line);

            if (line.hasOption("help") || line.hasOption("version")) {
                throw new ParseException("");
            }

            ObjectMapper mapper = new ObjectMapper();
            File list = new File(line.getOptionValue("list"));
            List<File> files = new LinkedList<>();
            if (list.isDirectory()) {
                for (File file : list.listFiles()) {
                    if (file.getName().endsWith(".json")) {
                        files.add(file);
                    }
                }
            } else {
                files.add(list);
            }
            Set<Pair<String, String>> result = new HashSet<>();
            for (File file : files) {
                try {
                    logger.info("Parsing file "+file.getName());
                    JsonParser parser = new JsonFactory().createParser(file);

                    parser.nextToken();
                    while (parser.nextToken() != JsonToken.END_OBJECT) {
                        String fieldname = parser.getCurrentName();
                        parser.nextToken();
                        if ("results".equals(fieldname)) {
                            break;
                        }
                        mapper.readValue(parser, Object.class);
                    }
                    while (parser.nextToken() != JsonToken.END_OBJECT) {
                        String fieldname = parser.getCurrentName();
                        parser.nextToken();
                        if ("bindings".equals(fieldname)) {
                            break;
                        }
                        mapper.readValue(parser, Object.class);
                    }
                    parser.nextToken();
                    while (parser.nextToken() != JsonToken.END_ARRAY) {
                        Map<String, Object> rawCity = mapper.readValue(parser, new TypeReference<Map<String, Object>>() {});
                        if (rawCity == null) {
                            continue;
                        }
                        String city = getValue(rawCity.get("city"));
                        if (city == null) {
                            continue;
                        }
                        String country = getValue(rawCity.get("country"));
                        String name = city.substring(city.lastIndexOf('/')+1);
                        int comma = name.indexOf(',');
                        if (comma != -1) {
                            name = name.substring(0, comma);
                        }
                        if (name.contains("_")) {
                            continue;
                        }
                        if (country == null) {
                            country = file.getName().substring(0, file.getName().indexOf('.'));
                        } else {
                            country = country.substring(country.lastIndexOf('/')+1);
                        }
                        result.add(new Pair<>(name, country));
                    }
                } catch (IOException e) {
                    logger.error("Unable to load page list", e);
                }
            }
            try {
                new JSONToList(line.getOptionValue("output"), line.getOptionValue("output-test"), result);
            } catch (IOException e) {
                logger.error("Error while saving to disk", e);
            }

            logger.info("extraction ended " + new Date());

        } catch (ParseException e) {
            // oops, something went wrong
            System.out.println("Parsing failed: " + e.getMessage() + "\n");
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(400, "java -cp dist/thewikimachine.jar org.fbk.cit.hlt.thewikimachine.xmldump.WikipediaTextExtractor", "\n", options, "\n", true);
        }
    }

    private static String getValue(Object rawMap) {
        if (rawMap == null || !(rawMap instanceof Map)) {
            return null;
        }
        Object rawValue = ((Map<String, Object>) rawMap).get("value");
        if (rawValue == null || !(rawValue instanceof String)) {
            return null;
        }
        return (String) rawValue;
    }
}
