package eu.fbk.ict.fm.bk2vec;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.cli.*;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.fbk.cit.hlt.thewikimachine.ExtractorParameters;
import org.fbk.cit.hlt.thewikimachine.xmldump.AbstractWikipediaExtractor;
import org.fbk.cit.hlt.thewikimachine.xmldump.AbstractWikipediaXmlDumpParser;
import org.fbk.cit.hlt.thewikimachine.xmldump.WikipediaTextExtractor;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.*;

/**
 * Extracts wikipedia text from the predefined list of articles
 *
 * @author Yaroslav Nechaev (remper@me.com)
 */
public class ExtractWikipediaText extends WikipediaTextExtractor {
    private static Logger logger = Logger.getLogger(ExtractWikipediaText.class);
    private List<String> pageList = new LinkedList<>();

    public ExtractWikipediaText(int numThreads, int numPages, Locale locale) {
        super(numThreads, numPages, locale);
    }

    public List<String> getPageList() {
        return pageList;
    }

    @Override
    public void contentPage(String text, String title, int wikiID) {
        if (pageList.size() > 0 && !pageList.contains(title)) {
            return;
        }
        super.contentPage(text, title, wikiID);
    }

    public static void main(String[] args) {
        String logConfig = System.getProperty("log-config");
        if (logConfig == null) {
            logConfig = "log-config.properties";
        }

        PropertyConfigurator.configure(logConfig);

        Options options = new Options();
        try {
            options.addOption(Option.builder("d").desc("Wikipedia xml dump file").required().hasArg().argName("file").longOpt("dump").build());
            options.addOption(Option.builder("l").desc("List of pages to extract").hasArg().argName("list").longOpt("list").build());
            options.addOption(Option.builder("o").desc("Output directory in which to store output files").required().hasArg().argName("file").longOpt("output").build());
            options.addOption(Option.builder("t").desc("Number of threads (default " + AbstractWikipediaXmlDumpParser.DEFAULT_THREADS_NUMBER + ")").hasArg().argName("int").longOpt("threads").build());
            options.addOption(Option.builder("p").desc("Number of pages to process (default all)").hasArg().argName("int").longOpt("pages").build());
            options.addOption(Option.builder("n").desc("Receive notification every n pages (default " + AbstractWikipediaExtractor.DEFAULT_NOTIFICATION_POINT + ")").hasArg().argName("int").longOpt("notification").build());

            options.addOption("h", "help", false, "print this message");
            options.addOption("v", "version", false, "output version information and exit");

            CommandLine line = new DefaultParser().parse(options, args);
            logger.debug(line);

            if (line.hasOption("help") || line.hasOption("version")) {
                throw new ParseException("");
            }

            int numThreads = AbstractWikipediaXmlDumpParser.DEFAULT_THREADS_NUMBER;
            if (line.hasOption("threads")) {
                numThreads = Integer.parseInt(line.getOptionValue("threads"));
            }

            int numPages = AbstractWikipediaExtractor.DEFAULT_NUM_PAGES;
            if (line.hasOption("pages")) {
                numPages = Integer.parseInt(line.getOptionValue("pages"));
            }

            int notificationPoint = AbstractWikipediaExtractor.DEFAULT_NOTIFICATION_POINT;
            if (line.hasOption("notification")) {
                notificationPoint = Integer.parseInt(line.getOptionValue("notification"));
            }

            ExtractorParameters extractorParameters = new ExtractorParameters(line.getOptionValue("dump"), line.getOptionValue("output"));
            ExtractWikipediaText wikipediaPageParser = new ExtractWikipediaText(numThreads, numPages, extractorParameters.getLocale());
            wikipediaPageParser.setNotificationPoint(notificationPoint);

            if (line.hasOption("list")) {
                List<String> pages = wikipediaPageParser.getPageList();
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
                            Object city = rawCity.get("city");
                            if (city == null || !(city instanceof Map)) {
                                continue;
                            }
                            Object name = ((Map<String, Object>) city).get("value");
                            if (name == null || !(name instanceof String)) {
                                continue;
                            }
                            pages.add(((String) name).substring(((String) name).lastIndexOf('/')+1));
                        }
                    } catch (IOException e) {
                        logger.error("Unable to load page list", e);
                    }
                }
            }

            wikipediaPageParser.start(extractorParameters);

            logger.info("extraction ended " + new Date());

        } catch (ParseException e) {
            // oops, something went wrong
            System.out.println("Parsing failed: " + e.getMessage() + "\n");
            HelpFormatter formatter = new HelpFormatter();
            formatter.printHelp(400, "java -cp dist/thewikimachine.jar org.fbk.cit.hlt.thewikimachine.xmldump.WikipediaTextExtractor", "\n", options, "\n", true);
        }
    }
}
