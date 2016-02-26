# bk2vec
Injecting background knowledge into the word vectors


## How to train word vectors
- Place your texts under the `corpora` folder. Script assumes one text per line, splits sentences on dot.
- Launch TextReader.py. It will train word vectors and save them into `model.w2v` file
- Fill free to implement a proper script that will actually take parameters and won't use the hardcoded ones

## How to extract texts from the wikipedia dump

```
java -cp thewikimachine.jar org.fbk.cit.hlt.thewikimachine.xmldump.WikipediaTextExtractor 
-d <path-to-dump.xml>
-o <path-to-output-directory>
-t <amount of threads>
```
