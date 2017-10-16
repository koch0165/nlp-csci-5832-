from collections import Counter, defaultdict
import numpy as np
import math
import sys
from collections import namedtuple
import random

class TrainingSet:

    def __init__(self,sentences):

        self.sentences = sentences
        self.tags = {}
        self.wordset = {}
        self.tagCount = defaultdict(dict)
        self.wordTagCount = defaultdict(dict)
        self.tagToTagCount = defaultdict(dict)
        self.minWordCountForTag = defaultdict(dict)

    def getTagsWordsCounts(self):
        totalBigrams = 0
        tagsPreceding = defaultdict(int)
        self.tags['START'] = 1
        self.tagCount['START'] = len(sentences)
        for sentence in sentences:
            prevTag = 'START'
            for line in sentence.lines:
                words = line.split('\t')

                if prevTag in self.tagToTagCount and words[2] in self.tagToTagCount[prevTag]:
                    self.tagToTagCount[prevTag][words[2]] = self.tagToTagCount[prevTag][words[2]] + 1
                else:
                    self.tagToTagCount[prevTag][words[2]] = 1
                    totalBigrams = totalBigrams + 1
                    if words[2] in tagsPreceding:
                        tagsPreceding[words[2]] = tagsPreceding[words[2]] + 1
                    else:
                        tagsPreceding[words[2]] = 1
                prevTag = words[2]

                if words[1] in self.wordTagCount and words[2] in self.wordTagCount[words[1]]:
                    self.wordTagCount[words[1]][words[2]] = self.wordTagCount[words[1]][words[2]] + 1
                else:
                    self.wordTagCount[words[1]][words[2]] = 1

                if (words[2] in self.tagCount):
                    self.tagCount[words[2]] = self.tagCount[words[2]] + 1
                else:
                    self.tagCount[words[2]] = 1

                if (not (words[2] in self.tags)):
                    self.tags[words[2]] = 1
                if (not (words[1] in self.wordset)):
                    self.wordset[words[1]] = 1

        return self.wordset,self.tags,self.tagToTagCount,tagsPreceding,totalBigrams

    def getTransitionMatrix(self,tagsPreceding,totalBigrams):
        transition_matrix = defaultdict(dict)
        for tag1 in self.tags:
            for tag2 in self.tags:
                if ((tag1 in self.tagToTagCount) and (tag2 in self.tagToTagCount[tag1])):
                    length = len(self.tagToTagCount[tag1])
                    transition_matrix[tag1][tag2] = ((self.tagToTagCount[tag1][tag2] - 0.75 if (self.tagToTagCount[tag1][tag2] - 0.75 >0) else 0) / self.tagCount[tag1]) + ((0.75/self.tagCount[tag1])*(length)*(tagsPreceding[tag2]/totalBigrams))
                else:
                    length = len(self.tagToTagCount[tag1])
                    val = int(tagsPreceding[tag2])
                    transition_matrix[tag1][tag2] = (0.75/self.tagCount[tag1])*(length)* (val/totalBigrams)

        return transition_matrix

    def getObservationMatrix(self):
        observation_matrix = defaultdict(dict)
        minWordCountForTag = defaultdict(dict)
        maxTagForWord = defaultdict(dict)
        maxCountForWord = defaultdict(int)
        for word in wordset:
            maxTagForWord[word] = ''
            maxCountForWord[word] = 0
        for word in self.wordset:
            for tag1 in self.tags:
                if ((word in self.wordTagCount) and (tag1 in self.wordTagCount[word])):
                    if(self.wordTagCount[word][tag1]>maxCountForWord[word]):
                        maxCountForWord[word]=self.wordTagCount[word][tag1]
                        maxTagForWord[word]=tag1
                    observation_matrix[word][tag1] = self.wordTagCount[word][tag1] / self.tagCount[tag1]
                    if (tag1 in minWordCountForTag):
                        if (minWordCountForTag[tag1] > observation_matrix[word][tag1]):
                            minWordCountForTag[tag1] = observation_matrix[word][tag1]
                    else:
                        minWordCountForTag[tag1] = observation_matrix[word][tag1]
                else:
                    observation_matrix[word][tag1] = 0

        return observation_matrix,minWordCountForTag,maxTagForWord



class ValidationTest:

    def __init__(self):

        print("")
        # self.wordset = wordset
        # self.observation_matrix = observation_matrix
        # self.transition_matrix = transition_matrix
        # self.tags = tags
        # self.minWordCountForTag= minWordCountForTag

    def getBaselineAccuracy(self, sentences,wordTagCount,wordset,maxTagForWord):

        wordcount = 0
        accCount = 0
        for sentence in sentences:
            for line in sentence.lines:
                words = line.split('\t')
                count = words.count('.')
                if(count == 2):
                    continue
                wordcount = wordcount + 1
                if(words[1] in wordset):
                    tag = maxTagForWord[words[1]]
                    if tag == words[2]:
                        accCount = accCount + 1

        return accCount/wordcount




    def viterbi(self,sentences,wordset,tags,observation_matrix,transition_matrix,minWordCountForTag):

        viterbi = defaultdict(dict)
        backtrack = defaultdict(dict)
        counter = 0
        wordcount = 0
        accCount = 0
        sentenceno = 0
#        print("Sentences %d"%(len(sentences)))
        for sentence in sentences:
#            print('Sentence no %d'%(sentenceno))
            sentenceno = sentenceno +1
            firstWord = 1
            counter=0
            actualtags = []
     #       wordcount = 0
            for line in sentence.lines:
                words = line.split('\t')
                actualtags.append(words[1])
                if(firstWord == 1):
                    if words[1] in wordset:
                        for tag in tags:
                            if(transition_matrix['START'][tag] !=0 and observation_matrix[words[1]][tag]!=0):
                                viterbi[counter][tag] = -math.log(transition_matrix['START'][tag]) + (-math.log(observation_matrix[words[1]][tag]))
                            else:
                                viterbi[counter][tag] = 0
                    else:
                        for tag in tags:
                            if (transition_matrix['START'][tag] !=0 and  minWordCountForTag[tag] != 0):
                                val = -math.log(transition_matrix['START'][tag]) + (-math.log(minWordCountForTag[tag]))
                                viterbi[counter][tag] = val
                            else:
                                viterbi[counter][tag] = 0
                    firstWord = 0
                else:
                    if words[1] in wordset:
                        for tag in tags:
                            min = sys.maxsize
                            besttag = ''
                            for prevtag in tags:
                                if(viterbi[counter-1][prevtag]!=0 and transition_matrix[prevtag][tag] !=0 and observation_matrix[words[1]][tag]!=0):
                                    val = viterbi[counter-1][prevtag]-math.log(transition_matrix[prevtag][tag]) + (-math.log(observation_matrix[words[1]][tag]))
                                    if(min > val):
                                        min = val
                                        besttag = prevtag
                            if min == sys.maxsize:
                                viterbi[counter][tag] = 0
                            else:
                                viterbi[counter][tag] = min
                                backtrack[counter][tag] = besttag

                    else:
                        for tag in tags:
                            min = sys.maxsize
                            besttag = ''
                            for prevtag in tags:
                                if (viterbi[counter-1][prevtag]!=0 and transition_matrix[prevtag][tag] !=0 and minWordCountForTag[tag] != 0):
                                    val = viterbi[counter-1][prevtag] -math.log(transition_matrix[prevtag][tag]) + (-math.log(minWordCountForTag[tag]))
                                    if (min > val):
                                        min = val
                                        besttag = prevtag
                            if min == sys.maxsize:
                                viterbi[counter][tag] = 0
                            else:
                                viterbi[counter][tag] = min
                                backtrack[counter][tag] = besttag
                if(len(sentence.lines)==wordcount):
                    min = sys.maxsize
                    besttag = ''
                    for tag in tags:
                        if(counter-1 in viterbi and tag in viterbi[counter-1] and viterbi[counter-1][tag]!=0 ):
                            if( tag in transition_matrix[tag] and words[2] in transition_matrix[tag] and transition_matrix[tag][words[2]]!=0):
                                val = viterbi[counter-1][tag] -math.log(transition_matrix[tag][words[2]])
                                if(min > val):
                                    min = val
                                    besttag = tag
                    backtrack[counter][words[2]] = besttag
                counter=counter+1
            counter = counter-1
            tagsList = []
            prevtag = ''
            while counter > 0:
                if counter == len(sentence.lines)-1:
                    tag = backtrack[counter]['.']
                    prevtag = tag
                else:
                    tag = backtrack[counter][prevtag]
                    prevtag = tag
                tagsList.append(tag)
                counter = counter - 1
            counter = len(sentence.lines)-2
            #print("Sentence : %s"%(sentence.lines))
            if(len(sentence.lines) ==1):
                continue
            wordcount = wordcount + len(sentence.lines)-1
            for line in sentence.lines :
                words = line.split('\t')
                if tagsList[counter] == words[2]:
                    accCount = accCount + 1
                counter = counter-1
                if(counter<0):
                    break

        accuracy = accCount/wordcount
        print("Viterbi Accuracy : %f"%(accuracy))
        return accuracy

class TestSetViterbi:

    def __init__(self):

        print("")
        # self.wordset = wordset
        # self.observation_matrix = observation_matrix
        # self.transition_matrix = transition_matrix
        # self.tags = tags
        # self.minWordCountForTag= minWordCountForTag

    def viterbi(self,sentences,wordset,tags,observation_matrix,transition_matrix,minWordCountForTag):
        f = open("/Users/koushikreddy/Downloads/test-output.txt", "w")
        viterbi = defaultdict(dict)
        backtrack = defaultdict(dict)
        counter = 0
        wordcount = 0
        accCount = 0
        sentenceno = 0
        for sentence in sentences:
            sentenceno = sentenceno +1
            firstWord = 1
            counter=0
            actualtags = []
     #       wordcount = 0
            for line in sentence.lines:
                words = line.split('\t')
                actualtags.append(words[1])
                if(firstWord == 1):
                    if words[1] in wordset:
                        for tag in tags:
                            if(transition_matrix['START'][tag] !=0 and observation_matrix[words[1]][tag]!=0):
                                viterbi[counter][tag] = -math.log(transition_matrix['START'][tag]) + (-math.log(observation_matrix[words[1]][tag]))
                            else:
                                viterbi[counter][tag] = 0
                    else:
                        for tag in tags:
                            if (transition_matrix['START'][tag] !=0 and  minWordCountForTag[tag] != 0):
                                val = -math.log(transition_matrix['START'][tag]) + (-math.log(minWordCountForTag[tag]))
                                viterbi[counter][tag] = val
                            else:
                                viterbi[counter][tag] = 0
                    firstWord = 0
                else:
                    if words[1] in wordset:
                        for tag in tags:
                            min = sys.maxsize
                            besttag = ''
                            for prevtag in tags:
                                if(viterbi[counter-1][prevtag]!=0 and transition_matrix[prevtag][tag] !=0 and observation_matrix[words[1]][tag]!=0):
                                    val = viterbi[counter-1][prevtag]-math.log(transition_matrix[prevtag][tag]) + (-math.log(observation_matrix[words[1]][tag]))
                                    if(min > val):
                                        min = val
                                        besttag = prevtag
                            if min == sys.maxsize:
                                viterbi[counter][tag] = 0
                            else:
                                viterbi[counter][tag] = min
                                backtrack[counter][tag] = besttag

                    else:
                        for tag in tags:
                            min = sys.maxsize
                            besttag = ''
                            for prevtag in tags:
                                if (viterbi[counter-1][prevtag]!=0 and transition_matrix[prevtag][tag] !=0 and minWordCountForTag[tag] != 0):
                                    val = viterbi[counter-1][prevtag] -math.log(transition_matrix[prevtag][tag]) + (-math.log(minWordCountForTag[tag]))
                                    if (min > val):
                                        min = val
                                        besttag = prevtag
                            if min == sys.maxsize:
                                viterbi[counter][tag] = 0
                            else:
                                viterbi[counter][tag] = min
                                backtrack[counter][tag] = besttag
                if(len(sentence.lines)==wordcount):
                    min = sys.maxsize
                    besttag = ''
                    for tag in tags:
                        if(viterbi[counter-1][tag]!=0 and transition_matrix[tag][words[2]]!=0):
                            val = viterbi[counter-1][tag] -math.log(transition_matrix[tag][words[2]])
                            if(min > val):
                                min = val
                                besttag = tag
                    backtrack[counter][words[2]] = besttag
                counter=counter+1
            counter = counter-1
            tagsList = []
            prevtag = ''
            while counter > 0:
                if counter == len(sentence.lines)-1:
                    tag = backtrack[counter]['.']
                    prevtag = tag
                else:
                    tag = backtrack[counter][prevtag]
                    prevtag = tag
                tagsList.append(tag)
                counter = counter - 1
            counter = len(sentence.lines)-2
            if(len(sentence.lines) ==1):
                continue
            wordcount = wordcount + len(sentence.lines)-1
            lineno = 1
            for line in sentence.lines:
                tokens = line.split('\t')
                f.write(tokens[0] + '\t' + tokens[1] + '\t' + tagsList[counter])
                f.write('\n')
                lineno = lineno + 1
                counter = counter - 1
                if (counter < 0):
                    break;
            f.write('%d' % (lineno))
            f.write('\t.\t.\n')
            f.write('\n')
        f.close()


class Sentence:

    def __init__(self, lines):

        self.lines = lines

class Kfold:

    def __init__(self):
        #self.num_folds = num_folds
      #  self.length = length
        print("")

    def split_cv(self,length,num_folds):
        """
        This function splits index [0, length - 1) into num_folds (train, test) tuples.
        """
        SplitIndices = namedtuple("SplitIndices", ["train", "test"])
        splits = [SplitIndices([], []) for _ in range(num_folds)]
        indices = list(range(length))

        random.shuffle(indices)
        index = 0
        subarrays = np.array_split(indices, num_folds)

        for split in splits:
            split.test.extend(subarrays[index])
            i = 0
            while i < index:
                split.train.extend(subarrays[i])
                i = i + 1

            i = index + 1
            while i < num_folds:
                split.train.extend(subarrays[i])
                i = i + 1
            index = index + 1

        # Finish this function to populate `folds`.
        # All the indices are split into num_folds folds.
        # Each fold is the testing set in a split, and the remaining indices
        # are added to the corresponding training set.

        return splits


if __name__ == "__main__":

    f = open("/Users/koushikreddy/Downloads/berp-POS-training.txt", "r")

    tags = {}
    wordset = {}
    tagCount = defaultdict(dict)
    wordTagCount = defaultdict(dict)
    tagToTagCount = defaultdict(dict)
    lineCount = 0
    firstWord = 1

    sentences = []
    lines = []
    for line in f:

        if line == '\n':
            # firstWord = 1
            continue

        length = len(line)
        line = line[:length-1]
        words = line.split('\t')
        count = words.count('.')

        if count == 2:
            lines.append(line)
            sentence = Sentence(lines)
            sentences.append(sentence)
            lines = []
            continue

        lines.append(line)
    f.close()
    indices = list(range(len(sentences)))
    length = len(sentences)
    random.shuffle(indices)
    kfold = Kfold()
    splits = kfold.split_cv(length,10)
    accuracy_array = []
    highest_accuracy = 0
    best_transition_matrix = defaultdict(dict)
    best_observation_matrix = defaultdict(dict)
    best_minWordCountForTag = defaultdict(dict)
    best_wordset = {}
    best_tags = {}
    for split in splits:
        # Finish this function to use the training instances
        # indexed by `split.train` to train the classifier,
        # and then store the accuracy
        # on the testing instances indexed by `split.test`
        train = [sentences[index] for index in split.train]
        test = [sentences[index] for index in split.test]

        trainingSet = TrainingSet(train)

        (wordset, tags, tagToTagCount,tagsPreceding,totalBigrams) = trainingSet.getTagsWordsCounts()
        (observation_matrix, minWordCountForTag,maxTagForWord) = trainingSet.getObservationMatrix()
        (transition_matrix) = trainingSet.getTransitionMatrix(tagsPreceding,totalBigrams)

        validationTest = ValidationTest()
        baseline_accuracy = validationTest.getBaselineAccuracy(sentences,wordTagCount,wordset,maxTagForWord)
        print('baseline : %f'%(baseline_accuracy))
        accuracy = validationTest.viterbi(test, wordset, tags, observation_matrix, transition_matrix, minWordCountForTag)
        if(accuracy>highest_accuracy):
            highest_accuracy = accuracy
            best_tags = tags
            best_wordset = wordset
            best_observation_matrix = observation_matrix
            best_transition_matrix = transition_matrix
            best_minWordCountForTag = minWordCountForTag

        accuracy_array.append(accuracy)

    trainingSet = TrainingSet(sentences)
    (wordset, tags, tagToTagCount, tagsPreceding, totalBigrams) = trainingSet.getTagsWordsCounts()
    (observation_matrix, minWordCountForTag, maxTagForWord) = trainingSet.getObservationMatrix()
    (transition_matrix) = trainingSet.getTransitionMatrix(tagsPreceding, totalBigrams)

    g = open("/Users/koushikreddy/Downloads/assgn2-test-set.txt", "r")
    sentences = []
    lines = []
    for line in g:
        if line == '\n':
            # firstWord = 1
            continue

        length = len(line)
        line = line[:length - 1]
        words = line.split('\t')
        count = words.count('.')

        if count == 1:
            lines.append(line)
            sentence = Sentence(lines)
            sentences.append(sentence)
            lines = []
            continue

        lines.append(line)
    g.close()
    testing = TestSetViterbi()
    testing.viterbi(sentences,wordset,tags,observation_matrix,transition_matrix,minWordCountForTag)






