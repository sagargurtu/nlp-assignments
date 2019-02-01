#!/bin/python

import re
import string

url_regex = re.compile(
            r'(^(?:http|ftp)s?://)?'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

prefixes = ('anti', 'auto', 'de', 'dis', 'down', 'extra', 'hyper', 'il', 'im', 'in', 'ir',
            'inter', 'mega', 'mid', 'mis', 'non', 'over', 'out', 'post', 'pre', 'pro', 're',
            'semi', 'sub', 'super', 'tele', 'trans', 'ultra', 'un', 'under', 'up')

noun_suffixes = ('age', 'al', 'ance', 'ence', 'dom', 'ee', 'er', 'or', 'hood', 'ism', 'ist', 'ity',
                 'ty', 'ment', 'ness', 'ry', 'ship', 'sion', 'tion', 'xion')

adjective_suffixes = ('able', 'ant', 'ary', 'ible', 'al', 'en', 'ese', 'ful', 'i', 'ic', 'ical',
                      'ish', 'ive', 'ian', 'like', 'less', 'ly', 'ous', 'y')

verb_suffixes = ('ate', 'en', 'ed', 'efy', 'ify', 'ise', 'ize')

adverb_suffixes = ('ly', 'ward', 'wards', 'wise')

abbreviations = ['aaf', 'adad', 'adih', 'adip', 'aeap', 'af', 'afaicr', 'afaics', 'afaict', 'afaik',
                 'afair', 'afaiu', 'afaiui', 'afap', 'afk', 'alol', 'asap', 'asl', 'a/s/l', 'aslp',
                 'a/s/l/p', 'ateotd', 'atb', 'atm', 'awol', 'aybabtu', 'ayb', 'b2b', 'b2c', 'b4',
                 'bbiab', 'bbq', 'bbl', 'bbs', 'bcnu', 'bff', 'bfn', 'blog', 'bofh', 'bot', 'brb',
                 'bsod', 'btdt', 'bttt', 'btw', 'cmiiw', 'cu', 'cya', 'dftt', 'dftba', 'dgaf', 'diaf',
                 'd/l', 'dnd', 'doa', 'eli5', 'eof', 'eom', 'eol', 'esad', 'eta', 'f9', 'faq', 'ffs',
                 'fmcdh', 'fml', 'foad', 'foaf', 'ftfy', 'ftr', 'ftl', 'ftw', 'ftw?', 'fu',
                 'fubar', 'fud', 'fwiw', 'fyi', 'gbtw', 'gf', 'gfu', 'gfy', 'gg', 'ggs', 'gj', 'gl',
                 'gmta', 'gtfo', 'gtg', 'g2g', 'gr', 'giyf;', 'hf', 'hth', 'ianal', 'ibtl', 'idk', 'iht',
                 'iirc', 'iiuc', 'ily', 'imao', 'imo', 'imho', 'imnsho', 'iow', 'irc', 'irl', 'istm',
                 'itym', 'iwsn', 'iydmma', 'iykwim', 'jftr', 'jk', 'j/k', 'jfgi;', 'k',
                 'kk', 'kms', 'kos', 'kthx', 'kthxbye', 'l8r', 'lfg',
                 'lfm', 'lmao', 'lmbo', 'lmfao', 'lmgtfy', 'lmirl', 'lmk', 'lol', 'ltns', 'lulz',
                 'lylab', 'lylas', 'mfw', 'mmo', 'mmorpg', 'motd', 'mtfbwy',
                 'myob', 'm8', 'n1', 'ne1', 'newb', 'n00b', 'ngl', 'nifoc', 'nm', 'n0rp', 'np', 'ns',
                 'nsoh', 'nsfw', 'nvm', 'nvmd', 'nm', 'oic', 'ofn', 'omg', 'omfg', 'omgwtf', 'omgwtfbbq',
                 'omw', 'onoz', 'op', 'os', 'ot', 'otb', 'otoh', 'otp', 'p2p', 'pebkac', 'pebcak',
                 'plmk', 'pmsl', 'pov', 'pl', 'ppl', 'pr0n', 'pw', 'pwned', 'qft',
                 'qwp', 'rehi', 'rl', 'rms', 'rofl', 'rotfl', 'roflmao', 'rotflmao', 'roflol', 'rotflol',
                 'rsn', 'rtfb', 'rtfs', 'rtfm', 'rtm', 'scnr', 'sfw', 'sk8', 'sk8r', 'sk8er',
                 'smh', 'snafu', 'sohf', 'sos', 'stfu', 'stfw', 'tanstaafl', 'tbf', 'tbh', 'tfw', 'tg',
                 'tgif', 'thx', 'thnx', 'tnx', 'tx', 'til', 'timwtk', 'tinc', 'tins', 'tl;dr',
                 'tmi', 'tos', 'ttbomk', 'ttfn', 'ttyl', 'ttyn', 'ttys', 'ty', 'tyt', 'tyvm', 'u',
                 'utfse', 'ugo', 'urs', 'w00t', 'w00t', 'woot', 'w/', 'w/o', 'wb', 'wbu', 'w/e', 'w/e',
                 'wrt', 'wtb', 'wtf', 'wtg', 'wth', 'wts', 'wtt', 'wug', 'wubu2', 'wuu2', 'wysiwyg', 'w8',
                 'ygm', 'yhbt', 'ykw', 'ymmv', 'yolo', 'yoyo', 'ytmnd', 'yw']

cluster_dict = {}


# Encodes a character to X, x, d or ?
def encode(char):
    if char.isupper():
        return 'X'
    elif char.islower():
        return 'x'
    elif char.isdigit():
        return 'd'
    else:
        return '?'


def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """

    f = open("brown_clustering.txt", "r")
    lines = f.readlines()

    for line in lines:
        line = line.strip()
        words = line.split()
        cluster_dict[words[0]] = words[1][:5]

    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # Feature 1: URL Check
    if re.match(url_regex, word):
        ftrs.append("IS_URL")

    # Feature 2: Check if word contains punctuation
    if any(char in set(string.punctuation) for char in word):
        ftrs.append("CONT_PUNC")

    # Feature 3: Check if word is a punctuation
    if word in set(string.punctuation):
        ftrs.append("IS_PUNC")

    # Feature 4: Check if the word's first letter is uppercase
    if word.istitle():
        ftrs.append("IS_TITLECASE")

    # Feature 5: Hashtag Check
    if word.startswith('#'):
        ftrs.append("IS_HASHTAG")

    # Feature 6: Usertag Check
    if word.startswith('@'):
        ftrs.append("IS_USERTAG")

    # Feature 7: Check if the word is a hyphenated word
    if len(re.findall(r'\w+(?:-\w+)+', word)) > 0:
        ftrs.append("IS_HYPHENWORD")

    # Feature 8: Add Brown Cluster membership
    '''if word in cluster_dict.keys():
        ftrs.append("CLUSTER_" + cluster_dict[word])'''

    word_shape = ''
    for char in word:
        word_shape += encode(char)
    # Feature 9: Add word shape
    ftrs.append("SHAPE=" + word_shape)

    word = word.lower()

    # Feature 10: Check if word is an internet slang abbreviation
    if word in abbreviations:
        ftrs.append("IS_ABBR")

    # Feature 11: Check if word contains a prefix
    if word.startswith(prefixes):
        ftrs.append("CONT_PREFIX")

    # Feature 12: Check if the word ends with a noun suffix
    if word.endswith(noun_suffixes):
        ftrs.append("CONT_NOUN_SUFFIX")

    # Feature 13: Check if the word ends with an adjective suffix
    if word.endswith(adjective_suffixes):
        ftrs.append("CONT_ADJ_SUFFIX")

    # Feature 14: Check if the word ends with a verb suffix
    if word.endswith(verb_suffixes):
        ftrs.append("CONT_VERB_SUFFIX")

    # Feature 15: Check if the word ends with an adverb suffix
    if word.endswith(adverb_suffixes):
        ftrs.append("CONT_ADV_SUFFIX")

    # Feature 16: Check if the word ends with a verb apostrophe
    if word.endswith(("'d", "'ve", "'m", "'ll", "'re")):
        ftrs.append("VERB_APOSTROPHE")

    # Feature 17: Check if the word ends with a noun apostrophe
    if word.endswith(("'s", "s'")):
        ftrs.append("NOUN_APOSTROPHE")

    # Feature 18: Check if the word ends with an adverb apostrophe
    if word.endswith("'t"):
        ftrs.append("ADV_APOSTROPHE")

    # Feature 19: Add Prefix Characters
    # Feature 20: Add Suffix Characters
    if len(word) > 3:
        ftrs.append("PREFIX=" + word[:3])
        ftrs.append("SUFFIX=" + word[-3:])
    elif len(word) == 3:
        ftrs.append("PREFIX=" + word[:2])
        ftrs.append("SUFFIX=" + word[-2])

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
