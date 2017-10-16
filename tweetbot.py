import urllib, json, requests
import tracery
import tracery.modifiers
import numpy as np
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pymongo
import random

# This conversational twitter bot is running on a 1 minute cron job on the fog.today server,
# posting tweets from @SnapshotReflect.

# The main method at the end of the file instantiates a QuestionGenerator and processes new
# tweets and DMs. QuestionGenerator uses the Microsoft Computer Vision API to get image
# content information, then expands a Tracery (https://github.com/galaxykate/tracery)
# grammar to generate a hopefully-relevant question. The conversation history is stored
# in a MongoDB database so that the bot can avoid repeating itself.

# TODO:
#   - generalize some of the question generators (i.e. "Is that your [object]" rather than
#     "Is that your book", etc
#   - database operations should be more atomic. right now it can put its database in a bad state
#     if it encounters a Computer Vision API error
#   - conversation should maybe be its own class?
#   - ending excuses are funny but need some work
#   - switch to real logging

class QuestionGenerator():
    def __init__(self, api_keys, cookies, verbose=True):
        self.api_keys = api_keys
        self.verbose = verbose

        # QUESTION GENERATION GRAMMAR
        # capitalized replacement values are one-time use per conversation
        self.generator = {
            # categories for specific types of images
            "single_person": ["#Isselfie#", "#Isselfie#", "#single_person_other#"], "single_person_other": ["#Feelings#", "#Technology#", "#Social#"],
            "people_group": ["#Where#", "#Whygroup#", "#Meet#", "#Know#"],
            "cat": ["#Awwname#", "I've always fancied myself a cat-chatbot. #Howlongpet#", "#Catsound#", "#Catstory#"],
            "dog": ["#Awwname#", "#Dog#", "#Dogappreciate#", "#Dogsound#"],
            "animal": ["#Awwname#", "#Whatkindanimal#", "#Whatanimaldoing#", "#Why#"],
            "plant": ["#Garden#", "#Why#", "#Context#", "#Grow#"],
            "outdoor": ["#Where#", "#Garden#", "#Technology#"],
            "mountain": ["#Where#", "#Vista#", "#Alone#", "#Untaken#"],
            "city": ["#Neighborhood#", "#Where#", "#Rediscover#", "#Untaken#"],
            "food": ["#Food#", "#Eatenwith#", "#Whofor#", "#Couldi#", "#Feelmeal#", "#Social#"],
            "indoor": ["#Comfortable#", "#Where#", "#Whofor#", "#Rediscover#"],
            "holding": ["#Holding#", "#Why#"],
            "book": ["#Book#", "#Why#", "#Whofor#"],
            "child": ["#Child#", "#Rediscover#", "#Untaken#"],
            "box": ["#Box#", "#Why#"],
            "officey": ["#Remember#", "#Feelings#", "#Why#"],
            
            # conversation
            "followup_selfie": ["#Social#", "#Whofor#", "#Context#", "#Feelings#"],
            "followup_generic": ["#Tellmemore#", "#Rediscover#", "#Feelings#"],
            "followup_outdoor": ["#Weather#"],
            "followup_city": ["#Return# #Dodifferent#"],
            "followup_food": ["#origin#", "#food#", "#LinkFood#"],
            
            # wildcard category
            "origin": ["#Why#",
                "#Whofor#",
                "#Where#",
                "#Rediscover#",
                "#Feelings#",
                "#Isms#",
                "#Context#",
                "#Encouraging#",
                "#Surprised#",
                "#Social#",
                "#Untaken#",
                "#Composition#",
                "#Alone#",
                "#Technology#",
                "#Remember#",
                "#Return#"],
            
            # individual question concepts (not to be repeated)
            # animals
            "Dog": ["I love dogs. #Howlongpet#"],
            "Cat": ["Oh, a cat! I love cats, but sometimes they can be real jerks. What's yours like?"],
            "Catsound": ["Meow! Does your cat like people?"],
            "Dogsound": ["Woof! Do you have more than one dog?"],
            "Dogappreciate": ["Oh, a dog! I don't like dogs much myself... what do you appreciate about this one?"],
            "Catstory": ["How did you first meet this cat?", "What's your favorite story with this cat?", "What's the story of this cat?"],
            "Whatanimaldoing": ["What did you seem them doing that #inspire.ed# you to take a #picture#?"],
            "Whatkindanimal": ["What kind of animal is that?"],
            "Awwname": ["Aww, what's their name?"],
            "Howlongpet": ["How long have you had them?"],

            # plants
            "Garden": ["In some sense, it's like a garden. What do you like about it?", "Is that a garden?", "Is that your garden?", "What a nice place to sit and think. Do you agree?"],
            "Grow": ["Did you grow that?", "How long have you known that plant?", "How long has that been growing?", "When did you first notice that plant?"],

            # food
            "Food": ["Did you make that?",  "How did that taste?", "How would you make that differently next time?"],
            "Feelmeal": ["#howwhat.capitalize# did you feel #beforeafter# #this# meal?", "#howwhat.capitalize# did you feel #beforeafter# you ate #this#?"],
            "Eatenwith": ["Who did you eat that with?", "Who do you wish you could have eaten that with?"],
            "LinkFood": ["Do you have another food photo to share?"],
            "Couldi": ["That looks good, do you think I could make that?", "That recipe looks complicated, I bet I couldn't make it. Or maybe I could?", "That looks hard to cook, I wonder if I could make it. Did you?"],

            # children
            "Child": ["Is that your #kid#?", "What does that #kid# want to be when they grow up?"],

            # selfie
            "Isselfie": ["Is #this# you?", "Is #this# a #picture# of you?", "Is #this# a selfie?"],
            
            # generic followup
            "Tellmemore": ["Can you tell me more about that?", "What do you mean?", "Why do you think that is?", "Is that really true?", "Really?"],
            
            # place
            "Weather": ["Was the weather what you expected?"],
            "Where": ["Where was #this# #taken#?", "Where did you #take# #this# #picture#?", "Do you #visit# this #place# #often#?", "Have you visited this #place# before?"],
            "Vista": ["That #scenery# is #encouraging_words#. Did you go #recreation#?"],
            "Neighborhood": ["Did you enjoy your time in this neighborhood?", "What would you compare this neighborhood to?"],
            "Return": ["Do you plan to return here?", "Will you visit this place again?"],
            "Dodifferent": ["What will you do differently?", "What would you do differently?", "What else would you do?"],
            "Comfortable": ["Is this a place you like to be?", "Is this a place you feel comfortable?", "Is this a place you feel at home?"],

            # composition
            "Isms": ["Were you thinking about #isms# when you #took# #this# #picture#?"],
            "Composition": ["Why didn't you zoom in?", "Why didn't you zoom out?", "What was behind you?"],
            "Technology": ["How would #this# #picture# have been different if you took it with #device#?"],

            # why
            "Encouraging": ["#encouraging_words.capitalize#! What #inspire.ed# you to #take# #this#?"],
            "Surprised": ["#interjections.capitalize#! I'm not sure what to say... do you #like# #this# #picture#?"],
            "Why": ["Why did you #take# #this# #picture#?"],

            # context
            "Untaken": ["#encouraging_words.capitalize#... but what didn't you take a picture of that day?", "#encouraging_words.capitalize#... but what didn't you take a picture of there?"],
            "Context": ["Where did you go after you #took# #this# #picture#?", "What did you do #beforeafter# you #took# #this# #picture#?", "What were you doing when you #took# #this# #picture#?"],
            "Social": ["I bet you didn't put #this# on #facebook#. Why not?", "I bet you put #this# on #facebook#. Who did you want to see it?", "Did you think about uploading this to #facebook#?"],

            # people
            "Alone": ["Were you alone when you #took# #this# #picture#? Why?", "Who were you with when you #took# #this# #picture#?", "Who did you spend this day with?"],
            "Whygroup": ["#Why#", "Why did you #take# #this# #picture# here?", "Why did you choose this spot?"],
            "Whofor": ["Who did you #take# this #picture# for?", "#whowhat.capitalize# were you thinking about when you #took# #this# #picture#?"],
            "Meet": ["How did you meet #these# people?", "Who did you most recently meet?", "Who have you known the longest?"],
            "Know": ["How do you know #these# people?", "Do you consider yourself close to #this# group?", "What do you normally do with #these# people?"],

            # feelings
            "Feelings": ["#howwhat.capitalize# did you feel #beforeafter# you #took# #this# #picture#?"],
            "LinkFeelings": ["Did taking the #picture# change how you feel?", "Do you frequently feel like #this# when you take photos?"],

            # memories
            "Remember": ["What did you want to remember from #this# scene?", "What do you want to remember most from #this# #picture#?"],
            "Rediscover": ["If you were to #find# this #picture# in #duration#, what #wouldwouldnt# you want to #remember# about it?"],
            "Important": ["In #duration#, do you think this #picture# will be more or less important to you than it is now?", "Is this #picture# important to you? Why?"],

            # objects
            "Holding": ["What do they have?", "What are they holding?", "What is that?"],
            "Book": ["What's your favorite book?", "Can you recommend a book?", "Have you read anything good lately?"],
            "Box": ["What's in the box?", "Ooh, what's inside the box?", "Open up the box! Tell me what's inside!"],

            # thesauruses to make question generation more interesting
            "device": ["a film camera", "an SLR", "your old smartphone", "your grandfather's camera"],
            "find": ["rediscover", "find", "run across", "turn up"],
            "visit": ["visit", "pass through", "walk around", "#take# #picture.s# in", "explore"], 
            "place": ["place", "location", "area", "site", "neighborhood", "street"],
            "often": ["often", "frequently", "occasionally", "sometimes", "all the time", "every day"],
            "facebook": ["Facebook", "Snapchat", "Instagram", "Twitter"],
            "beforeafter": ["before", "after", "just before", "just after"],
            "whowhat": ["who", "what"],
            "howwhat": ["how", "what"],
            "take": ["take", "capture", "photograph", "shoot", "record"],
            "taken": ["taken", "captured", "photographed", "shot", "recorded"],
            "took": ["took", "captured", "photographed", "shot", "recorded"],
            "picture": ["picture", "photo", "image", "snapshot", "pic", "photograph", "shot"],
            "this": ["this", "that"],
            "these": ["these", "those"],
            "yearmonth": ["year", "month"],
            "duration": ["#singular#", "#plural#"],
            "singular": ["one #yearmonth#"],
            "yearmonth": ["year", "month"],
            "plural": ["#num# #yearsmonths#"],
            "num": ["two", "three", "four", "five"],
            "yearsmonths": ["years", "months"],
            "remember": ["remember", "tell yourself", "tell #person#", "explain", "forget", "analyze", "figure out", "feel", "believe", "remember"],
            "wouldwouldnt": ["would", "wouldn't"],
            "person": ["yourself", "your children", "your partner", "your coworker", "your father", "your mother", "your brother", "your sister", "your ex"],
            "inspire": ["inspire", "motivate", "prompt", "influence", "provoke", "spark"],
            "like": ["like", "love", "dislike", "enjoy", "feel strongly about", "admire", "regret"],
            "scenery": ["scenery", "view", "vista"],
            "recreation": ["hiking", "for a walk", "exploring", "climbing", "camping", "roadtripping"],
            "kid": ["kid", "child", "little one"],

            "isms": ["abstract expressionism","action painting","aestheticism","art deco","art nouveau","avant-garde","baroque","bauhaus","classicism","cloisonnism","color field painting","conceptual art","cubism","cubo-futurism","dada","dadaism","deformalism","divisionism","eclecticism","ego-futurism","existentialism","expressionism","fauvism","fluxus","formalism","futurism","geometric abstraction","gothic art","historicism","humanism","hyperrealism","idealism","illusionism","impressionism","installation art","intervention art","jugendstil","kinetic art","land art","luminism","lyrical abstraction","mail art","manierism","mannerism","maximalism","merovingian","metaphysical art ","minimalism","modern art","modernism","monumentalism","multiculturalism","naturalism","neo-classicism","neo-dada","neo-expressionism","neo-fauvism","neo-geo","neo-impressionism","neo-minimalism","neoclassicism","neoism","new media art","new objectivity","nonconformism","nouveau realisme","op art","orphism","outsider art","performance art","perspectivism","photorealism","pointilism","pop art","post-conceptualism","post-impressionism","post-minimalism","post-structuralism","postminimalism","postmodernism","precisionism","primitivism","purism","rayonism","realism","relational art","remodernism","renaissance","rococo","romanesque","romanticism","russian futurism","russian symbolism","secularism","situationalism","social realism","socialist realism","sound art","street art","structuralism","suprematism","surrealism","symbolism","synchromism","synthetism","tachism","tachisme","tonalism","video art","video game art","vorticism"],
            "encouraging_words": ["amazing","awesome","beautiful","bravo","brilliant","breathtaking","congratulations","cool","dazzling","delightful","electrifying","elegant","enchanting","excellent","exciting","fabulous","fantastic","fun","genius","groundbreaking","heavenly","impressive","innovative","inventive","kind","legendary","lifechanging","lovely","magical","marvelous","masterful","miraculous","original","perfect","phenomenal","powerful","remarkable","rejuvenating","resounding","skillful","stupendous","stunning","sweet","terrific","thoughtful","thrilling","wonderful","wondrous"],
            "interjections": ["aah","ack","agreed","ah","aha","ahem","alas","all right","amen","argh","as if","aw","ay","aye","bah","blast","boo hoo","bother","boy","brr","by golly","bye","cheerio","cheers","chin up","come on","crikey","curses","dear me","doggone","drat","duh","easy does it","eek","egads","er","exactly","fair enough","fiddle-dee-dee","fiddlesticks","fie","foo","fooey","gadzooks","gah","gangway","g'day","gee","gee whiz","geez","gesundheit","get lost","get outta here","go on","good","good golly","good job","gosh","gracious","great","grr","gulp","ha","ha-ha","hah","hallelujah","harrumph","haw","hee","here","hey","hmm","ho hum","hoo","hooray","hot dog","how","huh","hum","humbug","hurray","huzza","I say","ick","is it","ixnay","jeez","just kidding","just a sec","just wondering","kapish","la","la-di-dah","lo","look","look here","long time","lordy","man","meh","mmm","most certainly","my","my my","my word","nah","naw","never","no","no can do","nooo","not","no thanks","no way","nuts","oh","oho","oh-oh","oh no","okay","okey-dokey","om","oof","ooh","oopsey","over","oy","oyez","peace","pff","pew","phew","pish posh","psst","ptui","quite","rah","rats","ready","right","right on","roger","roger that","rumble","say","see ya","shame","shh","shoo","shucks","sigh","sleep tight","snap","sorry","sssh","sup","ta","ta-da","ta ta","take that","tally ho","tch","thanks","there","there there","time out","toodles","touche","tsk","tsk-tsk","tut","tut-tut","ugh","uh","uh-oh","um","ur","urgh","very nice","very well","voila","vroom","wah","well","well done","well, well","what","whatever","whee","when","whoa","whoo","whoopee","whoops","whoopsey","whew","why","word","wow","wuzzup","ya","yea","yeah","yech","yikes","yippee","yo","yoo-hoo","you bet","you don't say","you know","yow","yum","yummy","zap","zounds","zowie","zzz"
          ]
        }

        # PROMPT GENERATOR GRAMMAR
        # this grammar is used to generate requests for more photos
        self.prompt_generator = {
            "origin": [
                "#show_me# #picture.a# that makes you feel #feeling#.", "#can_i_see# #picture.a# that makes you feel #feeling#?",
                "#show_me# #picture.a# from #duration# ago.", "#can_i_see# #picture.a# from #duration# ago?",
                "#show_me# your favorite recent #picture#.", "#can_i_see# your favorite recent #picture#?",
                "#show_me# the last #picture#  you put on #facebook#.", "#can_i_see# the last #picture# you put on #facebook#?",
                "#show_me# the #picture_adjective# #picture# you took in the last #duration#.", "#can_i_see# the #picture_adjective# #picture# you took in the last #duration#?",
                "#show_me# #encouraging_words.a# #picture#.", "#can_i_see# #encouraging_words.a# #picture#?",
                "#show_me# #picture.a# that makes you say \"#interjections#!\"", "#can_i_see# #picture.a# that makes you say \"#interjections#?\"",
                "#show_me# #picture.a# you haven't seen in a while.", "#can_i_see# #picture.a# you haven't looked at in a while?",
                "#show_me# #picture.a# you #took# for #person#.", "#can_i_see# #picture.a# you #took# for #person#?",
                "#show_me# #picture.a# from #place.a# you #visit# #often#.", "#can_i_see# #picture.a# from #place.a# you #visit# #often#?",
                "#show_me# #picture.a# you #took# #alone#.", "#can_i_see# #picture.a# you #took# #alone#?",
                "#show_me# #picture.a# of a recent meal.", "#can_i_see# #picture.a# of a recent meal?"
            ],
            
            "picture_adjective": ["most interesting", "most exciting", "most playful", "most beautiful", "biggest", "smallest", "darkest", "brightest", "most colorful", "loudest", "quietest", "most experimental", "most surprising", "least expected", "least desired", "least interesting", "ugliest", "worst", "best"],
            "feeling_noun": ["proud", "in awe", "accepting", "neglectful", "regretful"],
            "feeling": ["happy", "nostalgic", "anxious", "energetic", "tired", "forgetful", "old", "young", "love", "regret", "fortunate", "realized", "successful", "proud", "uncomfortable", "fantastic", "surprised", "satisfied", "proud"],

            "show_me": ["Show me", "Let me see", "Let me look at", "Find", "Let's talk about", "Let's take a look at", "I'd like to take a look at", "I'd like to see", "Search for", "Look for", "Browse for", "Send me"],
            "can_i_see": ["Can I see", "Can you find", "Can you show me", "Could I see", "Could you let me take a look at", "Could we talk about", "Can we talk about", "Can you send me", "Could you send me"],

            "alone": ["alone", "by yourself", "with friends", "with a friend", "with a group of friends", "with co-workers", "with classmates", "with family", "with a family member", "with a stranger", "with a new friend"],
            "find": ["rediscover", "find", "run across", "turn up"],
            "visit": ["visit", "pass through", "walk around", "#take# pictures in"], 
            "place": ["place", "location", "area", "site", "neighborhood", "street"],
            "often": ["often", "frequently", "occasionally", "sometimes", "all the time", "every day"],
            "facebook": ["Facebook", "Snapchat", "Instagram", "Twitter"],
            "beforeafter": ["before", "after", "just before", "just after"],
            "whowhat": ["who", "what"],
            "howwhat": ["how", "what"],
            "take": ["take", "capture", "photograph", "shoot", "record"],
            "taken": ["taken", "captured", "photographed", "shot", "recorded"],
            "took": ["took", "captured", "photographed", "shot", "recorded", "took", "took"],
            "picture": ["picture", "photo", "image", "snapshot", "pic", "photograph", "shot"],
            "this": ["this", "that"],
            "yearmonth": ["year", "month", "week"],
            "duration": ["#singular#", "#plural#"],
            "singular": ["one #yearmonth#"],
            "plural": ["#num# #yearsmonths#"],
            "num": ["two", "three", "four", "five"],
            "yearsmonths": ["years", "months", "weeks"],
            "remember": ["remember", "tell yourself", "tell #person#", "explain", "forget", "analyze", "figure out", "feel", "believe", "remember"],
            "wouldwouldnt": ["would", "wouldn't"],
            "person": ["yourself", "your children", "your partner", "your coworker", "your father", "your mother", "your brother", "your sister", "your ex"],
            "inspire": ["inspire", "motivate", "prompt", "influence", "provoke", "spark", "sway"],
            "like": ["like", "love", "dislike", "enjoy", "feel strongly about", "admire", "regret"],
            
            "encouraging_words": ["amazing","awesome","beautiful","bravo","brilliant","breathtaking","congratulations","cool","dazzling","delightful","electrifying","elegant","enchanting","excellent","exciting","fabulous","fantastic","fun","genius","groundbreaking","heavenly","impressive","innovative","inventive","kind","legendary","lifechanging","lovely","magical","marvelous","masterful","miraculous","original","perfect","phenomenal","powerful","remarkable","rejuvenating","resounding","skillful","stupendous","stunning","sweet","terrific","thoughtful","thrilling","wonderful","wondrous"],
            "interjections": ["aah","ack","agreed","ah","aha","ahem","alas","all right","amen","argh","as if","aw","ay","aye","bah","blast","boo hoo","bother","boy","brr","by golly","bye","cheerio","cheers","chin up","come on","crikey","curses","dear me","doggone","drat","duh","easy does it","eek","egads","er","exactly","fair enough","fiddle-dee-dee","fiddlesticks","fie","foo","fooey","gadzooks","gah","gangway","g'day","gee","gee whiz","geez","gesundheit","get lost","get outta here","go on","good","good golly","good job","gosh","gracious","great","grr","gulp","ha","ha-ha","hah","hallelujah","harrumph","haw","hee","here","hey","hmm","ho hum","hoo","hooray","hot dog","how","huh","hum","humbug","hurray","huzza","I say","ick","is it","ixnay","jeez","just kidding","just a sec","just wondering","kapish","la","la-di-dah","lo","look","look here","long time","lordy","man","meh","mmm","most certainly","my","my my","my word","nah","naw","never","no","no can do","nooo","not","no thanks","no way","nuts","oh","oho","oh-oh","oh no","okay","okey-dokey","om","oof","ooh","oopsey","over","oy","oyez","peace","pff","pew","phew","pish posh","psst","ptui","quite","rah","rats","ready","right","right on","roger","roger that","rumble","say","see ya","shame","shh","shoo","shucks","sigh","sleep tight","snap","sorry","sssh","sup","ta","ta-da","ta ta","take that","tally ho","tch","thanks","there","there there","time out","toodles","touche","tsk","tsk-tsk","tut","tut-tut","ugh","uh","uh-oh","um","ur","urgh","very nice","very well","voila","vroom","wah","well","well done","well, well","what","whatever","whee","when","whoa","whoo","whoopee","whoops","whoopsey","whew","why","word","wow","wuzzup","ya","yea","yeah","yech","yikes","yippee","yo","yoo-hoo","you bet","you don't say","you know","yow","yum","yummy","zap","zounds","zowie","zzz"]
        }

        # CONVERSATION ENDING GRAMMAR (not Tracery, yet)
        #  this is used to "gracefuly" exit conversations
        self.conversation_excuses = [
            "I've got another call in a couple minutes; thanks so much for speaking with me, and I'll talk to you again [soon/in X days].",
            "My battery's pretty low, so I'm going to hop off. Have an amazing day!",
            "It sounds like we've covered everything we needed to, so I'll let you go. Thank you for such a productive meeting!",
            "Can't believe it's already [time of day]. I'm sure you've got lots of things on your agenda, so I'll let you get to them. Let me know if there's anything else I can do for you.",
            "Please excuse me, I'm going to make a quick restroom trip. It was lovely to meet you!",
            "I've had such a nice time talking to you. And I'll definitely connect with you on LinkedIn so I can keep up with all of your cool ventures.",
            "I'm sorry to leave so quickly, but it's been a pleasure and I hope we can reconnect soon. Do you have a business card?",
            "I'm going to mingle a bit more, but before I go, can I introduce you to someone? [Introduce them to each other.] I'll let you guys talk!",
            "I've got to head back to my desk and work on [X project]. Let's catch up at happy hour!",
            "I know you've got a crazy schedule, so I'll let you get back to it.",
            "I'd love to hear about your [work/side gig/current initiative] when we've got more time, so let's plan lunch!",
            "There are a couple emails I have to send before [time], so I'm going to have to excuse myself.",
            "Looks like we've hit everything on the agenda. If no one has anything else to discuss, see you all at next week's meeting.",
            "There's another meeting in this conference room right after us, so we should probably clear out and let the next guys in.",
            "Great to see we finished 15 minutes early! Going to go knock out some quick emails.",
            "[Person], are you walking back to your desk? I'll walk with you.",
            "Thanks, everyone, for a productive meeting! I can send around our notes later this afternoon.",
            "I really appreciate you taking the time to speak with me. Have a fantastic rest of your day, and I'll look for your [email/notes/report/follow-up].",
            "Your ideas sound really promising; can't wait to see them in action. In the meantime, you've probably got a lot on your plate, so I'll let you get back to work.",
            "I want to get you the answers to your questions as soon as possible, so I'm going to get off now - look for my email by the end of the [day/week].",
            "Wow, I can't believe it's already [time]. Do you mind if I hang up and finish up my to-do list?",
            "All right, I need to go check in with my team. It's been great chatting.",
            "Anyway, I don't want to monopolize all your time.",
            "So, listen, it's been great catching up with you. Let's exchange cards?",
            "Hold on, I've got to take this."
        ]

        client = pymongo.MongoClient('localhost', 27017)
        self.db = client.reflect

        self.grammar = tracery.Grammar(self.generator)
        self.grammar.add_modifiers(tracery.modifiers.base_english)

        self.prompt_grammar = tracery.Grammar(self.prompt_generator)
        self.prompt_grammar.add_modifiers(tracery.modifiers.base_english)

        auth = tweepy.OAuthHandler(self.api_keys['twitter']['consumer_token'], self.api_keys['twitter']['consumer_secret'])
        auth.set_access_token(self.api_keys['twitter']['key'], self.api_keys['twitter']['secret'])
        self.twitter = tweepy.API(auth)
        self.cookies = cookies
       
    def analyze_image(self, im_url, dm=False):
        subscription_key = self.api_keys['microsoft']
        uri_base = 'https://westcentralus.api.cognitive.microsoft.com'

        headers = {
            # Request headers.
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': subscription_key,
        }

        params = urllib.urlencode({
            # Request parameters. All of them are optional.
            'visualFeatures': 'Tags,Faces',
            'language': 'en',
        })
            
        try:
            # if it's a DM we have to load up some cookies, download the image, and reupload it to microsoft
            if dm:
                session = requests.Session()

                for cookie in cookies:
                    session.cookies.set(cookie['name'], cookie['value'])

                response = session.get(im_url, stream=True)
                response.raw.decode_content = True
                headers['Content-Type'] = 'application/octet-stream'
                body = response.raw
            
            # otherwise, it's publicly accessible
            else:
                body = "{'url':'" + im_url + "'}"
            
            r = requests.post(uri_base + "/vision/v1.0/analyze?%s" % params, headers=headers, data=body)
            analysis = r.json()
            print(analysis)
            return analysis
        except Exception as e:
            print('Error:')
            print(e)
            
            return e

    def get_question(self, conversation, last_response='', dm=False):
        # is there a response earlier in the conversation, and did we ask a question?
        if (last_response != ''):
            if conversation['asked'] == 'selfie':
                isselfie = self.confirmed(last_response)
                if self.verbose:
                    print "- get_question | believe selfie is " + str(isselfie) + ' from "' + last_response + '"'
                conversation['is_selfie'] = isselfie
            
        response = ''

        if 'image_details' not in conversation:
            if self.verbose:
                print "- get_question | analyzing image"
            image_details = self.analyze_image(conversation['image'], dm=dm)
            conversation['image_details'] = image_details
        
            num_faces = len(conversation['image_details']['faces'])
            
            if num_faces > 0:
                face_sizes = [float(f['faceRectangle']['height'] * f['faceRectangle']['width'])/float(conversation['image_details']['metadata']['height'] * conversation['image_details']['metadata']['width']) for f in conversation['image_details']['faces']]
                conversation['image_details']['face_sizes'] = face_sizes
                num_prominent_faces = np.sum(conversation['image_details']['face_sizes'] > 0.1)
                num_children = np.sum([face['age'] < 10 for face in conversation['image_details']['faces']])
            else:
                num_prominent_faces = 0
                num_children = 0
            
            conversation['num_faces'] = num_faces
            conversation['num_children'] = num_children
            conversation['num_prominent_faces'] = num_prominent_faces
        
        tags = [t['name'] for t in conversation['image_details']['tags'] if t['confidence'] > 0.6]

        # different tags that MS reports in the image have different priority levels
        # only if there is no high priority tag is a medium priority tag selected, etc
        high_priority = ['dog', 'cat', 'people', 'person', 'food']
        medium_priority = ['mountain', 'road', 'city', 'street', 'animal', 'book']
        low_priority = ['outdoor', 'indoor', 'box', 'holding', 'document', 'text', 'envelope']
        
        high_present = [t for t in high_priority if t in tags]
        medium_present = [t for t in medium_priority if t in tags]
        low_present = [t for t in low_priority if t in tags]
        
        # people are important! if someone is in the photo but it wasn't already tagged, tag it with people now
        if (conversation['num_prominent_faces'] > 0):
            high_present.append('people')

        if conversation['num_messages'] == 0:
            if (conversation['num_children']) > 0:
                high_present = ['child']
            elif (conversation['num_prominent_faces'] == 1):
                high_present = ['person']
        
        choosen = 'default'
        if len(high_present) > 0:
            choosen = random.choice(high_present)
        elif len(medium_present) > 0:
            choosen = random.choice(medium_present)
        elif len(low_present) > 0:
            choosen = random.choice(low_present)
            
        conversation['topic'] = choosen
        
        expansion = {
            'dog': '#dog#',
            'cat': '#cat#',
            'animal': '#animal#',
            'person': '#single_person#',
            'people': '#people_group#',
            'plant': '#plant#', 
            'outdoor': '#outdoor#',
            'mountain': '#mountain#',
            'road': '#city#',
            'city': '#city#',
            'street': '#city#',
            'indoor': '#origin#',
            'food': '#food#',
            'box': '#box#',
            'holding': '#holding',
            'book': '#book#',
            'document': '#officey#',
            'envelope': '#officey#',
            'text': '#officey#',
            'default': '#origin#'
        }
            
        expansion_point = expansion[choosen]
            
        # continuing a conversation    
        if conversation['is_selfie']:
            expansion_point = "#followup_selfie#"
        try:
            if conversation['eliminated_expansions'][-1] == "Feelings":
                expansion_point = "#LinkFeelings#"
        except:
            pass

        if self.verbose:
            print "- get_question | expansion_point: " + expansion_point
        expansion = self.grammar.expand(expansion_point)
        num_tries = 0
        response = ''
        
        flattened_grammar = self.flatten_grammar(expansion)
        
        while(len(response) == 0):
            questions_used = [c.raw for c in flattened_grammar if c.type == 1 and c.raw[0].isupper()]
            print "- get_question | using question " + str(questions_used)
            if any([q in conversation['eliminated_expansions'] for q in questions_used]):
                if self.verbose:
                    print "- get_question | already used question"
                # now we need to try again
                num_tries += 1
                if num_tries > 5:
                    expansion_point = '#origin#'
                expansion = self.grammar.expand(expansion_point)
                flattened_grammar = self.flatten_grammar(expansion)
            else:
                response = expansion.finished_text
                conversation['eliminated_expansions'] += questions_used
        
        if 'Isselfie' in [c.raw for c in flattened_grammar if c.type == 1]:
            conversation['asked'] = 'selfie'
            if self.verbose:
                print "- get_question | asking if this is a selfie"
        else:
            conversation['asked'] = None
        
        conversation['history'].append(response)
        
        conversation['num_messages'] += 1
        
        return (response, conversation)

    # flatten the grammar tree into a list
    def flatten_grammar(self, a):
        if a.type != 0:
            return [a] + reduce(lambda x,y: x+self.flatten_grammar(y), a.children,[])
        else:
            return [a]

    def new_conversation(self, url):
        conversation = {
            'num_messages': 0,
            'history': [],
            'eliminated_expansions': [],
            'user': '',
            'image': url,
            'is_selfie': False,
            'asked': None,
            'involved_tweets': [],
            'last_tweet_id': ''}
        return conversation

    def cleanup_tweet(self, tw, dm=False):
        if dm:
            splat = tw.text.split(' ')
        else:
            splat = tw.full_text.split(' ')
        splat = [s for s in splat if s[:4] != 'http' and s[:1] != '@']
        return " ".join(splat)

    def process_tweet(self, tw):
        if 'media' in tw.entities:
            if self.verbose:
                print '- process_tweet | tweet has image, creating new conversation'
            conversation = self.new_conversation(tw.entities['media'][0]['media_url'])
            self.db.conversations.insert_one(conversation)
        else:
            if tw.in_reply_to_status_id:
                conversations = list(self.db.conversations.find({'last_tweet_id': tw.in_reply_to_status_id}))
                if len(conversations) == 1:
                    if self.verbose:
                        print '- process_tweet | found matching conversation for thread'
                    conversation = conversations[0]
                elif len(conversations) == 0:
                    print '! process_tweet | ERROR, no matching conversation found'
                    output = self.prompt_grammar.flatten("#origin#")
                    response = self.twitter.update_status('@' + tw.user.screen_name + ' ' + output, tw.id)
                    print '* process_tweet | tweeting new tweet:\n\t' + output
                    return -1
                else:
                    print '! process_tweet | ERROR, multiple matching conversations found'
                    output = self.prompt_grammar.flatten("#origin#")
                    response = self.twitter.update_status('@' + tw.user.screen_name + ' ' + output, tw.id)
                    print '* process_tweet | tweeting new tweet:\n\t' + output
                    return -1
            else:
                print '! process_tweet | ERROR, tweet has no media and is not reply'
                output = self.prompt_grammar.flatten("#origin#")
                response = self.twitter.update_status('@' + tw.user.screen_name + ' ' + output, tw.id)
                print '* process_tweet | tweeting new tweet:\n\t' + output    
                return -1
        
        if conversation['num_messages'] < 4:
            output = self.get_question(conversation, last_response=self.cleanup_tweet(tw))
            
            # tweet generated question
            print '* process_tweet | tweeting new tweet:\n\t' + output[0]
            
            response = self.twitter.update_status('@' + tw.user.screen_name + ' ' + output[0], tw.id)
            conversation = output[1]
            conversation['involved_tweets'] += [tw.id, response.id]
            conversation['last_tweet_id'] = response.id
            
            self.db.conversations.update_one({'image': conversation['image']}, {'$set': conversation})
        elif conversation['num_messages'] < 5:
            output = random.choice(conversation_excuses)

            # tweet generated question
            print '* process_tweet | tweeting new tweet:\n\t' + output
            
            response = self.twitter.update_status('@' + tw.user.screen_name + ' ' + output, tw.id)
            conversation['num_messages'] += 1
            conversation['involved_tweets'] += [tw.id, response.id]
            conversation['last_tweet_id'] = response.id
            
            self.db.conversations.update_one({'image': conversation['image']}, {'$set': conversation})
        else:
            print '- process_tweet | too many tweets'

        return

    # processing DMs is a little bit different because there is no threading on the conversation
    def process_dm(self, tw):
        if 'media' in tw.entities:
            if self.verbose:
                print '- process_dm | tweet has image, creating new conversation'
            conversation = self.new_conversation(tw.entities['media'][0]['media_url'])
            conversation['sender_id'] = tw.sender_id

            # disable old conversations with this user, but maintain them in the database for posterity
            self.db.conversations.update_many({'sender_id': tw.sender_id}, {'$set': {'sender_id': tw.sender_id*-1}})
            self.db.conversations.insert_one(conversation)
        else:
            conversations = list(self.db.conversations.find({'sender_id': tw.sender_id}))
            if len(conversations) == 1:
                if self.verbose:
                    print '- process_dm | found matching conversation for thread'
                conversation = conversations[0]
            elif len(conversations) == 0:
                print '! process_dm | ERROR, no matching conversation found'
                # respond with a photo prompt
                output = self.prompt_grammar.flatten("#origin#")
                response = self.twitter.send_direct_message(tw.sender_id, text=output)
                print '* process_dm | tweeting new tweet:\n\t' + output
                return -1
            else:
                print '! process_dm | ERROR, multiple matching conversations found'
                # respond with a photo prompt
                output = self.prompt_grammar.flatten("#origin#")
                response = self.twitter.send_direct_message(tw.sender_id, text=output)
                print '* process_dm | tweeting new tweet:\n\t' + output
                return -1
        
        if conversation['num_messages'] < 4:
            # for DMs we also have to let the analyze_image method know that it needs to upload a copy of the photo to microsoft
            output = self.get_question(conversation, last_response=self.cleanup_tweet(tw, dm=True), dm=True)
            
            # tweet generated question
            print '* process_dm | tweeting new tweet:\n\t' + output[0]
            
            response = self.twitter.send_direct_message(tw.sender_id, text=output[0])
            conversation = output[1]
            conversation['involved_tweets'] += [tw.id, response.id]
            conversation['last_tweet_id'] = response.id
            
            self.db.conversations.update_one({'image': conversation['image']}, {'$set': conversation})
        elif conversation['num_messages'] < 5:
            output = random.choice(conversation_excuses)

            # tweet generated question
            print '* process_dm | tweeting new tweet:\n\t' + output
            
            response = self.twitter.send_direct_message(tw.sender_id, text=output)
            conversation['num_messages'] += 1
            conversation['involved_tweets'] += [tw.id, response.id]
            conversation['last_tweet_id'] = response.id
            
            self.db.conversations.update_one({'image': conversation['image']}, {'$set': conversation})
        else:
            print '- process_dm | too many responses... waiting for new media message'

        return

    def confirmed(self, text):
        sid = SentimentIntensityAnalyzer()
        sent = sid.polarity_scores(text)
        return sent['compound'] > 0

    def process_new_tweets(self):
        status = list(self.db.status.find({'type': 'current'}))[0]
        last_tweet = int(status['last_tweet'])
        
        tweets = self.twitter.mentions_timeline(since_id=last_tweet, tweet_mode='extended', count=100)
        if len(tweets) > 0:
            if self.verbose:
                print '- process_new_tweets | found ' + str(len(tweets)) + ' new tweets since ' + str(last_tweet)
            
        for tweet in tweets[::-1]:
            if self.verbose:
                print '- process_new_tweets | replying to tweet ' + str(tweet.id)
            
            self.process_tweet(tweet)
        
        if len(tweets) > 0:
            if self.verbose:
                print '- process_new_tweets | updating last_tweet status to ' + str(tweets[0].id)

            self.db.status.update_one({'type': 'current'}, {'$set': {'last_tweet': tweets[0].id}})

    def process_new_dms(self):
        status = list(self.db.status.find({'type': 'current'}))[0]
        last_dm = int(status['last_dm'])
        
        tweets = self.twitter.direct_messages(since_id=last_dm, count=100)
        if len(tweets) > 0:
            if self.verbose:
                print '- process_new_dms | found ' + str(len(tweets)) + ' new DMs since ' + str(last_dm)
            
        for tweet in tweets[::-1]:
            if self.verbose:
                print '- process_new_dms | replying to DM ' + str(tweet.id)
            
            self.process_dm(tweet)
        
        if len(tweets) > 0:
            if self.verbose:
                print '- process_new_dms | updating last_dm status to ' + str(tweets[0].id)

            self.db.status.update_one({'type': 'current'}, {'$set': {'last_dm': tweets[0].id}})

    def clear_new_tweets(self):
        tweets = self.twitter.mentions_timeline(tweet_mode='extended', count=1)
        self.db.status.update_one({'type': 'current'}, {'$set': {'last_tweet': tweets[0].id}})

if __name__ == '__main__':
    with open('/home/loganw/tweetbot/api_keys.json') as key_file:
        api_keys = json.load(key_file)

    with open('/home/loganw/tweetbot/cookies.json') as cookie_file:    
        cookies = json.load(cookie_file)

    generator = QuestionGenerator(api_keys, cookies)

    generator.process_new_dms()
    generator.process_new_tweets()