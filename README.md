# snapshot-reflect
SnapshotReflect is a conversational twitter bot, currently tweeting from the account @SnapshotReflect.

`tweetbot.py` contains all of the source. Right now it runs on a regular cron job to send tweets. You will need an api_keys.json file containing keys for the Microsoft Computer Vision API and a Twitter account in order for it to run properly. To succesfully respond to DMs requires session cookies in a cookies.json file, because cookies are required in order to download media associated with DMs.
