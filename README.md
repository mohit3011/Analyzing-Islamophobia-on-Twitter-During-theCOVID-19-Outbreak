# Analyzing-Islamophobia-on-Twitter-During-theCOVID-19-Outbreak
The code repository for our paper "“A Virus Has No Religion”: Analyzing Islamophobia on Twitter During theCOVID-19 Outbreak" accepted at 32nd ACM Conference on Hypertext and Social Media (HT'21)

### Dataset Information
------

`TweetIDs.tsv` contains the tweet IDs from the CoronaIslam dataset.


### Topic Modelling
------

The directory topic_modelling_files contains 3 topic modelling scripts in python.
- `nmfmacro.py` - Does Macro topic modelling for the entire dataset. The variable `directorylist` must be modified to the directory structure used by the user. `python nmfmacro.py` can be used to run the program. The output includes `topics.csv` and `topicstweets.csv` which contain the topics and tweets under each topic respectively.
- `nmfmicro1.py` - Does micro topic modelling for subsets of the dataset. The variable `directorylist` must be modified to the directory structure used by the user. The subset of the data can also be chosen by modifying the date variables. `python nmfmicro1.py` can be used to run the program. The output includes `topics.csv` and `topicstweets.csv` which contain the topics and tweets under each topic respectively.
- `nmfmicro2.py`- Does micro topic modelling for subsets of the dataset. The variable `directorylist` must be modified to the directory structure used by the user. The subset of the data can also be chosen by modifying the date variables. `python nmfmicro2.py` can be used to run the program. The output includes `topics.csv` and `topicstweets.csv` which contain the topics and tweets under each topic respectively.
