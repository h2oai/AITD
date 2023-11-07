
# Abuse in Transaction Description Detection (AITD)


Family and domestic violence is a critical global issue that affects millions, with the WHO reporting 1 in 3 women and 1 in 13 men worldwide have experiences physical violence in their lifetime. In Australia alone around 3.6 million people have experienced emotional abuse from a partner [[1]](#references).

Technology-assisted abuse has been identified within financial institutions, where real-time transactions, typically of low value, were used as a form of communication. In some instances the communication was a targeted form of domestic and family violence. To tackle this problem, [Commonwealth Bank of Australia](https://www.commbank.com.au) (CommBank) experts have created novel AI methods to detect abuse in transactions and embedded it in their internal processes [[5]](https://www.commbank.com.au/articles/newsroom/2021/10/artificial-intelligence-detects-abusive-behaviour.html). The model is fully operational, detecting more than 1,500 cases per year [[2]](https://www.commbank.com.au/articles/newsroom/2023/08/next-chapter-nsw-police-pilot.html). 

Other financial institutions can benefit by reusing this approach and therefore CommBank collaborated with [H2O.ai](https://h2o.ai/) to create a version of the end-to-end pipeline and the model publicly. This is an open-source repository to ease adoption of abusive case detection for other organisations. The approach uses a combination of Machine Learning (ML), Natural Language Processing (NLP), pre-trained Large Language Models (LLM) on public data, text analysis, and elements of graph concepts to identify abusive relationships. The model was trained using labelled data of verified cases of high-risk abuse from CommBank. For a quick overview checkout the [H2O world presentation](https://www.youtube.com/watch?v=ScrsdEBW1gk) [[3]](#references) and more thorough overview of methodology please refer to CommBank's paper [Detection of Abuse in Financial Transaction Descriptions Using Machine Learning](https://doi.org/10.48550/arXiv.2303.08016). [[4]](#references)


### Getting Started

The AITD pipeline has been tested on Linux and Mac based systems using CPU only and a combination of CPU and GPU. We do recommend running the feature transformation pipeline with GPUs to speed up the pipeline. More details on runtime is found below.

#### Recommended Setup

The recommended way to install AITD is using conda environment with Python 3.10 and download the required models. For ease of setup you can run:

```
# supports python 3.8/3.9/3.10
bash ./setup.sh
```

This bash script will create a virtual environment called **aitd** with all the dependencies installed. It will also download all of the models needed to run the functions.

Note: In order to run the H2O model, you will also need to satisfies the [Java requirements](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#java-requirements)

 #### Docker Setup

Alternatively, we have also created a Dockerfile. To build the docker container you could use the provided Dockerfile and run:

```
docker build -t aitd . && docker run -it aitd bash
```

#### Running the pipeline

To run the AITD pipeline and score the model with your own data, you need to ensure your dataset contains the following columns named as listed.
- tx_description: transaction description
- sender_id: the identification of the sender who initiated the transaction, e.g. customer id
- receiver_id: The identification of the source who received the transaction, e.g. customer id 
- tx_date: the date of the transaction
- amount: transaction amount

If you want to score one month of data (the trasactions between the first and the last day of the corresponding month), you need to use model ```CBA_AITD_Short```. If your data span on three consecutive months, you need to use the model ```CBA_AITD_Long```. See [Model card](#model-card) for more detail.

Please note: The current pipeline is built to score a month worth of data at a time. By default it will take the latest month as the score month. To ensure the feature set is properly created, please specify the score month in the transform function (see [end_to_end.ipynb](end_to_end.ipynb)).
 
##### 1. Jupyter Notebook

We have provided an example notebook [end_to_end.ipynb](end_to_end.ipynb) and an [example dataset](data/example_input.csv). You can also use this notebook to run the pipeline with your own data, all you need is to  replace the example.csv file with your own data and ensure that the column names are consistent with the above.

##### 2. Running end_to_end.py

You can also run the whole pipeline by running the ```end_to_end.py``` function. To run you need to pass the input dataset (saved as a csv) and the output location as shown below.

```
conda activate aitd
python src/end_to_end.py -i <input_file> -o <output_file> -l <lag (optional), default=2> -sm <score_month (optional), default=None> 
```

Using the example dataset, run:

```CBA_AITD_Long```: Default Model
```
conda activate aitd
python end_to_end.py -i data/example_input.csv -o data/example_input_output.csv -m models/CBA_AITD_Long.zip 
```

```CBA_AITD_Short```:  You need to pass -m models/CBA_AITD_Short.zip and -l 0 
```
conda activate aitd
python end_to_end.py -i data/example_input.csv -o data/example_input_output.csv -m models/CBA_AITD_Short.zip -l 0
```

### Model card
There are two types of pre-trained models provided with this package located in [/models](/models):

  - ```CBA_AITD_Long```: The model is trained on 3 months of transactions and aims to detect prolonged high-risk abuse. 
  - ```CBA_AITD_Short``` : The model is trained on 1 month of transactions and aims to detect short-term abuse. 

For full details of these models please see [[4]](#references).

### Custom Datasets Scoring

#### Data preparation
To complete this process, get your transaction data cleaned and organise the dataset in the following format:

Schema: | sender_id | receiver_id | transaction date | amount | transaction description |

#### Feature Extraction 

The feature extraction follows these steps:

  1. Generate NLP Features at the transaction level. It is executed by ```transform``` function in [transfer_predict.py](src/transform_predict.py). These features include:

        - Simple Text Features: length of the transaction description, number of words, case-based features, punctuation based features, the proportion of words to length of the description, whether numbers are present.
        - Toxicity Features: Using the detoxify BERT-based model to get model scores for each transaction for toxicity, severe toxicity, obscene language, identity attack, insult, threat, sexually explicit.
        - Emotion Features: Using the emotion detection BERT-based model to determine the emotion of the transaction description which can be either anger, fear, love, neutral, sadness, surprise, joy.
        - Vadar Sentiment: Using NLTK Sentiment model VADAR (Valence Aware Dictionary for Sentiment Reasoning) to determine the sentiment (polarity and intensity of the emotion) of the transaction description, which can be either positive, negative, compound, and neutral.

  2. Aggregate the NLP features to a relationship level (between a sender and a receiver) on a monthly basis. A range of different aggregations (mean, min, max, median etc) are used. Transaction and value based features also will be aggregated at this level.

  3. Include the aggregated features that are calculated on the reciprocal response (replies of a relationship).
   
  4. Add the Aggregated features for the previous two months, if you are using ```CBA_AITD_Long```.
   

### Model Training

This library does not yet provide means for model training and uses the pre-trained model to calculate the probabilities on the customer dataset on how much abusive they are.


### Pipeline for Scoring Abusive Cases

The following image shows the end-to-end pipeline for scoring cases of abuse in transaction description.

<img src="img/workflow.png" width="500" height="300" class="center"/>
<!-- ![alt text]( "App Framework") -->


#### Pipeline and Model Performance

The performance of the models on the out-of-sample test set of the transaction descriptions collected within one month, which no overlap of this month of data with our training dataset, are as shown in the following tables.

For the ```CBA_AITD_Long``` model, tested on features with 3-months of data.

|Prec |Rec  |F1   |AUC-PR|ROC AUC|
|-    |-    |-    |-     |-      |
|0.678|0.738|0.703|0.570 |0.800  |

For the ```CBA_AITD_Short``` model, tested on features with one month of data.

|Prec |Rec  |F1   |AUC-PR|ROC AUC|
|-    |-    |-    |-     |-      |
|0.680|0.713|0.689|0.552 |0.790  | 
 
Runtime using CPU and GPU.
  
|Hardware|No of transactions|Processing time|
|-       |-                 | -             |
|CPU     | 100,000          |12 hours       |
|GPU     | 100,000          |20 minutes     |
|GPU     | 1,000,000        |3 hours 40 min |
 
### Known Issues & Release Notes
- Currently this tool works within the boundaries of a calendar month. The future releases will incorporate different start and end dates.

### References
  
  [1] Australian Institute of Health and Welfare. Family, domestic and sexual violence. Canberra: Australian Institute of Health and Welfare, 2023. Available from: [https://www.aihw.gov.au/reports/domestic-violence/family-domestic-and-sexual-violence](https://www.aihw.gov.au/reports/domestic-violence/family-domestic-and-sexual-violence)
  
  [2] Commonwealthbank of Autralia. CBA to launch police referral pilot in NSW to address technology-facilitated abuse, August 2023. Available from: [https://www.commbank.com.au/articles/newsroom/2023/08/next-chapter-nsw-police-pilot.html](https://www.commbank.com.au/articles/newsroom/2023/08/next-chapter-nsw-police-pilot.html)
  
  [3] Pizzato, L., Bansal, S., and Richards, G. Fighting Transactions Abuse Using Artificial Intelligence at H2O World Sydney 2022 - 14 Nov 2022. [https://youtu.be/ScrsdEBW1gk](https://youtu.be/ScrsdEBW1gk)
  
  [4] Leontjeva, A., Richards, G., Sriskandaraja, K., Perchman, J. and Pizzato, L. Detection of Abuse in Financial Transaction Descriptions Using Machine Learning. arXiv preprint arXiv:2303.08016. 10 Mar 2023. [https://doi.org/10.48550/arXiv.2303.08016](https://doi.org/10.48550/arXiv.2303.08016)

  [5] Commonwealthbank of Autralia. CBA introduces new artificial intelligence technology to detect abusive behaviour, October 2021. Available from: [https://www.commbank.com.au/articles/newsroom/2021/10/artificial-intelligence-detects-abusive-behaviour.html](https://www.commbank.com.au/articles/newsroom/2021/10/artificial-intelligence-detects-abusive-behaviour.html)


### Licence Information
Apache 2.0

### Version
0.1.0

### Contacts for support? Raise a github issue?
To raise any feedback or bugs, please raise a github issue for the repo or raise a ticket by sending an email to support@h2o.ai with the necessary details.

### FAQ
To be filled based on feedback.
