# Assignment 3: Work Vectorization

### Nanda Rajiv | 2021115002


<br/>
As mentioned in the assignment document, the followed have been submitted with this zip file:

+ Source Code 
    + `svd.py`: script to get the SVD-based embeddings
    + `svd-classification.py`: performs the downstream task of new classification using the embeddings from SVD
    + `skip-gram.py`: model to train skip-gram embeddings
    + `skip-gram-classification.py`: performs the downstream task of new classification using the embeddings from skip-gram
    + `report.pdf`: contains the report for the assignment
    + `README.md`

The model weights are word vectors exceed the Moodle submission limit, and have been uploaded to OneDrive, which can be accessed [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/nanda_rajiv_research_iiit_ac_in/EgzAAIe2PKtMoYOLuln2dlAB49VCMrkCaec2KtzG6Yu9Bw?e=Pq6b1H). The link contains:
+ `svd-word-vectors.pt`: word vectors obtained from SVD
+ `skip-gram-word-vectors.pt`: word vectors obtained from skip-gram
+ `svd-classification-model.pt`: model weights for the classification task using SVD embeddings
+ `skip-gram-classification-model.pt`: model weights for the classification task using skip-gram embeddings


-----
Note: Please maintain the file structure of this submission. Also, ensure the Dataset is provided in the same format. It should be saved in this folder as:
+ ANLP-2
    + train.csv
    + test.csv

-----------

### Running the code

The code can be run using the following commands:

```bash
> python svd.py
> python svd-classification.py
> python skip-gram.py
> python skip-gram-classification.py
```



--------