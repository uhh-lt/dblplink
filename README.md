
# DBLP Entity Linker 

Demo is hosted on https://ltdemos.informatik.uni-hamburg.de/dblplink.

Download pre-trained models from http://ltdata1.informatik.uni-hamburg.de/debayan-dblplink/ and untar them in ```api/```.

### Docker

```
docker compose build
docker compose up
```
At the moment, the Elasticsearch dump for the KG embeddings exceeds 100 GB in size, hence we do not upload it in public. Kindly contact us if you wish to receive the dumps individually. 

### Training label span detector:

```
cd extras/
python -u train.py --model_name t5-small --batch_size 4 --epochs 10 --lr 0.001
```
### Evaluating label span detector:

```
cd extras/
python -u evaluate.py --model_name t5-base --embedding_name transe
```

### Training re-ranker model

```
cd reranker/
python siamese.py data_distmult/train.jsonlines data_distmult/valid.jsonlines output_model_dir/
```

### Running Streamlit UI

```
streamlit run Home.py staging

OR

streamlit run Home.py production
```

### Citation
```
@inproceedings{10.1145/3448016.3457280,
	author = {Banerjee, Debayan and , Arefa and Usbeck, Ricardo and Biemann, Chris},
	title = {DBLPLink: An Entity Linker for the DBLP Scholarly Knowledge Graph},
	year = {2023},
	booktitle = {The Semantic Web – ISWC 2023: 22nd International Semantic Web Conference, Athens, Greece, November 6–10, 2023, Proceedings},
	location = {Athens, Greece}
}
```
