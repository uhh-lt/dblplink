import streamlit as st

body = ''' 
### DBLPLink API Usage 

DBLPLink is also accessible via an API. A sample API call looks like this: 

    curl -X POST -H "Content-Type: application/json" -d '{"question": "where does debayan banerjee work?"}' 
    https://ltdemos.informatik.uni-hamburg.de/dblplinkapi/api/entitylinker/t5-small/distmult


The last two parts of the endpoint URL are the label generator model and the embedding re-ranker models, respectively. The following options exist for each of them:

#### Label and Type Generator
-   `t5-small`
-   `t5-base`

#### Embedding Re-ranker

-   `nosort`: The labels of the candidates in the previous step are already arranged in text sorting order as returned by Elasticsearch. No further re-ranking is performed.
-   `transe`: If more than one entity has the same label in the candidates, a further embedding-based re-ranking is performed using the `transe` embedding.
-   `complex`: If more than one entity has the same label in the candidates, a further embedding-based re-ranking is performed using the `complex` embedding.
-   `distmult`: If more than one entity has the same label in the candidates, a further embedding-based re-ranking is performed using the `distmult` embedding.
-   `transe-pure`: Embedding-based re-ranking is performed always, ignoring text sorting order.
-   `complex-pure`: Embedding-based re-ranking is performed always, ignoring text sorting order.
-   `distmult-pure`: Embedding-based re-ranking is performed always, ignoring text sorting order.
'''


st.markdown(body)
