# Normalization Pipeline

Content below from the appendix of [our paper](TODO fill in this link when online).

Real world task oriented dialogue systems can interface users with thousands or more entities, such as restaurants
or hotels in MultiWOZ. Since reasoning directly over all such entities is intractable, dialogue understanding modules
often first predict a surface form (e.g. a restaurant name mentioned by a user) which another module links to a 
canonical form (e.g. that restaurants name in a database). While dialogue state trackers evaluated on MultiWOZ do not
need to interact with a database, handling of typos and unexpected surface forms is important for a realistic 
assessment of system performance, since predictions for a slot are evaluated on exact string match.

As such, most research systems including the baselines in this paper use rule-based functions to fix typos and 
unexpected surface forms. We propose a robust rule-based method for effective linking of surface forms to canonical 
forms described below.

## Mapping to canonical forms (`S" -> C`)
We begin by first reading in canonical forms for every informable slot in the MultiWOZ system. For categorical slots, 
these are defined in a schema file, as released with MultiWOZ 2.1, and included in the 
[main MultiWOZ repo]((https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.2/schema.json)). 
For non-categorical slots, we read in values from the 
[database defined with the original MultiWOZ data collection](https://github.com/budzianowski/multiwoz/tree/master/db). 
Neither source of information contains dialogue data, only information defining the task. The taxi and train service 
have informable slots for departure and destination locations. In addition to the locations listed for these slots in a 
database (i.e. scheduled train journeys), we accept the name of any entity which has an address as a canonical form 
for these slots. For time slots we consider any time represented in "hh:mm" form as canonical. Overall, this gives us 
a mapping from a slot name $s_i$ to a set of canonical forms $\mathcal{C_i}$ for all slot names. 

Given a slot name $s_i$ and a slot value surface form $v_j$, we select the correct canonical form $c_j$ as follows: 

1. we first generate a set of aliases for $v_j$. These are acceptable re-phrasings of $v_j$, such as adding the leading 
article "the", a domain specifying suffix such as "hotel" or "museum", or switching numbers to/from digit form 
(e.g. "one" $\leftrightarrow$ "1"). 
2. We then consider a surface form $v_j$ as mapped to a canonical form $c_j$ if any of the aliases $a_j \in A_j$ is a 
fuzzy match for the canonical form $c_j$, using the `fuzz.ratio` scorer in the 
[`fuzzywuzzy`](https://pypi.org/project/fuzzywuzzy/) package. We require a score of 90 or higher, and verify in the 
development data that no surface form maps to more than one canonical form. 

## Choosing the most likely surface form (`C -> S'`) 

While in a real world dialogue system we would only need to link to canonical forms, 
**gold dialogue state states in MultiWOZ are themselves annotated with surface forms**, not always matching the name of 
the entity in the database and occasionally disagreeing on an entity name. So as to not alter the evaluation process 
and make sure we can fairly compare to prior work, we use the training data available in each experimental setting to 
choose the most likely surface form for a given canonical form $c_j$. To do this, we simply count the occurrences of 
each surface form in the gold labels of the training set for that experiment, and select the most frequently occurring 
one for $c_j$. However for low data regimes, we often do not observe all canonical forms. Following numerous prior 
works, we make use of the [ontology file](../db/multiwoz/2.4/ontology.json) released with the dataset, 
as processed by the IC-DST model, which lists all observed surface forms for a slot name, and treat each of these as 
if we had seen them 10 times. This serves as a smoothing factor for selecting the most likely surface form. For the 
zero-shot experiments, we use only the counts derived from the ontology file, as we have no training data to observe.

Overall, we find this approach to normalization to be robust when compared to other works, [which rely on hard-coded 
fixes for commonly observed typos](../utils/ic_dst_typo_fix.py). Further, our normalization can be initialized 
with any similarly formatted system definition and data set, allowing for use in other domains.

To verify that our approach to normalization is not the key factor distinguishing our performance from previous methods,
we apply it to a faithful re-implementation of our [IC-DST Codex baseline](https://arxiv.org/abs/2203.08568) in our 
ablation in [Table 4](https://aclanthology.org/2023.findings-acl.344.pdf).
