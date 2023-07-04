# MultiWOZ DB Files

All `_db.json` files are the files located in `db/` as distributed with the original MultiWOZ repository:
[https://github.com/budzianowski/multiwoz/tree/master/db](https://github.com/budzianowski/multiwoz/tree/master/db)

`schema.json` is distributed in the original repo for 2.2 onwards, at 
[https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.2/schema.json](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.2/schema.json)

`2.4/ontology.json` is a dictionary from slot names to possible values, and is derived from the process 
described in [YushiHu/IC-DST](https://github.com/Yushi-Hu/IC-DST#data)

To re-create:
```bash
git clone https://github.com/Yushi-Hu/IC-DST.git
cd IC-DST
pip install -r requirements.txt
cd data
python create_data.py --main_dir mwz21 --mwz_ver 2.1 --target_path mwz2.1  # for MultiWOZ 2.1
python create_data.py --main_dir mwz24 --mwz_ver 2.4 --target_path mwz2.4  # for MultiWOZ 2.4
```
`2.1/ontology.json` and `2.4/ontology.json` will be in `mwz2.1/ontology.json` and `mwz2.4/ontology.json` respectively.
The files may not be identical due to arbitrary list ordering in the processing, but the contents are otherwise the 
same.